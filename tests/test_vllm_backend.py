"""Offline HTTP contract tests. No GPU, model weights, or real network calls."""

import asyncio
import json
import unittest

try:
    import httpx
except ModuleNotFoundError:
    httpx = None

from post_thesis.llm_judge.judge import BackendError, JudgeInput, build_request, judge_once
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

P = load_profile()
ITEM = JudgeInput('Evidence says "blue".', 'Evidence says "blue".')


def completion(**changes):
    result = {"id": "response-1", "model": P["served_model_name"],
              "usage": {"prompt_tokens": 30, "completion_tokens": 8},
              "choices": [{"finish_reason": "stop", "message": {
                  "role": "assistant", "content": '{"unsupported_probability":0.2}'}}]}
    result.update(changes)
    return result


@unittest.skipIf(httpx is None, "install requirements-client.txt for HTTP contract tests")
class BackendTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, final=None, **kwargs):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 30, "max_model_len": P["max_model_len"]})
            return httpx.Response(200, json=completion() if final is None else final)

        return VLLMBackend(transport=httpx.MockTransport(handler), **kwargs)

    async def test_exact_wire_payload_and_structured_output(self):
        async with self.adapter() as backend:
            result = await judge_once(ITEM, config=backend.config, backend=backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.usage.input_tokens, 30)
        self.assertEqual([r.url.path for r in self.requests], ["/tokenize", "/v1/chat/completions"])
        token_payload, wire = [json.loads(r.content) for r in self.requests]
        self.assertEqual(token_payload["messages"], wire["messages"])
        self.assertEqual(json.loads(wire["messages"][1]["content"]), {"answer": ITEM.answer, "context": ITEM.context})
        self.assertEqual(wire["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(wire["response_format"]["type"], "json_schema")
        self.assertTrue(wire["response_format"]["json_schema"]["strict"])
        self.assertEqual(wire["max_completion_tokens"], 128)
        self.assertNotIn("truncate_prompt_tokens", wire)
        self.assertNotIn("tools", wire)
        self.assertNotIn("sample_id", wire)

    async def test_preflight_version_and_model(self):
        def handler(request):
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": P["vllm_version"]})
            return httpx.Response(200, json={"data": [{"id": P["served_model_name"], "max_model_len": P["max_model_len"]}]})
        async with VLLMBackend(transport=httpx.MockTransport(handler)) as backend:
            result = await backend.preflight()
            self.assertEqual(result["version"]["version"], P["vllm_version"])

    async def test_preflight_mismatch(self):
        async with VLLMBackend(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"version": "wrong", "data": []}))) as backend:
            with self.assertRaises(BackendError) as error:
                await backend.preflight()
            self.assertEqual(error.exception.code, "server_version_mismatch")

    async def test_oversized_input_never_generates(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={"count": P["max_model_len"], "max_model_len": P["max_model_len"]})
        async with VLLMBackend(transport=httpx.MockTransport(handler)) as backend:
            result = await judge_once(ITEM, config=backend.config, backend=backend)
        self.assertEqual(result.error_code, "input_too_long")
        self.assertIsNone(result.unsupported_probability)
        self.assertEqual(calls, ["/tokenize"])

    async def test_bad_token_counts_never_generate(self):
        for value in (True, -1, 0, None):
            async with VLLMBackend(transport=httpx.MockTransport(lambda r: httpx.Response(
                    200, json={"count": value, "max_model_len": P["max_model_len"]}))) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
                self.assertEqual(result.error_code, "invalid_token_count")

    async def test_refused_incomplete_and_invalid_score(self):
        for finish, content, refusal, expected in (
            ("stop", None, "declined", "refused"),
            ("content_filter", None, None, "refused"),
            ("length", '{"unsupported_probability":0.2}', None, "incomplete"),
            ("stop", '```json {} ```', None, "invalid_output"),
        ):
            data = completion(choices=[{"finish_reason": finish, "message": {
                "role": "assistant", "content": content, "refusal": refusal}}])
            async with self.adapter(data) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.status, expected)
            self.assertIsNone(result.unsupported_probability)
            self.assertEqual(result.usage.output_tokens, 8)

    async def test_model_and_token_mismatch_preserve_usage(self):
        for data, expected in ((completion(model="different"), "returned_model_mismatch"),
                               (completion(usage={"prompt_tokens": 29, "completion_tokens": 8}), "input_token_mismatch")):
            async with self.adapter(data) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.error_code, expected)
            self.assertIsNone(result.unsupported_probability)
            self.assertEqual(result.usage.output_tokens, 8)

    async def test_missing_usage_stays_unknown(self):
        async with self.adapter(completion(usage=None)) as backend:
            result = await judge_once(ITEM, config=backend.config, backend=backend)
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.usage.input_tokens)

    async def test_malformed_envelopes(self):
        for data in (completion(choices=[]), completion(choices=[{}, {}]),
                     completion(usage={"prompt_tokens": True}),
                     completion(choices=[{"finish_reason": None, "message": {"role": "assistant"}}])):
            async with self.adapter(data) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.status, "backend_error")
            self.assertIsNone(result.unsupported_probability)

    async def test_reasoning_is_not_silently_discarded(self):
        data = completion()
        data["choices"][0]["message"]["reasoning"] = "unexpected thinking"
        async with self.adapter(data) as backend:
            result = await judge_once(ITEM, config=backend.config, backend=backend)
        self.assertEqual(result.error_code, "unexpected_output_mode")

    async def test_http_failures_no_redirect_no_retry_no_raw_error_echo(self):
        for code in (301, 401, 403, 429, 500):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(code, headers={"Location": "https://example.invalid"}, text="private data")
            async with VLLMBackend(transport=httpx.MockTransport(handler)) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.error_code, f"http_{code}")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("private", str(result))

    async def test_invalid_http_json_and_oversize(self):
        for raw in ('{"count":1,"count":2}', 'NaN', '[]', 'x' * 2_000_001):
            async with VLLMBackend(transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, content=raw))) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertIn(result.error_code, ("invalid_http_json", "response_too_large"))

    async def test_transport_timeout_and_cancellation(self):
        for error, expected in ((httpx.ConnectError("private"), "transport_error"),
                                (httpx.ReadTimeout("private"), "timeout")):
            def handler(request):
                raise error
            async with VLLMBackend(transport=httpx.MockTransport(handler)) as backend:
                result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.error_code, expected)
        async def block(request):
            await asyncio.Event().wait()
        async with VLLMBackend(timeout_seconds=0.01, transport=httpx.MockTransport(block)) as backend:
            result = await judge_once(ITEM, config=backend.config, backend=backend)
            self.assertEqual(result.error_code, "timeout")
        async def cancel(request):
            raise asyncio.CancelledError()
        async with VLLMBackend(transport=httpx.MockTransport(cancel)) as backend:
            with self.assertRaises(asyncio.CancelledError):
                await backend.complete(build_request(ITEM, backend.config))

    async def test_identity_includes_endpoint_timeout_but_excludes_api_key(self):
        async with self.adapter(api_key="secret-one") as first, self.adapter(api_key="secret-two") as second:
            self.assertEqual(first.config, second.config)
            self.assertNotIn("secret", str(first.config))
        async with self.adapter() as first, self.adapter(base_url="http://127.0.0.1:9000") as second:
            self.assertNotEqual(first.config, second.config)
            with self.assertRaises(ValueError):
                await second.complete(build_request(ITEM, first.config))
        async with self.adapter() as first, self.adapter(timeout_seconds=10) as second:
            self.assertNotEqual(first.config, second.config)

    async def test_invalid_endpoint_and_timeout(self):
        for url in ("http://user:password@localhost", "http://localhost/v1", "file:///tmp/server", "http://localhost?key=secret"):
            with self.assertRaises(ValueError):
                VLLMBackend(url)
        with self.assertRaises(ValueError):
            VLLMBackend(timeout_seconds=True)


if __name__ == "__main__":
    unittest.main()
