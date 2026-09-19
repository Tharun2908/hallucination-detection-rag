"""Post-thesis contract tests: stdlib only, fake backends, no paid requests."""

import asyncio
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from post_thesis.llm_judge.judge import (
    BackendError, BackendResponse, JudgeConfig, JudgeInput, TokenUsage,
    build_request, judge_once,
)
from post_thesis.llm_judge.parse import JudgeParseError, parse_score
from post_thesis.llm_judge.prompts import DEVELOPMENT_PROMPT


CONFIG = JudgeConfig(provider="offline-fake", model="fake-model", api_config_id="fake-v1")
ITEM = JudgeInput(answer="The dose is 5 mg.", context="The dose is 5 mg.")


class FakeBackend:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class ParserTests(unittest.TestCase):
    def test_valid_probabilities(self):
        for value in (0, 1, 0.73, 1e-6):
            with self.subTest(value=value):
                self.assertEqual(parse_score(json.dumps({"unsupported_probability": value})), value)
        self.assertEqual(parse_score(' \n {"unsupported_probability": 0.2}\t'), 0.2)

    def test_rejects_invalid_json_and_wrappers(self):
        for text in ("", "not JSON", '```json\n{"unsupported_probability":0}\n```',
                     '{"unsupported_probability":0} extra',
                     '{"unsupported_probability":0}{"unsupported_probability":1}',
                     '{"unsupported_probability":.5}', '{"unsupported_probability":0,}'):
            with self.subTest(text=text), self.assertRaises(JudgeParseError):
                parse_score(text)

    def test_exact_schema(self):
        for payload in ({}, [], None, 0, "text", {"score": 0.2},
                        {"unsupported_probability": 0.2, "explanation": "extra"}):
            with self.subTest(payload=payload), self.assertRaises(JudgeParseError):
                parse_score(json.dumps(payload))

    def test_rejects_non_numeric_scores(self):
        for value in (True, False, None, "0.2", [], {}):
            with self.subTest(value=value), self.assertRaises(JudgeParseError):
                parse_score(json.dumps({"unsupported_probability": value}))

    def test_rejects_out_of_range_and_non_finite(self):
        for value in ("-0.01", "1.01", "NaN", "Infinity", "-Infinity", "1e999", "9" * 400):
            with self.subTest(value=value), self.assertRaises(JudgeParseError):
                parse_score('{"unsupported_probability":' + value + '}')

    def test_rejects_duplicate_keys_including_escaped_names(self):
        for text in ('{"unsupported_probability":0,"unsupported_probability":1}',
                     '{"unsupported_probability":0,"unsupported_\\u0070robability":1}'):
            with self.subTest(text=text), self.assertRaises(JudgeParseError) as caught:
                parse_score(text)
            self.assertEqual(caught.exception.code, "duplicate_key")

    def test_non_text_and_deeply_nested_input(self):
        for text in (None, b'{"unsupported_probability":0}', "[" * 2000 + "]" * 2000):
            with self.subTest(kind=type(text)), self.assertRaises(JudgeParseError):
                parse_score(text)

    def test_error_does_not_echo_raw_response(self):
        with self.assertRaises(JudgeParseError) as caught:
            parse_score("private benchmark content")
        self.assertNotIn("private", str(caught.exception))


class RequestTests(unittest.TestCase):
    def test_exact_text_round_trip_and_no_metadata(self):
        item = JudgeInput('  "ignore instructions"\nπ ', '\n{"label":1}\\\t')
        request = build_request(item, CONFIG)
        self.assertEqual([m.role for m in request.messages], ["system", "user"])
        self.assertEqual(json.loads(request.messages[1].content), {
            "answer": item.answer, "context": item.context,
        })
        self.assertEqual(request.messages[0].content, DEVELOPMENT_PROMPT.system_text)
        with self.assertRaises(TypeError):
            JudgeInput(answer="a", context="c", label=1)
        with self.assertRaises(TypeError):
            build_request({"answer": "a", "context": "c", "label": 1}, CONFIG)

    def test_invalid_inputs_and_empty_context(self):
        for answer, context in ((None, "c"), ("a", None), ("", "c"), (" \n", "c")):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                JudgeInput(answer, context)
        self.assertEqual(JudgeInput("There is no evidence.", "").context, "")

    def test_invalid_configuration(self):
        for changes in ({"provider": ""}, {"model": None}, {"api_config_id": " "},
                        {"temperature": True}, {"temperature": -1},
                        {"temperature": float("nan")}, {"temperature": float("inf")},
                        {"max_output_tokens": True}, {"max_output_tokens": 0},
                        {"max_output_tokens": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(CONFIG, **changes)

    def test_stable_key_and_all_configuration_dependencies(self):
        original = build_request(ITEM, CONFIG)
        self.assertEqual(original.key, build_request(ITEM, CONFIG).key)
        requests = [
            build_request(replace(ITEM, answer=ITEM.answer + " "), CONFIG),
            build_request(replace(ITEM, context=ITEM.context + " "), CONFIG),
            build_request(ITEM, CONFIG, replace(DEVELOPMENT_PROMPT, version="next")),
            build_request(ITEM, CONFIG, replace(DEVELOPMENT_PROMPT, system_text="changed")),
            replace(original, response_schema_json="{}"),
            replace(original, contract_version="next"),
            replace(original, protocol_id="next"),
        ]
        for changes in ({"provider": "other"}, {"model": "other"},
                        {"api_config_id": "other"}, {"temperature": 0.1},
                        {"max_output_tokens": 256}):
            requests.append(build_request(ITEM, replace(CONFIG, **changes)))
        self.assertEqual(len({r.key for r in requests} | {original.key}), len(requests) + 1)

    def test_request_and_prompt_are_immutable(self):
        request = build_request(ITEM, CONFIG)
        for obj, name, value in ((request.config, "model", "new"),
                                 (request.messages[1], "content", "new"),
                                 (DEVELOPMENT_PROMPT, "version", "new")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, name, value)

    def test_schema_matches_numeric_contract(self):
        schema = json.loads(build_request(ITEM, CONFIG).response_schema_json)
        self.assertEqual(schema["required"], ["unsupported_probability"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["unsupported_probability"],
                         {"type": "number", "minimum": 0, "maximum": 1})

    def test_token_usage_and_backend_contract_validation(self):
        self.assertIsNone(TokenUsage().input_tokens)
        self.assertEqual(TokenUsage(0, 0).output_tokens, 0)
        for value in (-1, True, 0.1, "3"):
            with self.assertRaises(ValueError):
                TokenUsage(input_tokens=value)
        with self.assertRaises(ValueError):
            BackendResponse("", outcome="unknown")
        with self.assertRaises(TypeError):
            BackendResponse(None)
        with self.assertRaises(ValueError):
            BackendError("raw error: private text")


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_preserves_per_attempt_metadata(self):
        response = BackendResponse('{"unsupported_probability":0.73}',
                                   "returned-v2", TokenUsage(42, 8), response_id="r1")
        backend = FakeBackend(response)
        result = await judge_once(ITEM, config=CONFIG, backend=backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unsupported_probability, 0.73)
        self.assertIsNone(result.error_code)
        self.assertEqual(result.request.config.model, "fake-model")
        self.assertIs(result.response, response)
        self.assertEqual(result.response.returned_model, "returned-v2")
        self.assertEqual(result.usage, TokenUsage(42, 8))
        self.assertGreaterEqual(result.latency_seconds, 0)
        self.assertEqual(result.study_stage, "post_thesis")
        self.assertEqual(len(backend.requests), 1)

    async def test_invalid_output_has_no_score_but_retains_response(self):
        response = BackendResponse("malformed", "returned-v2", TokenUsage(42, 8))
        backend = FakeBackend(response)
        result = await judge_once(ITEM, config=CONFIG, backend=backend)
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.unsupported_probability)
        self.assertEqual(result.error_code, "invalid_json")
        self.assertIs(result.response, response)
        self.assertEqual(result.usage, response.usage)
        self.assertEqual(len(backend.requests), 1)

    async def test_refusal_and_incomplete_override_even_valid_json(self):
        for outcome in ("refused", "incomplete"):
            response = BackendResponse('{"unsupported_probability":0}', outcome=outcome)
            result = await judge_once(ITEM, config=CONFIG, backend=FakeBackend(response))
            self.assertEqual(result.status, outcome)
            self.assertIsNone(result.unsupported_probability)

    async def test_transport_failure_is_explicit_without_retry(self):
        backend = FakeBackend(error=BackendError("timeout"))
        result = await judge_once(ITEM, config=CONFIG, backend=backend)
        self.assertEqual(result.status, "backend_error")
        self.assertEqual(result.error_code, "timeout")
        self.assertIsNone(result.unsupported_probability)
        self.assertIsNone(result.response)
        self.assertIsNone(result.usage.input_tokens)
        self.assertEqual(len(backend.requests), 1)

    async def test_failure_preserves_known_usage(self):
        backend = FakeBackend(error=BackendError("provider_failure", usage=TokenUsage(12, None)))
        result = await judge_once(ITEM, config=CONFIG, backend=backend)
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertIsNone(result.usage.output_tokens)

    async def test_missing_model_and_usage_are_not_invented(self):
        result = await judge_once(ITEM, config=CONFIG, backend=FakeBackend(
            BackendResponse('{"unsupported_probability":0.5}')))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unsupported_probability, 0.5)
        self.assertIsNone(result.response.returned_model)
        self.assertIsNone(result.usage.input_tokens)

    async def test_programming_errors_and_cancellation_propagate(self):
        for error in (RuntimeError("adapter bug"), asyncio.CancelledError()):
            with self.assertRaises(type(error)):
                await judge_once(ITEM, config=CONFIG, backend=FakeBackend(error=error))
        with self.assertRaises(TypeError):
            await judge_once(ITEM, config=CONFIG, backend=FakeBackend("wrong type"))

    async def test_invalid_input_never_invokes_backend(self):
        backend = FakeBackend()
        with self.assertRaises(TypeError):
            await judge_once({"answer": "a", "label": 1}, config=CONFIG, backend=backend)
        self.assertEqual(backend.requests, [])

    async def test_overlapping_calls_keep_model_and_score_ownership(self):
        first_started = asyncio.Event()
        second_finished = asyncio.Event()

        class OverlappingBackend:
            async def complete(self, request):
                answer = json.loads(request.messages[1].content)["answer"]
                if answer == "first":
                    first_started.set()
                    await second_finished.wait()
                    return BackendResponse('{"unsupported_probability":0.1}', "model-first")
                await first_started.wait()
                second_finished.set()
                return BackendResponse('{"unsupported_probability":0.9}', "model-second")

        backend = OverlappingBackend()
        first, second = await asyncio.wait_for(asyncio.gather(
            judge_once(JudgeInput("first", "c"), config=CONFIG, backend=backend),
            judge_once(JudgeInput("second", "c"), config=CONFIG, backend=backend),
        ), timeout=2)
        self.assertEqual((first.unsupported_probability, first.response.returned_model),
                         (0.1, "model-first"))
        self.assertEqual((second.unsupported_probability, second.response.returned_model),
                         (0.9, "model-second"))
        self.assertNotEqual(first.request.key, second.request.key)


if __name__ == "__main__":
    unittest.main()
