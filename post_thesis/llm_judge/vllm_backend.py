"""Pinned self-hosted vLLM adapter. No downloads or connections during import."""

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from urllib.parse import urlsplit

from .judge import BackendError, BackendResponse, JudgeConfig, TokenUsage
from .prompts import content_hash

PROFILE_PATH = Path(__file__).with_name("configs") / "qwen3_32b_h200.json"
ADAPTER_VERSION = "vllm-http-v1"


def load_profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate response field")
        value[key] = item
    return value


def _constant(_):
    raise ValueError("non-finite response field")


def decode_object(raw):
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


class VLLMBackend:
    """Owns an async HTTP client; use as an async context manager.

    No hidden retries or redirects. API keys never enter request identities.
    The optional transport is for offline HTTP tests only.
    """

    def __init__(self, base_url="http://127.0.0.1:8000", *, api_key=None,
                 timeout_seconds=60.0, transport=None):
        import httpx
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("base_url must be a server origin without credentials or /v1")
        if (type(timeout_seconds) not in (int, float)
                or not 0 < timeout_seconds <= 60):
            raise ValueError("timeout_seconds must be in (0,60]")
        self._profile_json = json.dumps(load_profile(), sort_keys=True)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._httpx = httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # AsyncHTTPTransport defaults to zero retries. Do not use SDK defaults.
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                        timeout=timeout_seconds, follow_redirects=False,
                                        trust_env=False, transport=transport)

    @property
    def profile(self):
        return json.loads(self._profile_json)

    @property
    def config(self):
        p = self.profile
        identity = content_hash({"adapter": ADAPTER_VERSION, "profile": p,
                                 "base_url": self.base_url, "timeout_seconds": self.timeout_seconds})
        return JudgeConfig("self_hosted_vllm", p["served_model_name"], identity,
                           p["temperature"], p["max_output_tokens"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self._client.aclose()

    async def _request(self, method, path, payload=None):
        try:
            async with self._client.stream(method, path, json=payload) as response:
                if response.status_code != 200:
                    raise BackendError(f"http_{response.status_code}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 2_000_000:
                        raise BackendError("response_too_large")
            return decode_object(body)
        except self._httpx.TimeoutException:
            raise BackendError("timeout") from None
        except self._httpx.TransportError:
            raise BackendError("transport_error") from None
        except (ValueError, UnicodeError, RecursionError):
            raise BackendError("invalid_http_json") from None

    async def preflight(self):
        """Check advertised server version/model/length. Does not attest weights."""
        async def check():
            p = self.profile
            version = await self._request("GET", "/version")
            models = await self._request("GET", "/v1/models")
            if version.get("version") != p["vllm_version"]:
                raise BackendError("server_version_mismatch")
            rows = models.get("data")
            if (not isinstance(rows, list) or len(rows) != 1
                    or not isinstance(rows[0], dict)
                    or rows[0].get("id") != p["served_model_name"]
                    or rows[0].get("max_model_len") != p["max_model_len"]):
                raise BackendError("server_model_mismatch")
            return {"version": version, "models": models}
        return await self._bounded(check())

    async def _bounded(self, operation):
        try:
            return await asyncio.wait_for(operation, timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            raise BackendError("timeout") from None

    async def complete(self, request):
        if request.config != self.config:
            raise ValueError("request configuration does not match this pinned adapter")
        return await self._bounded(self._complete(request))

    async def complete_counted(self, request, *, expected_input_tokens):
        """Generate only if the exact input still matches its frozen token audit."""
        if request.config != self.config:
            raise ValueError("request configuration does not match this pinned adapter")
        if type(expected_input_tokens) is not int or expected_input_tokens < 1:
            raise ValueError("expected a positive audited input token count")
        response = await self._bounded(self._complete(request, expected_input_tokens))
        if (response.usage.output_tokens is not None
                and response.usage.output_tokens > request.config.max_output_tokens):
            raise BackendError("output_token_limit_exceeded", usage=response.usage)
        return response

    def _input_payload(self, request):
        return {"model": request.config.model,
                "messages": [asdict(m) for m in request.messages],
                "chat_template_kwargs": {"enable_thinking": False},
                "add_generation_prompt": True, "add_special_tokens": False}

    async def count_input_tokens(self, request):
        """Count the exact generation input, even if too long. Never generates."""
        if request.config != self.config:
            raise ValueError("request configuration does not match this pinned adapter")
        return await self._bounded(self._count_input_tokens(request))

    async def _count_input_tokens(self, request):
        tokens = await self._request("POST", "/tokenize", self._input_payload(request))
        count = tokens.get("count")
        if (type(count) is not int or count < 1
                or tokens.get("max_model_len") != self.profile["max_model_len"]):
            raise BackendError("invalid_token_count")
        return count

    async def _complete(self, request, expected_input_tokens=None):
        p = self.profile
        common = self._input_payload(request)
        count = await self._count_input_tokens(request)
        if expected_input_tokens is not None and count != expected_input_tokens:
            raise BackendError("audited_input_token_mismatch")
        if count + request.config.max_output_tokens > p["max_model_len"]:
            raise BackendError("input_too_long")
        payload = dict(common, temperature=request.config.temperature,
                       max_completion_tokens=request.config.max_output_tokens,
                       n=1, stream=False, seed=p["seed"], top_p=1.0, top_k=-1,
                       min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                       frequency_penalty=0.0,
                       response_format={"type": "json_schema", "json_schema": {
                           "name": "unsupported_probability", "strict": True,
                           "schema": json.loads(request.response_schema_json)}})
        response = await self._request("POST", "/v1/chat/completions", payload)
        return self._response(response, count)

    def _response(self, data, counted_input):
        usage = TokenUsage()
        try:
            raw_usage = data.get("usage")
            if raw_usage is not None:
                usage = TokenUsage(raw_usage.get("prompt_tokens"), raw_usage.get("completion_tokens"))
            if data.get("model") != self.config.model:
                raise BackendError("returned_model_mismatch", usage=usage)
            if usage.input_tokens is not None and usage.input_tokens != counted_input:
                raise BackendError("input_token_mismatch", usage=usage)
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            message = choice["message"]
            if message.get("role") != "assistant":
                raise ValueError
            text = message.get("content")
            refusal = message.get("refusal")
            if refusal is not None and not isinstance(refusal, str):
                raise ValueError
            if message.get("tool_calls") or message.get("reasoning") or message.get("reasoning_content"):
                raise BackendError("unexpected_output_mode", usage=usage)
            finish = choice.get("finish_reason")
            if refusal or finish == "content_filter":
                outcome = "refused"
            elif finish == "length":
                outcome = "incomplete"
            elif finish == "stop":
                outcome = "completed"
            else:
                raise ValueError
            if text is None and outcome != "completed":
                text = ""
            return BackendResponse(text=text, returned_model=data["model"], usage=usage,
                                   outcome=outcome, response_id=data.get("id"))
        except (KeyError, ValueError, TypeError, AttributeError):
            raise BackendError("invalid_completion_envelope", usage=usage) from None
