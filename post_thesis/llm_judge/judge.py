"""One asynchronous judge attempt using an injected backend; no API integration.

Retries, persistence, budget enforcement, and dataset adapters belong to the
future runner. Backend adapters must not silently retry or switch models.
"""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal, Protocol

from .parse import JudgeParseError, parse_score
from .prompts import DEVELOPMENT_PROMPT, PromptSpec, canonical_json, content_hash

PROTOCOL_ID = "post_thesis_llm_judge_v1"
CONTRACT_VERSION = "judge-request-v1"
RESPONSE_SCHEMA_JSON = canonical_json({
    "type": "object",
    "properties": {"unsupported_probability": {
        "type": "number", "minimum": 0, "maximum": 1,
    }},
    "required": ["unsupported_probability"],
    "additionalProperties": False,
})


@dataclass(frozen=True)
class JudgeInput:
    answer: str
    context: str

    def __post_init__(self):
        if not isinstance(self.answer, str) or not isinstance(self.context, str):
            raise ValueError("answer and context must be strings")
        # Preserve exact text; an empty context is valid absence of evidence.
        if not self.answer.strip():
            raise ValueError("answer must contain text")


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: str
    api_config_id: str
    temperature: float = 0.0
    max_output_tokens: int = 128

    def __post_init__(self):
        for value in (self.provider, self.model, self.api_config_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("provider, model and api_config_id must be nonempty")
        if (type(self.temperature) not in (int, float)
                or not 0 <= self.temperature < float("inf")):
            raise ValueError("temperature must be finite and nonnegative")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class JudgeRequest:
    config: JudgeConfig
    messages: tuple[Message, ...]
    prompt_version: str
    prompt_sha256: str
    response_schema_json: str = RESPONSE_SCHEMA_JSON
    protocol_id: str = PROTOCOL_ID
    contract_version: str = CONTRACT_VERSION

    @property
    def key(self) -> str:
        return content_hash(asdict(self))


def build_request(item: JudgeInput, config: JudgeConfig,
                  prompt: PromptSpec = DEVELOPMENT_PROMPT) -> JudgeRequest:
    if not isinstance(item, JudgeInput):
        raise TypeError("item must be JudgeInput; metadata is not accepted")
    if not isinstance(config, JudgeConfig) or not isinstance(prompt, PromptSpec):
        raise TypeError("expected JudgeConfig and PromptSpec")
    return JudgeRequest(
        config=config,
        messages=(Message("system", prompt.system_text), Message("user", canonical_json({
            "answer": item.answer, "context": item.context,
        }))),
        prompt_version=prompt.version, prompt_sha256=prompt.sha256,
    )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("token counts must be nonnegative integers or None")


@dataclass(frozen=True)
class BackendResponse:
    text: str
    returned_model: str | None = None
    usage: TokenUsage = TokenUsage()
    outcome: Literal["completed", "refused", "incomplete"] = "completed"
    response_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not isinstance(self.usage, TokenUsage):
            raise TypeError("backend must return text and TokenUsage")
        if self.outcome not in ("completed", "refused", "incomplete"):
            raise ValueError("unrecognized backend outcome")
        for value in (self.returned_model, self.response_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("response identifiers must be nonempty or None")


class BackendError(Exception):
    """Adapter-normalized operational failure; never wrap programming errors.

    Use a sanitized code (e.g. timeout), not the raw provider error message.
    Usage may be unknown even if the provider charged for a failed attempt.
    """

    def __init__(self, code: str, *, usage: TokenUsage = TokenUsage()):
        if (not isinstance(code, str) or not code or len(code) > 64
                or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in code)):
            raise ValueError("backend error code must be a short lowercase identifier")
        if not isinstance(usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        self.code = code
        self.usage = usage
        super().__init__(code)


class JudgeBackend(Protocol):
    async def complete(self, request: JudgeRequest) -> BackendResponse:
        """Execute one attempt, honoring the exact input/config/schema.

        Normalize refusals and truncation into outcome; do not silently truncate
        inputs, add prompt text, retry, substitute a model, or invent usage.
        """
        ...


@dataclass(frozen=True)
class JudgeResult:
    request: JudgeRequest
    status: Literal["ok", "invalid_output", "refused", "incomplete", "backend_error"]
    unsupported_probability: float | None
    error_code: str | None
    response: BackendResponse | None
    usage: TokenUsage
    latency_seconds: float
    study_stage: str = "post_thesis"


async def judge_once(item: JudgeInput, *, config: JudgeConfig, backend: JudgeBackend,
                     prompt: PromptSpec = DEVELOPMENT_PROMPT) -> JudgeResult:
    """Score one attempt. Invalid input raises before invoking the backend.

    Operational failures return no-score results. Programming errors and task
    cancellation propagate. No retries, fallback scores, or shared model state.
    """
    request = build_request(item, config, prompt)
    start = perf_counter()
    try:
        response = await backend.complete(request)
    except BackendError as error:
        return JudgeResult(request, "backend_error", None, error.code, None,
                           error.usage, perf_counter() - start)
    latency = perf_counter() - start
    if not isinstance(response, BackendResponse):
        raise TypeError("backend must return BackendResponse")
    if response.outcome != "completed":
        return JudgeResult(request, response.outcome, None, response.outcome,
                           response, response.usage, latency)
    try:
        score = parse_score(response.text)
    except JudgeParseError as error:
        return JudgeResult(request, "invalid_output", None, error.code,
                           response, response.usage, latency)
    return JudgeResult(request, "ok", score, None, response, response.usage, latency)
