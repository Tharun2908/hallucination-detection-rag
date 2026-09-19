"""Separate binary diagnostic contract; never interprets verdicts as probabilities."""

from dataclasses import replace
import json

from .judge import build_request
from .parse import JudgeParseError, _reject_constant, _unique_object
from .prompts import DEVELOPMENT_PROMPT_V2, PromptSpec, canonical_json

# Preserve all v2 support criteria. Change only the task wording and final
# probability/output instructions; save the resolved prompt and hash in requests.
BINARY_PROMPT = PromptSpec(
    version="faithfulness-binary-diagnostic-v1",
    system_text=DEVELOPMENT_PROMPT_V2.system_text.split("\n\nReturn the probability of that response-level event.", 1)[0]
    .replace("The event to estimate is:", "The question to decide is:")
    .replace("the probability. A mostly supported answer", "the verdict. A mostly supported answer")
    + """

Return unsupported if at least one factual assertion is unsupported by or
contradicts the context. Return supported only if all factual assertions are
supported by the context, or the answer contains no factual assertions.
Return only one JSON object with exactly the key verdict and the string value
supported or unsupported. Do not include probabilities, explanations, markdown,
or additional keys.""",
)
BINARY_SCHEMA_JSON = canonical_json({
    "title": "faithfulness_verdict",
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["supported", "unsupported"]}},
    "required": ["verdict"], "additionalProperties": False,
})


def binary_request(item, config):
    return replace(build_request(item, config, BINARY_PROMPT),
                   response_schema_json=BINARY_SCHEMA_JSON,
                   contract_version="binary-diagnostic-request-v1")


def parse_verdict(text):
    if not isinstance(text, str):
        raise JudgeParseError("invalid_text_type")
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except JudgeParseError:
        raise
    except (ValueError, RecursionError):
        raise JudgeParseError("invalid_json") from None
    if not isinstance(payload, dict) or set(payload) != {"verdict"}:
        raise JudgeParseError("invalid_schema")
    if type(payload["verdict"]) is not str or payload["verdict"] not in ("supported", "unsupported"):
        raise JudgeParseError("invalid_verdict")
    return payload["verdict"]
