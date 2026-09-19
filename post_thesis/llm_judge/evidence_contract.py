"""Post-thesis evidence diagnostic: offline request contract and strict parsing.

Exact quote membership is a syntactic check, not a semantic support judgment.
No backend, runner, retries, score conversion or generation command is provided.
"""

from dataclasses import dataclass, replace
import json

from .binary_contract import BINARY_PROMPT
from .judge import JudgeInput, build_request
from .parse import JudgeParseError, _reject_constant, _unique_object
from .prompts import PromptSpec, canonical_json

EVIDENCE_CONTRACT_VERSION = "evidence-diagnostic-request-v1"
ANSWER_QUOTE_MAX_CHARS = 400
CONTEXT_QUOTE_MAX_CHARS = 600
EXPLANATION_MAX_CHARS = 320
RESPONSE_MAX_CHARS = 16384

# Retain the binary support criteria; replace only its output instructions.
EVIDENCE_PROMPT = PromptSpec(
    version="faithfulness-evidence-diagnostic-v1",
    system_text=BINARY_PROMPT.system_text.split("\nReturn only one JSON object", 1)[0]
    + """

Return one JSON object with exactly these five fields:
answer_quote, context_quote, issue_type, explanation, verdict.
Do not output probabilities, markdown, a reasoning trace, or extra fields.

If you find an unsupported factual assertion, select one clear example. Copy a
contiguous exact excerpt from answer into answer_quote (at most 400 characters).
Include enough words to preserve the assertion's entity, negation and scope.
Do not paraphrase, correct, trim within, or join fragments of the quoted text.

Use issue_type contradiction only when supplied context explicitly conflicts
with that assertion. Copy a relevant contiguous exact context excerpt into
context_quote (at most 600 characters). A missing or unknown field alone is not
a contradiction. Inspect other context passages for support or conflicting
statements; a single selected excerpt does not override the rest of the context.

Use issue_type insufficient_support when the full assertion lacks support in
the context without an explicit contradiction. If a relevant field or passage
helps show what is and is not established, copy it exactly into context_quote.
Otherwise set context_quote to null. A null quote is not proof that support is
absent: inspect the whole supplied context before deciding.

For either issue type, write a brief evidence comparison in explanation (at most
320 characters) and set verdict to unsupported. Explain the unsupported addition
or conflict without outside knowledge; do not give step-by-step reasoning.

If all factual assertions are supported, or there are no factual assertions,
set verdict to supported, issue_type to none, and answer_quote, context_quote,
and explanation to null. One supported claim is not enough to justify a supported
answer. Do not output a token supported-claim example instead of checking the
whole answer. Use JSON null, not the string null or an empty string.""",
)


def _nullable_text(limit):
    return {"type": ["string", "null"], "minLength": 1, "maxLength": limit}


EVIDENCE_SCHEMA_JSON = canonical_json({
    "title": "faithfulness_evidence_verdict",
    "type": "object",
    "properties": {
        "answer_quote": _nullable_text(ANSWER_QUOTE_MAX_CHARS),
        "context_quote": _nullable_text(CONTEXT_QUOTE_MAX_CHARS),
        "issue_type": {"type": "string", "enum": ["none", "contradiction", "insufficient_support"]},
        "explanation": _nullable_text(EXPLANATION_MAX_CHARS),
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
    },
    "required": ["answer_quote", "context_quote", "issue_type", "explanation", "verdict"],
    "additionalProperties": False,
})


@dataclass(frozen=True)
class EvidenceVerdict:
    answer_quote: str | None
    context_quote: str | None
    issue_type: str
    explanation: str | None
    verdict: str


def evidence_request(item, config):
    """Build an isolated request; constructing it performs no model calls."""
    return replace(build_request(item, config, EVIDENCE_PROMPT),
                   response_schema_json=EVIDENCE_SCHEMA_JSON,
                   contract_version=EVIDENCE_CONTRACT_VERSION)


def parse_evidence(text: str, item: JudgeInput) -> EvidenceVerdict:
    """Reject malformed/cross-field-invalid output and quotes absent from input.

    The caller must handle refused/incomplete backend outcomes before parsing.
    Successful parsing does not verify the explanation or verdict semantically.
    """
    if not isinstance(item, JudgeInput):
        raise TypeError("item must be JudgeInput")
    if not isinstance(text, str):
        raise JudgeParseError("invalid_text_type")
    if len(text) > RESPONSE_MAX_CHARS:
        raise JudgeParseError("response_too_long")
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except JudgeParseError:
        raise
    except (ValueError, RecursionError):
        raise JudgeParseError("invalid_json") from None
    keys = {"answer_quote", "context_quote", "issue_type", "explanation", "verdict"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise JudgeParseError("invalid_schema")
    if type(payload["verdict"]) is not str or payload["verdict"] not in ("supported", "unsupported"):
        raise JudgeParseError("invalid_verdict")
    if type(payload["issue_type"]) is not str or payload["issue_type"] not in (
            "none", "contradiction", "insufficient_support"):
        raise JudgeParseError("invalid_issue_type")
    for key, limit in (("answer_quote", ANSWER_QUOTE_MAX_CHARS),
                       ("context_quote", CONTEXT_QUOTE_MAX_CHARS),
                       ("explanation", EXPLANATION_MAX_CHARS)):
        value = payload[key]
        if value is not None and (type(value) is not str or not value.strip() or len(value) > limit):
            raise JudgeParseError("invalid_" + key)
    if payload["verdict"] == "supported":
        if payload["issue_type"] != "none" or any(payload[k] is not None for k in (
                "answer_quote", "context_quote", "explanation")):
            raise JudgeParseError("inconsistent_supported_output")
    else:
        if (payload["issue_type"] == "none" or payload["answer_quote"] is None
                or payload["explanation"] is None):
            raise JudgeParseError("incomplete_unsupported_output")
        if payload["issue_type"] == "contradiction" and payload["context_quote"] is None:
            raise JudgeParseError("missing_contradiction_evidence")
    for key, source in (("answer_quote", item.answer), ("context_quote", item.context)):
        quote = payload[key]
        if quote is not None and quote not in source:
            raise JudgeParseError(key + "_not_in_input")
    return EvidenceVerdict(**payload)
