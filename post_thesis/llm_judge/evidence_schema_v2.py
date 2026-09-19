"""Schema-only evidence revision: freeze field dependencies, preserve prompt/parser.

No scoring CLI or new generation budget. Quote membership and semantic quality
cannot be enforced by this generic JSON schema and still need separate checks.
"""

from dataclasses import replace

from .evidence_contract import (
    ANSWER_QUOTE_MAX_CHARS, CONTEXT_QUOTE_MAX_CHARS, EXPLANATION_MAX_CHARS,
    evidence_request,
)
from .prompts import canonical_json

EVIDENCE_V2_CONTRACT_VERSION = "evidence-diagnostic-request-v2"


def _text(limit):
    return {"type": "string", "minLength": 1, "maxLength": limit}


def _branch(verdict, issue, answer, context, explanation):
    return {
        "type": "object",
        "properties": {
            "answer_quote": answer,
            "context_quote": context,
            "explanation": explanation,
            "issue_type": {"type": "string", "enum": [issue]},
            "verdict": {"type": "string", "enum": [verdict]},
        },
        "required": ["answer_quote", "context_quote", "explanation", "issue_type", "verdict"],
        "additionalProperties": False,
    }


EVIDENCE_SCHEMA_V2_JSON = canonical_json({
    "title": "faithfulness_evidence_verdict_v2",
    "type": "object",
    "anyOf": [
        _branch("supported", "none", {"type": "null"}, {"type": "null"}, {"type": "null"}),
        _branch("unsupported", "contradiction", _text(ANSWER_QUOTE_MAX_CHARS),
                _text(CONTEXT_QUOTE_MAX_CHARS), _text(EXPLANATION_MAX_CHARS)),
        _branch("unsupported", "insufficient_support", _text(ANSWER_QUOTE_MAX_CHARS),
                {"anyOf": [_text(CONTEXT_QUOTE_MAX_CHARS), {"type": "null"}]},
                _text(EXPLANATION_MAX_CHARS)),
    ],
})


def evidence_request_v2(item, config):
    """Identical messages/config to v1; distinct schema and request identity."""
    return replace(evidence_request(item, config),
                   response_schema_json=EVIDENCE_SCHEMA_V2_JSON,
                   contract_version=EVIDENCE_V2_CONTRACT_VERSION)
