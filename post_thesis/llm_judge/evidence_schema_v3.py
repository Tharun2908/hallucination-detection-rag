"""Offline verdict-first candidate; unchanged v2 validation semantics and prompt."""

from dataclasses import replace
import hashlib
import json

from .evidence_schema_v2 import EVIDENCE_SCHEMA_V2_JSON, evidence_request_v2

EVIDENCE_V3_CONTRACT_VERSION = 'evidence-diagnostic-request-v3'
EVIDENCE_V3_FIELD_ORDER = ('verdict', 'issue_type', 'answer_quote', 'context_quote', 'explanation')


def _ordered_schema():
    schema = json.loads(EVIDENCE_SCHEMA_V2_JSON)
    for branch in schema['anyOf']:
        fields = branch['properties']
        branch['properties'] = {key: fields[key] for key in EVIDENCE_V3_FIELD_ORDER}
    # Keep title, required arrays, branches, constraints and all other values.
    # Canonical JSON would sort properties back to the order we are testing.
    return json.dumps(schema, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


EVIDENCE_SCHEMA_V3_JSON = _ordered_schema()
EVIDENCE_V3_WIRE_SHA256 = hashlib.sha256(EVIDENCE_SCHEMA_V3_JSON.encode('utf-8')).hexdigest()


def evidence_request_v3(item, config):
    return replace(evidence_request_v2(item, config),
                   response_schema_json=EVIDENCE_SCHEMA_V3_JSON,
                   contract_version=EVIDENCE_V3_CONTRACT_VERSION)
