"""Complete-string order checks, separate from JSON Schema object validity."""

import json
from pathlib import Path

from .evidence_schema_v3 import EVIDENCE_V3_FIELD_ORDER
from .prompts import canonical_json, content_hash
from .schema_v2_checks import check_cases as v2_check_cases

OBSERVATIONS_PATH = Path(__file__).resolve().parents[2] / 'results/post_thesis/llm_judge/evidence_diagnostic_v2_20260919.json'


def verdict_first_json(payload):
    ordered = {key: payload[key] for key in EVIDENCE_V3_FIELD_ORDER if key in payload}
    ordered.update({key: value for key, value in payload.items() if key not in ordered})
    return json.dumps(ordered, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def check_cases():
    original = v2_check_cases()
    cases = [{'name': 'verdict_first_' + row['name'],
              'text': verdict_first_json(row['payload']),
              'expected_acceptance': row['expected_schema_acceptance']} for row in original]
    # Each branch (including both insufficient-support variants) must accept the
    # target order and reject alphabetical order for a fixed-order experiment.
    # JSON Schema itself accepts either key order for these four valid objects.
    for row in original[:4]:
        cases.append({'name': 'legacy_order_' + row['name'],
                      'text': canonical_json(row['payload']), 'expected_acceptance': False})
    report = json.loads(OBSERVATIONS_PATH.read_text(encoding='utf-8'))
    if content_hash(report['records']) != report['records_sha256']:
        raise ValueError('changed observation record checksum')
    for row in report['records']:
        if row['raw_response'] is not None:
            cases.append({'name': 'raw_legacy_order_' + row['sample_id'],
                          'text': row['raw_response'], 'expected_acceptance': False})
    return tuple(cases)


def checks_sha256():
    # Text values are retained byte-for-byte: sorting outer record keys does not
    # discard the response member order stored inside these strings.
    return content_hash(check_cases())


def compare_acceptance(accept_text):
    results = []
    for row in check_cases():
        accepted = bool(accept_text(row['text']))
        results.append({'name': row['name'], 'expected': row['expected_acceptance'],
                        'accepted': accepted, 'matches_expected': accepted == row['expected_acceptance']})
    return results
