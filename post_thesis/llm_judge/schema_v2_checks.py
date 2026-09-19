"""Offline schema acceptance fixtures; these are not model requests or results."""

from copy import deepcopy
import json
from pathlib import Path

from .prompts import content_hash

OBSERVATIONS_PATH = Path(__file__).resolve().parents[2] / 'results/post_thesis/llm_judge/evidence_diagnostic_v1_20260919.json'


def check_cases():
    supported = dict(answer_quote=None, context_quote=None, explanation=None,
                     issue_type='none', verdict='supported')
    contradiction = dict(answer_quote='The venue offers outdoor seating.',
                         context_quote="{'OutdoorSeating': False}",
                         explanation='The field rules out seating.',
                         issue_type='contradiction', verdict='unsupported')
    insufficient = {**contradiction, 'context_quote': None, 'issue_type': 'insufficient_support'}
    cases = []

    def add(name, payload, expected):
        cases.append({'name': name, 'payload': deepcopy(payload), 'expected_schema_acceptance': expected})

    add('supported_all_null', supported, True)
    add('contradiction_with_quotes', contradiction, True)
    add('insufficient_without_context_quote', insufficient, True)
    add('insufficient_with_context_quote', {**insufficient, 'context_quote': "{'OutdoorSeating': None}"}, True)
    for key, value in [('answer_quote', 'Claim.'), ('context_quote', 'Source.'), ('explanation', 'Supported.')]:
        add('supported_nonnull_' + key, {**supported, key: value}, False)
    add('supported_wrong_issue', {**supported, 'issue_type': 'contradiction'}, False)
    for key in ('answer_quote', 'context_quote', 'explanation'):
        add('contradiction_null_' + key, {**contradiction, key: None}, False)
    add('unsupported_none_issue', {**contradiction, 'issue_type': 'none'}, False)
    add('unknown_issue', {**contradiction, 'issue_type': 'unknown'}, False)
    add('boolean_verdict', {**supported, 'verdict': True}, False)
    add('extra_key', {**supported, 'extra': 'value'}, False)
    missing = dict(supported); missing.pop('explanation')
    add('missing_key', missing, False)
    for key, limit in [('answer_quote', 400), ('context_quote', 600), ('explanation', 320)]:
        add('empty_' + key, {**contradiction, key: ''}, False)
        add('overlong_' + key, {**contradiction, key: 'x' * (limit + 1)}, False)
    # These must remain schema-valid: source membership, whitespace policy and
    # semantic relevance are extra parser/review responsibilities, not this schema.
    add('source_membership_not_enforced', {**contradiction, 'answer_quote': 'Invented assertion.'}, True)
    add('whitespace_parser_still_needed', {**contradiction, 'answer_quote': ' '}, True)
    observations = json.loads(OBSERVATIONS_PATH.read_text(encoding='utf-8'))
    if content_hash(observations['records']) != observations['records_sha256']:
        raise ValueError('changed observation record checksum')
    for row in observations['records']:
        add('observed_' + row['sample_id'], json.loads(row['raw_response']), row['status'] == 'ok')
    return tuple(cases)


def checks_sha256():
    return content_hash(check_cases())


def compare_acceptance(accept):
    results = []
    for case in check_cases():
        observed = bool(accept(case['payload']))
        results.append({'name': case['name'], 'expected': case['expected_schema_acceptance'],
                        'accepted': observed, 'matches_expected': observed == case['expected_schema_acceptance']})
    return results
