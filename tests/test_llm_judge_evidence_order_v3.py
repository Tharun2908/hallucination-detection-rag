"""Order identity, observed results and HTTP serialization; no native/GPU claims."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest

try:
    import httpx
except ImportError:
    httpx = None
try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from post_thesis.llm_judge.check_evidence_schema import schema_spec
from post_thesis.llm_judge.evidence_cases import case_records
from post_thesis.llm_judge.evidence_contract import parse_evidence, EVIDENCE_PROMPT
from post_thesis.llm_judge.evidence_schema_v2 import EVIDENCE_SCHEMA_V2_JSON, evidence_request_v2
from post_thesis.llm_judge.evidence_schema_v3 import (
    EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_FIELD_ORDER, EVIDENCE_V3_WIRE_SHA256, evidence_request_v3)
from post_thesis.llm_judge.judge import JudgeInput, JudgeConfig
from post_thesis.llm_judge.prompts import canonical_json, content_hash
from post_thesis.llm_judge.schema_v3_checks import OBSERVATIONS_PATH, check_cases, checks_sha256, compare_acceptance
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile


class OrderTests(unittest.TestCase):
    def test_report_matches_parser_and_preserves_two_wrong_verdicts(self):
        report = json.loads(OBSERVATIONS_PATH.read_text())
        self.assertEqual(report['records_sha256'], content_hash(report['records']))
        verdicts = []
        issues = []
        raw_ids = []
        for row, case in zip(report['records'], case_records(), strict=True):
            self.assertEqual(row['sample_id'], case['sample_id'])
            parsed = parse_evidence(canonical_json(row['evidence']), JudgeInput(**case['input']))
            self.assertEqual(asdict(parsed), row['evidence'])
            verdicts.append(parsed.verdict == case['expected_verdict'])
            issues.append(parsed.issue_type == case['expected_issue_type'])
            if row['raw_response'] is not None:
                raw_ids.append(row['sample_id'])
                self.assertEqual(json.loads(row['raw_response']), row['evidence'])
                self.assertEqual(list(json.loads(row['raw_response'])),
                                 ['answer_quote', 'context_quote', 'explanation', 'issue_type', 'verdict'])
                self.assertEqual(parsed.verdict, 'unsupported')
                self.assertEqual(case['expected_verdict'], 'supported')
        self.assertEqual((sum(verdicts), sum(issues)), (12, 12))
        self.assertEqual(raw_ids, ['attribute-supported-short', 'absence-explicit-false-short'])
        self.assertEqual(report['execution']['total_tokens'], 14489)
        self.assertEqual(sum(r['manual_review'] != 'no_issue_identified_in_fixture_review' for r in report['records']), 3)

    def test_same_schema_values_but_order_sensitive_identity(self):
        old, new = json.loads(EVIDENCE_SCHEMA_V2_JSON), json.loads(EVIDENCE_SCHEMA_V3_JSON)
        self.assertEqual(old, new)  # Same validation rules, title and required arrays.
        self.assertEqual(content_hash(old), content_hash(new))
        self.assertNotEqual(EVIDENCE_SCHEMA_V2_JSON, EVIDENCE_SCHEMA_V3_JSON)
        self.assertNotEqual(hashlib.sha256(EVIDENCE_SCHEMA_V2_JSON.encode()).hexdigest(), EVIDENCE_V3_WIRE_SHA256)
        for branch in new['anyOf']:
            self.assertEqual(tuple(branch['properties']), EVIDENCE_V3_FIELD_ORDER)
        descriptor = json.loads((Path(__file__).resolve().parents[1] / 'post_thesis/llm_judge/evidence_schema_v3.json').read_text())
        self.assertEqual(descriptor['serialized_schema_sha256'], EVIDENCE_V3_WIRE_SHA256)
        self.assertEqual(descriptor['schema_canonical_sha256'], content_hash(new))
        self.assertEqual(descriptor['fixture_sha256'], checks_sha256())
        self.assertEqual(descriptor['prompt_sha256'], EVIDENCE_PROMPT.sha256)
        self.assertFalse(descriptor['generation_enabled'])
        self.assertIsNone(descriptor['execution_plan'])

    def test_request_key_changes_for_order_alone_and_prompt_is_unchanged(self):
        config = JudgeConfig('offline', 'model', 'version', max_output_tokens=512)
        for case in case_records():
            item = JudgeInput(**case['input'])
            old, new = evidence_request_v2(item, config), evidence_request_v3(item, config)
            self.assertEqual(old.messages, new.messages)
            self.assertEqual(old.config, new.config)
            changed = {key for key in asdict(old) if asdict(old)[key] != asdict(new)[key]}
            self.assertEqual(changed, {'response_schema_json', 'contract_version'})
            self.assertNotEqual(old.key, new.key)
            self.assertNotEqual(old.key, replace(new, contract_version=old.contract_version).key)

    def test_native_fixture_identity_and_order_blind_matcher_is_detected(self):
        rows = check_cases()
        self.assertEqual(len(rows), 44)
        self.assertEqual(len({r['name'] for r in rows}), 44)
        schema, digest = schema_spec('v3')
        self.assertEqual(schema, EVIDENCE_SCHEMA_V3_JSON)
        self.assertEqual(digest, checks_sha256())
        # Fake always-accept matcher tests reporting only, not native compatibility.
        results = compare_acceptance(lambda _: True)
        self.assertTrue(any(not row['matches_expected'] for row in results))
        self.assertEqual(sum(r['name'].startswith(('legacy_order_', 'raw_legacy_order_')) for r in rows), 6)
        for row in rows[:38]:
            keys = list(json.loads(row['text']))
            self.assertEqual(keys[0], 'verdict')

    @unittest.skipIf(Draft202012Validator is None, 'Requires independent JSON Schema validator')
    def test_schema_validity_does_not_test_order_or_semantic_correctness(self):
        validator = Draft202012Validator(json.loads(EVIDENCE_SCHEMA_V3_JSON))
        for row in check_cases():
            valid = validator.is_valid(json.loads(row['text']))
            if row['name'].startswith(('legacy_order_', 'raw_legacy_order_')):
                self.assertTrue(valid)  # Native order test intentionally expects rejection.
                self.assertFalse(row['expected_acceptance'])
            else:
                self.assertEqual(valid, row['expected_acceptance'])
        report = json.loads(OBSERVATIONS_PATH.read_text())
        self.assertTrue(all(validator.is_valid(r['evidence']) for r in report['records']))


@unittest.skipIf(httpx is None, 'Requires isolated HTTP client')
class WireTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_http_body_preserves_verdict_first_schema_properties(self):
        p = load_profile('evidence-v1')
        seen = []
        async def handler(request):
            seen.append(request)
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'count': 900, 'max_model_len': p['max_model_len']})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            return httpx.Response(200, json={'model': p['served_model_name'],
                'usage': {'prompt_tokens': 900, 'completion_tokens': 20},
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}]})
        async with VLLMBackend(profile_name='evidence-v1', transport=httpx.MockTransport(handler)) as backend:
            item = JudgeInput(**case_records()[0]['input'])
            request = evidence_request_v3(item, backend.config)
            await backend.complete_counted(request, expected_input_tokens=900)
            wire = json.loads(seen[-1].content)
            schema = wire['response_format']['json_schema']['schema']
            for branch in schema['anyOf']:
                self.assertEqual(tuple(branch['properties']), EVIDENCE_V3_FIELD_ORDER)
            self.assertEqual(schema, json.loads(EVIDENCE_SCHEMA_V2_JSON))
            self.assertEqual(wire['messages'], [asdict(m) for m in evidence_request_v2(item, backend.config).messages])
            self.assertEqual(len(seen), 2)


if __name__ == '__main__':
    unittest.main()
