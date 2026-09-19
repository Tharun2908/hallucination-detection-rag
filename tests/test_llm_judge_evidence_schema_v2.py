"""Evidence schema regression checks; no model calls or native compiler required."""

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import unittest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from post_thesis.llm_judge.evidence_cases import case_records, cases_sha256
from post_thesis.llm_judge.evidence_contract import EVIDENCE_PROMPT, EVIDENCE_SCHEMA_JSON, evidence_request, parse_evidence
from post_thesis.llm_judge.evidence_schema_v2 import EVIDENCE_SCHEMA_V2_JSON, EVIDENCE_V2_CONTRACT_VERSION, evidence_request_v2
from post_thesis.llm_judge.judge import JudgeConfig, JudgeInput
from post_thesis.llm_judge.parse import JudgeParseError
from post_thesis.llm_judge.prompts import canonical_json, content_hash
from post_thesis.llm_judge.schema_v2_checks import OBSERVATIONS_PATH, check_cases, compare_acceptance


class ProvenanceTests(unittest.TestCase):
    def test_original_statuses_reproduce_without_salvaging_raw_verdicts(self):
        report = json.loads(OBSERVATIONS_PATH.read_text())
        self.assertEqual(report['records_sha256'], content_hash(report['records']))
        self.assertEqual(report['cases_sha256'], cases_sha256())
        valid, failures = 0, []
        for row, case in zip(report['records'], case_records(), strict=True):
            self.assertEqual(row['sample_id'], case['sample_id'])
            try:
                result = parse_evidence(row['raw_response'], JudgeInput(**case['input']))
            except JudgeParseError as error:
                self.assertEqual(row['status'], 'invalid_output')
                self.assertEqual(row['error_code'], error.code)
                self.assertIsNone(row['valid_verdict'])
                failures.append(row['sample_id'])
            else:
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(result.verdict, row['valid_verdict'])
                valid += 1
        self.assertEqual(valid, 11)
        self.assertEqual(failures, ['attribute-supported-short', 'numeric-supported-short', 'absence-explicit-false-short'])
        self.assertEqual(report['execution']['total_tokens'], 14504)
        concerns = [r for r in report['records'] if r['manual_review'] == 'misleading_unknown_or_not_offered_explanation']
        self.assertEqual([r['sample_id'] for r in concerns], ['attribute-unknown-embedded'])

    def test_schema_only_identity_change_preserves_prompt_parser_inputs(self):
        config = JudgeConfig('offline', 'model', 'configuration', max_output_tokens=512)
        for case in case_records():
            item = JudgeInput(**case['input'])
            old, new = evidence_request(item, config), evidence_request_v2(item, config)
            self.assertEqual(old.messages, new.messages)
            self.assertEqual(old.config, new.config)
            self.assertEqual(old.prompt_sha256, new.prompt_sha256)
            self.assertEqual(new.contract_version, EVIDENCE_V2_CONTRACT_VERSION)
            self.assertNotEqual(old.key, new.key)
            changed = {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]}
            self.assertEqual(changed, {'response_schema_json', 'contract_version'})
        descriptor = json.loads((Path(__file__).resolve().parents[1] / 'post_thesis/llm_judge/evidence_schema_v2.json').read_text())
        self.assertEqual(descriptor['schema_sha256'], content_hash(json.loads(EVIDENCE_SCHEMA_V2_JSON)))
        self.assertEqual(descriptor['prompt_sha256'], EVIDENCE_PROMPT.sha256)
        self.assertFalse(descriptor['generation_enabled'])
        self.assertEqual(content_hash(json.loads(EVIDENCE_SCHEMA_JSON)), 'ecf37ce6cc1775c9af61d7b30754f5fd84c9c7ccf6a8845907a6347cc8807f12')

    def test_acceptance_comparison_reports_overpermissive_matcher(self):
        # This fake matcher tests the reporting logic, not XGrammar compatibility.
        rows = compare_acceptance(lambda payload: True)
        self.assertEqual(len(rows), 38)
        self.assertTrue(any(not row['matches_expected'] for row in rows))
        self.assertEqual(len({c['name'] for c in check_cases()}), len(rows))


@unittest.skipIf(Draft202012Validator is None, 'Install requirements-schema.txt for independent JSON Schema validation')
class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(EVIDENCE_SCHEMA_V2_JSON)
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)

    def test_all_frozen_acceptance_fixtures_match_standard_validator(self):
        rows = compare_acceptance(self.validator.is_valid)
        self.assertTrue(all(row['matches_expected'] for row in rows), rows)

    def test_actual_three_failures_allowed_by_old_schema_rejected_by_new(self):
        old = Draft202012Validator(json.loads(EVIDENCE_SCHEMA_JSON))
        for row in json.loads(OBSERVATIONS_PATH.read_text())['records']:
            value = json.loads(row['raw_response'])
            self.assertTrue(old.is_valid(value))
            self.assertEqual(self.validator.is_valid(value), row['status'] == 'ok')

    def test_field_dependency_matrix_matches_unchanged_parser(self):
        item = JudgeInput('The venue offers outdoor seating.', "{'OutdoorSeating': False}")
        for verdict, issue, answer, context, explanation in itertools.product(
            ('supported', 'unsupported'), ('none', 'contradiction', 'insufficient_support'),
            (None, item.answer), (None, item.context), (None, 'Evidence comparison.')
        ):
            value = dict(verdict=verdict, issue_type=issue, answer_quote=answer,
                         context_quote=context, explanation=explanation)
            try:
                parse_evidence(canonical_json(value), item)
                parser_accepts = True
            except JudgeParseError:
                parser_accepts = False
            with self.subTest(value=value):
                self.assertEqual(self.validator.is_valid(value), parser_accepts)

    def test_schema_does_not_claim_source_or_semantic_verification(self):
        case = next(c for c in check_cases() if c['name'] == 'source_membership_not_enforced')
        self.assertTrue(self.validator.is_valid(case['payload']))
        with self.assertRaises(JudgeParseError):
            parse_evidence(canonical_json(case['payload']), JudgeInput('Actual answer.', 'Actual context.'))
        row = next(r for r in json.loads(OBSERVATIONS_PATH.read_text())['records'] if r['sample_id'] == 'attribute-unknown-embedded')
        self.assertTrue(self.validator.is_valid(json.loads(row['raw_response'])))
        self.assertEqual(row['manual_review'], 'misleading_unknown_or_not_offered_explanation')


if __name__ == '__main__':
    unittest.main()
