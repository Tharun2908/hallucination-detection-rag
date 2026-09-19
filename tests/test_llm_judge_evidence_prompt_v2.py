"""Offline candidate contracts; these tests do not establish judge accuracy."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from post_thesis.llm_judge.check_evidence_prompt_v2 import check
from post_thesis.llm_judge.evidence_cases import case_records as historical_cases
from post_thesis.llm_judge.evidence_contract import EVIDENCE_PROMPT, parse_evidence
from post_thesis.llm_judge.evidence_prompt_v2 import EVIDENCE_PROMPT_V2, evidence_request_prompt_v2
from post_thesis.llm_judge.evidence_prompt_v2_cases import case_records, examples, new_cases
from post_thesis.llm_judge.evidence_schema_v3 import EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_FIELD_ORDER, evidence_request_v3
from post_thesis.llm_judge.judge import JudgeConfig, JudgeInput
from post_thesis.llm_judge.parse import JudgeParseError


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.config = JudgeConfig('offline', 'fixture-model', 'offline-only', max_output_tokens=512)

    def test_historical_fourteen_are_retained_exactly(self):
        self.assertEqual(case_records()[:14], historical_cases())
        self.assertEqual(len(case_records()), 26)
        self.assertEqual(len({r['sample_id'] for r in case_records()}), 26)
        self.assertEqual(EVIDENCE_PROMPT.sha256, '21ba57c9ec668c495a16e4635660c4504f26b5056638d4b17e6424e32ce52f4c')

    def test_only_prompt_fields_change_and_old_cache_keys_cannot_match(self):
        for ex in examples():
            with self.subTest(sample_id=ex.sample_id):
                old = evidence_request_v3(ex.item, self.config)
                new = evidence_request_prompt_v2(ex.item, self.config)
                changed = {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]}
                self.assertEqual(changed, {'messages', 'prompt_version', 'prompt_sha256'})
                self.assertEqual(old.messages[1], new.messages[1])
                self.assertEqual(new.response_schema_json, EVIDENCE_SCHEMA_V3_JSON)
                self.assertNotEqual(old.key, new.key)
                self.assertEqual(new.key, evidence_request_prompt_v2(ex.item, self.config).key)

    def test_expectations_references_and_case_metadata_stay_out_of_requests(self):
        for row, ex in zip(case_records(), examples(), strict=True):
            req = evidence_request_prompt_v2(ex.item, self.config)
            data = json.loads(req.messages[1].content)
            self.assertEqual(data, row['input'])
            self.assertEqual(set(data), {'answer', 'context'})
            self.assertEqual(req.messages[0].content, EVIDENCE_PROMPT_V2.system_text)

    def test_authored_references_obey_existing_parser_and_wire_order(self):
        for row in new_cases():
            with self.subTest(sample_id=row['sample_id']):
                ref = row['authored_reference']
                self.assertEqual(tuple(ref), EVIDENCE_V3_FIELD_ORDER)
                parsed = parse_evidence(json.dumps(ref), JudgeInput(**row['input']))
                self.assertEqual(asdict(parsed), ref)
                self.assertEqual(parsed.verdict, row['expected_verdict'])
                self.assertEqual(parsed.issue_type, row['expected_issue_type'])

    def test_reconstructed_dictionary_is_still_rejected(self):
        item = JudgeInput('The venue offers wireless internet.', "{'name': 'Juniper Hall', 'WiFi': 'no', 'Music': None}")
        ref = dict(verdict='unsupported', issue_type='contradiction', answer_quote=item.answer,
                   context_quote="{'WiFi': 'no'}", explanation='The context says WiFi is unavailable.')
        with self.assertRaises(JudgeParseError) as caught:
            parse_evidence(json.dumps(ref), item)
        self.assertEqual(caught.exception.code, 'context_quote_not_in_input')
        ref['context_quote'] = "'WiFi': 'no'"
        self.assertEqual(parse_evidence(json.dumps(ref), item).verdict, 'unsupported')

    def test_rewritten_answer_is_still_rejected(self):
        row = next(r for r in new_cases() if r['sample_id']=='paraphrase-contradiction')
        ref = {**row['authored_reference'], 'answer_quote': 'Use oil to brush the fabric.'}
        with self.assertRaises(JudgeParseError) as caught:
            parse_evidence(json.dumps(ref), JudgeInput(**row['input']))
        self.assertEqual(caught.exception.code, 'answer_quote_not_in_input')

    def test_null_context_allowed_only_for_insufficient_support(self):
        row = next(r for r in new_cases() if r['sample_id']=='cross-passage-support-absent')
        ref = row['authored_reference']
        item = JudgeInput(**row['input'])
        self.assertEqual(parse_evidence(json.dumps(ref), item).issue_type, 'insufficient_support')
        with self.assertRaises(JudgeParseError):
            parse_evidence(json.dumps({**ref, 'issue_type':'contradiction'}), item)

    def test_parser_success_is_not_semantic_correctness(self):
        row = next(r for r in new_cases() if r['sample_id']=='cross-passage-support-first')
        item = JudgeInput(**row['input'])
        wrong = dict(verdict='unsupported', issue_type='insufficient_support', answer_quote=item.answer,
                     context_quote=None, explanation='No passage mentions rosemary.')
        self.assertEqual(parse_evidence(json.dumps(wrong), item).verdict, 'unsupported')
        self.assertEqual(row['expected_verdict'], 'supported')
        self.assertIn(item.answer, item.context)

    def test_checker_is_offline_and_explicit_about_limits(self):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            result = check()
        self.assertEqual(result['synthetic_cases'], 26)
        self.assertEqual(result['authored_references_checked'], 12)
        self.assertEqual(result['generation_calls'], 0)
        self.assertFalse(result['model_behavior_tested'])
        self.assertFalse(result['token_lengths_audited'])
        self.assertIsNone(result['live_execution_plan'])

    def test_prior_scoring_plan_still_pins_prior_prompt(self):
        path = Path(__file__).resolve().parents[1] / 'post_thesis/llm_judge/configs/ragtruth_pilot_50_evidence_v3.json'
        plan = json.loads(path.read_text())
        self.assertEqual(plan['prompt_sha256'], EVIDENCE_PROMPT.sha256)
        self.assertNotEqual(plan['prompt_sha256'], EVIDENCE_PROMPT_V2.sha256)


if __name__ == '__main__':
    unittest.main()
