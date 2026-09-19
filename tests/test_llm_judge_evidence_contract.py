"""Offline post-thesis evidence contract: no network, model or benchmark data."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from post_thesis.llm_judge.binary_contract import BINARY_PROMPT, binary_request, parse_verdict
from post_thesis.llm_judge.diagnostic_cases import case_records as old_cases
from post_thesis.llm_judge.evidence_cases import case_records, cases_sha256, examples
from post_thesis.llm_judge.evidence_contract import (
    ANSWER_QUOTE_MAX_CHARS, CONTEXT_QUOTE_MAX_CHARS, EXPLANATION_MAX_CHARS,
    RESPONSE_MAX_CHARS, EVIDENCE_PROMPT, EVIDENCE_SCHEMA_JSON,
    EVIDENCE_CONTRACT_VERSION, evidence_request, parse_evidence,
)
from post_thesis.llm_judge.judge import JudgeConfig, JudgeInput, build_request
from post_thesis.llm_judge.parse import JudgeParseError, parse_score
from post_thesis.llm_judge.prompts import DEVELOPMENT_PROMPT_V2, content_hash


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.item = JudgeInput('The venue offers outdoor seating.', "{'OutdoorSeating': False}")
        self.valid = dict(answer_quote=self.item.answer, context_quote=self.item.context,
                          issue_type='contradiction', explanation='The field explicitly rules out seating.',
                          verdict='unsupported')
        self.supported = dict(answer_quote=None, context_quote=None, issue_type='none',
                              explanation=None, verdict='supported')

    def parse(self, value, item=None):
        return parse_evidence(json.dumps(value), item or self.item)

    def test_supported_is_a_verdict_not_verified_semantic_support(self):
        result = self.parse(self.supported)
        self.assertEqual(result.verdict, 'supported')
        self.assertIsNone(result.answer_quote)
        self.assertNotIn('unsupported_probability', asdict(result))
        # This source contradicts the answer: parsing must NOT claim correctness.
        self.assertEqual(self.item.context, "{'OutdoorSeating': False}")

    def test_contradiction_requires_both_exact_quotes(self):
        self.assertEqual(asdict(self.parse(self.valid)), self.valid)
        for key in ('answer_quote', 'context_quote'):
            with self.subTest(key=key), self.assertRaises(JudgeParseError):
                self.parse({**self.valid, key: None})

    def test_insufficient_support_allows_relevant_quote_or_null(self):
        item = JudgeInput(self.item.answer, "{'OutdoorSeating': None}")
        for quote in (item.context, None):
            result = self.parse({**self.valid, 'context_quote': quote,
                                 'issue_type': 'insufficient_support',
                                 'explanation': 'An unknown field does not establish availability.'}, item)
            self.assertEqual(result.issue_type, 'insufficient_support')
            self.assertEqual(result.context_quote, quote)
        empty = replace(item, context='')
        self.assertEqual(self.parse({**self.valid, 'context_quote': None,
                                    'issue_type': 'insufficient_support'}, empty).verdict, 'unsupported')

    def test_cross_field_contradictions_are_rejected(self):
        invalid = [{**self.supported, 'answer_quote': self.item.answer},
                   {**self.supported, 'context_quote': self.item.context},
                   {**self.supported, 'explanation': 'Everything is fine.'},
                   {**self.supported, 'issue_type': 'contradiction'},
                   {**self.valid, 'issue_type': 'none'},
                   {**self.valid, 'explanation': None}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(JudgeParseError):
                self.parse(value)

    def test_fabricated_paraphrased_and_wrong_source_quotes_rejected(self):
        for changes in ({'answer_quote': 'The venue has outdoor seating.'},
                        {'answer_quote': self.item.context},
                        {'context_quote': self.item.answer},
                        {'context_quote': "{'OutdoorSeating': True}"}):
            with self.subTest(changes=changes), self.assertRaises(JudgeParseError):
                self.parse({**self.valid, **changes})

    def test_exactness_does_not_normalize_whitespace_unicode_or_fragments(self):
        item = JudgeInput('Café has\ntwo rooms.', 'Café has one room; its foyer is blue.')
        good = {**self.valid, 'answer_quote': item.answer, 'context_quote': 'Café has one room'}
        self.parse(good, item)
        for changes in ({'answer_quote': 'Café has two rooms.'},
                        {'answer_quote': 'Cafe\u0301 has\ntwo rooms.'},
                        {'context_quote': 'Café has one room ... blue.'}):
            with self.subTest(changes=changes), self.assertRaises(JudgeParseError):
                self.parse({**good, **changes}, item)

    def test_quote_membership_is_checked_after_json_decoding(self):
        item = JudgeInput('The sign says "open".\nPath C:\\room', 'The sign says "closed".\nPath C:\\room')
        value = {**self.valid, 'answer_quote': item.answer, 'context_quote': item.context}
        self.assertEqual(self.parse(value, item).answer_quote, item.answer)

    def test_strict_json_schema_and_types(self):
        invalid = [None, '', 'null', '[]', '{}', 'true',
                   '{"verdict":"supported","verdict":"unsupported"}',
                   json.dumps({**self.valid, 'verdict': True}),
                   json.dumps({**self.valid, 'issue_type': ['contradiction']}),
                   json.dumps({**self.valid, 'issue_type': 'unknown'}),
                   json.dumps({**self.valid, 'extra': 'ignored?'}),
                   json.dumps({**self.valid, 'answer_quote': 123}),
                   json.dumps({**self.valid, 'context_quote': []}),
                   json.dumps({**self.valid, 'explanation': False}),
                   '{"verdict":NaN}',
                   json.dumps(self.valid) + ' trailing',
                   '```json\n' + json.dumps(self.valid) + '\n```',
                   json.dumps({'unsupported_probability': 0.8})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(JudgeParseError):
                parse_evidence(value, self.item)

    def test_empty_whitespace_null_strings_and_limits(self):
        for key, limit in (('answer_quote', ANSWER_QUOTE_MAX_CHARS),
                           ('context_quote', CONTEXT_QUOTE_MAX_CHARS),
                           ('explanation', EXPLANATION_MAX_CHARS)):
            for value in ('', ' \n\t', 'x' * (limit + 1)):
                with self.subTest(key=key, value=value), self.assertRaises(JudgeParseError):
                    self.parse({**self.valid, key: value})
        with self.assertRaises(JudgeParseError):
            self.parse({**self.supported, 'answer_quote': 'null'})
        with self.assertRaisesRegex(JudgeParseError, 'response_too_long'):
            parse_evidence(' ' * (RESPONSE_MAX_CHARS + 1), self.item)
        with self.assertRaises(TypeError):
            parse_evidence(json.dumps(self.valid), {'answer': 'text', 'context': 'text'})

    def test_unfaithful_explanation_can_pass_quote_membership(self):
        # Deliberately wrong assessment with real quotes: manual semantic review
        # remains necessary and cannot be replaced by a substring check.
        result = self.parse({**self.valid, 'explanation': 'The context says there are two swimming pools.'})
        self.assertEqual(result.verdict, 'unsupported')

    def test_request_isolation_preserves_input_and_separates_cache_keys(self):
        config = JudgeConfig('offline', 'model', 'frozen-version', max_output_tokens=512)
        for ex in examples():
            request = evidence_request(ex.item, config)
            binary = binary_request(ex.item, config)
            probability = build_request(ex.item, config, DEVELOPMENT_PROMPT_V2)
            self.assertEqual(request.messages[1], binary.messages[1])
            self.assertEqual(request.messages[1], probability.messages[1])
            self.assertEqual(set(json.loads(request.messages[1].content)), {'answer', 'context'})
            self.assertNotIn('expected_verdict', request.messages[1].content)
            self.assertEqual(request.contract_version, EVIDENCE_CONTRACT_VERSION)
            self.assertEqual(len({request.key, binary.key, probability.key}), 3)
            self.assertNotEqual(request.key, evidence_request(ex.item, replace(config, max_output_tokens=128)).key)
        with self.assertRaises(TypeError):
            evidence_request({'answer': 'a', 'context': 'c', 'label': 1}, config)

    def test_old_parsers_do_not_accept_new_contract(self):
        for parser in (parse_score, parse_verdict):
            with self.assertRaises(JudgeParseError):
                parser(json.dumps(self.valid))
        self.assertEqual(BINARY_PROMPT.sha256, 'e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683')
        self.assertEqual(DEVELOPMENT_PROMPT_V2.sha256, '7771610009b40b5cce476fa4abda9ecb6bd025ecf71d635d7d3e7feb3c70135a')

    def test_frozen_contract_descriptor_and_synthetic_controls(self):
        p = Path(__file__).resolve().parents[1] / 'post_thesis/llm_judge/evidence_contract_v1.json'
        saved = json.loads(p.read_text())
        self.assertEqual(saved['prompt_sha256'], EVIDENCE_PROMPT.sha256)
        self.assertEqual(saved['schema_sha256'], content_hash(json.loads(EVIDENCE_SCHEMA_JSON)))
        self.assertEqual(saved['cases_sha256'], cases_sha256())
        self.assertEqual(saved['contract_version'], EVIDENCE_CONTRACT_VERSION)
        self.assertFalse(saved['generation_enabled'])
        cases = case_records()
        self.assertEqual(len(cases), 14)
        self.assertEqual(len({r['sample_id'] for r in cases}), 14)
        self.assertEqual(sum(r['expected_verdict'] == 'supported' for r in cases), 6)
        for new, old in zip(cases[:10], old_cases()):
            self.assertEqual(new['input'], old['input'])
            self.assertEqual(new['sample_id'], old['sample_id'])
        for short, embedded in zip(cases[::2], cases[1::2]):
            self.assertEqual(short['input']['context'], embedded['input']['context'])
            self.assertIn(short['input']['answer'], embedded['input']['answer'])
            self.assertEqual(short['expected_verdict'], embedded['expected_verdict'])

    def test_canned_synthetic_responses_are_parser_fixtures_not_model_results(self):
        for row in case_records():
            item = JudgeInput(**row['input'])
            if row['expected_verdict'] == 'supported':
                payload = self.supported
            else:
                if row['family'] == 'numeric':
                    answer = 'The selected group comprises the highest 60% of recorded scores.'
                    context = 'The selected group comprises the lowest 60% of recorded scores.'
                else:
                    answer = ('The venue does not offer outdoor seating.' if row['family'] == 'absence'
                              else 'The venue offers outdoor seating.')
                    context = item.context.split('. ', 1)[0]
                payload = dict(answer_quote=answer, context_quote=context,
                               issue_type=row['expected_issue_type'],
                               explanation='Synthetic parser fixture; not a model response.',
                               verdict='unsupported')
            with self.subTest(sample_id=row['sample_id']):
                parsed = self.parse(payload, item)
                self.assertEqual(parsed.verdict, row['expected_verdict'])
                self.assertEqual(parsed.issue_type, row['expected_issue_type'])


if __name__ == '__main__':
    unittest.main()
