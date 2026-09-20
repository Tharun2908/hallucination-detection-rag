"""Evidence TRAIN audit: exact input boundary, identity and no generation."""

from copy import deepcopy
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx
except ImportError:
    httpx = None

from post_thesis.llm_judge import audit_pilot as audit_module
from post_thesis.llm_judge.evidence_cases import case_records
from post_thesis.llm_judge.evidence_contract import EVIDENCE_PROMPT, parse_evidence
from post_thesis.llm_judge.evidence_prompt_v2 import EVIDENCE_PROMPT_V2, evidence_request_prompt_v2
from post_thesis.llm_judge.evidence_schema_v3 import EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_WIRE_SHA256, evidence_request_v3
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prepare_pilot import build_bundle
from post_thesis.llm_judge.prompts import DEVELOPMENT_PROMPT, content_hash
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile


class EvidenceObservationsTests(unittest.TestCase):
    def test_report_preserves_one_false_negative_and_seven_reviewed_explanations(self):
        path = Path(__file__).resolve().parents[1] / 'results/post_thesis/llm_judge/evidence_diagnostic_v3_20260919.json'
        report = json.loads(path.read_text())
        self.assertEqual(content_hash(report['records']), report['records_sha256'])
        correct = 0
        explained = 0
        for row, case in zip(report['records'], case_records(), strict=True):
            self.assertEqual(row['sample_id'], case['sample_id'])
            parsed = parse_evidence(json.dumps(row['evidence']), JudgeInput(**case['input']))
            self.assertEqual(asdict(parsed), row['evidence'])
            correct += parsed.verdict == case['expected_verdict']
            explained += parsed.explanation is not None
        self.assertEqual((correct, explained), (13, 7))
        self.assertEqual(report['execution']['total_tokens'], 14378)
        self.assertEqual([r['sample_id'] for r in report['records'] if not r['verdict_matches_expected']],
                         ['absence-unknown-embedded'])

    def test_evidence_cli_rejects_custom_audit_namespace_before_artifact_reads(self):
        with patch('sys.argv', ['audit', '--expected-manifest-sha256', 'fixture',
                   '--prompt-version', EVIDENCE_PROMPT.version, '--audit-id', 'fresh-budget']), \
             patch.object(audit_module, 'code_revision') as revision, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                audit_module.main()
            self.assertEqual(error.exception.code, 2)
            revision.assert_not_called()


@unittest.skipIf(httpx is None, 'Requires isolated HTTP client')
class EvidenceAuditTests(unittest.IsolatedAsyncioTestCase):
    PROMPT = EVIDENCE_PROMPT
    REQUEST_FACTORY = staticmethod(evidence_request_v3)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'audit'
        self.bundle = build_bundle([
            {'id': str(i), 'context': f'Source {i}.', 'output': f'Answer {i}.',
             'model': 'DO_NOT_SEND_MODEL', 'task_type': 'DO_NOT_SEND_TASK', 'quality': 'good',
             'hallucination_labels_processed': {'evident_conflict': i % 2, 'baseless_info': 0}}
            for i in range(52)], revision='fixture')
        self.assertEqual(audit_module.EVIDENCE_PILOT_MANIFEST_SHA256,
                         'ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25')
        fixture_pin = patch.object(audit_module, 'EVIDENCE_PILOT_MANIFEST_SHA256', self.bundle['manifest_sha256'])
        fixture_pin.start()
        self.addCleanup(fixture_pin.stop)
        self.calls = []
        self.count = 100
        self.fail = False

    def backend(self, profile='evidence-v1'):
        p = load_profile(profile)
        def handle(request):
            self.calls.append(request)
            if request.url.path == '/version':
                return httpx.Response(200, json={'version': p['vllm_version']})
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': p['served_model_name'], 'max_model_len': p['max_model_len']}]})
            self.assertEqual(request.url.path, '/tokenize', 'Generation endpoint forbidden in token audit')
            return httpx.Response(503) if self.fail else httpx.Response(200, json={'count': self.count, 'max_model_len': p['max_model_len']})
        return VLLMBackend(profile_name=profile, transport=httpx.MockTransport(handle))

    async def execute(self, backend, **kwargs):
        return await audit_module.audit(self.bundle, expected_manifest_sha256=self.bundle['manifest_sha256'],
                                        backend=backend, directory=self.directory, revision='audit',
                                        prompt=self.PROMPT, **kwargs)

    async def test_fifty_exact_inputs_evidence_identity_and_zero_call_replay(self):
        before_bundle = deepcopy(self.bundle)
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertTrue(result['summary']['all_inputs_fit'])
            self.assertEqual(result['summary']['output_allowance_per_example'], 512)
            self.assertEqual(result['summary']['generation_calls'], 0)
            self.assertFalse(result['summary']['scoring_authorized_by_this_audit'])
            self.assertEqual(self.bundle, before_bundle)
            state = json.loads((self.directory / 'audit.json').read_text())['audit']
            identity = state['identity']
            self.assertEqual(identity['prompt'], {'version':self.PROMPT.version, 'sha256':self.PROMPT.sha256})
            self.assertEqual(identity['request_contract'], 'evidence-diagnostic-request-v3')
            self.assertEqual(identity['response_schema_json'], EVIDENCE_SCHEMA_V3_JSON)
            self.assertEqual(identity['serialized_schema_sha256'], EVIDENCE_V3_WIRE_SHA256)
            wires = [r for r in self.calls if r.url.path == '/tokenize']
            self.assertEqual(len(wires), 50)
            for wire, source in zip(wires, self.bundle['manifest']['pilot_inputs'], strict=True):
                payload = json.loads(wire.content)
                self.assertEqual(payload['messages'][0]['content'], self.PROMPT.system_text)
                self.assertEqual(json.loads(payload['messages'][1]['content']), source['input'])
                self.assertEqual(set(source['input']), {'answer', 'context'})
                request = self.REQUEST_FACTORY(JudgeInput(**source['input']), backend.config)
                self.assertEqual(state['counts'][source['sample_id']]['request_key'], request.key)
            count = len(self.calls)
            replay = await self.execute(backend)
            self.assertEqual(replay['tokenize_requests_started'], 0)
            self.assertEqual(count, len(self.calls))
            (self.directory / 'audit.json').unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(count, len(self.calls))

    async def test_wrong_profile_and_wrong_frozen_manifest_block_before_http(self):
        async with self.backend('default') as backend:
            with self.assertRaises(RunConflict):
                await self.execute(backend)
        async with self.backend() as backend:
            with patch.object(audit_module, 'EVIDENCE_PILOT_MANIFEST_SHA256', 'original-other-manifest'):
                with self.assertRaises(RunConflict):
                    await self.execute(backend)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.directory.exists())

    async def test_allowance_boundary_and_no_truncation(self):
        async with self.backend() as backend:
            self.count = 32768 - 512
            result = await self.execute(backend)
            self.assertTrue(result['summary']['all_inputs_fit'])
        self.directory = Path(self.temp.name) / 'overlength'
        async with self.backend() as backend:
            self.count += 1
            result = await self.execute(backend)
            self.assertFalse(result['summary']['all_inputs_fit'])
            self.assertEqual(len(result['summary']['overlength_ids']), 50)
            self.assertEqual(result['summary']['max_input_tokens'], self.count)
            self.assertEqual(result['summary']['generation_calls'], 0)

    async def test_terminal_failure_preserved_without_new_token_calls(self):
        self.fail = True
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['status'], 'blocked_by_terminal_attempt')
            self.assertEqual(result['tokenize_requests_started'], 1)
            self.assertEqual(result['summary']['examples_pending'], 49)
            count = len(self.calls)
            self.fail = False
            await self.execute(backend)
            self.assertEqual(count, len(self.calls))

    async def test_probability_audit_cannot_be_overwritten(self):
        async with self.backend('default') as backend:
            await audit_module.audit(self.bundle, expected_manifest_sha256=self.bundle['manifest_sha256'],
                                     backend=backend, directory=self.directory, revision='audit', prompt=DEVELOPMENT_PROMPT)
        original = (self.directory / 'audit.json').read_bytes()
        count = len(self.calls)
        async with self.backend() as backend:
            with self.assertRaises(RunConflict):
                await self.execute(backend)
        self.assertEqual((self.directory / 'audit.json').read_bytes(), original)
        self.assertEqual(len(self.calls), count)


class EvidencePromptV2AuditTests(EvidenceAuditTests):
    """Run the same audit safeguards for v2, without changing v1 expectations."""
    PROMPT = EVIDENCE_PROMPT_V2
    REQUEST_FACTORY = staticmethod(evidence_request_prompt_v2)

    async def test_v1_evidence_audit_cannot_be_reused_or_overwritten(self):
        async with self.backend() as backend:
            await audit_module.audit(self.bundle, expected_manifest_sha256=self.bundle['manifest_sha256'],
                backend=backend, directory=self.directory, revision='audit', prompt=EVIDENCE_PROMPT)
            before = (self.directory / 'audit.json').read_bytes()
            calls = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before, (self.directory / 'audit.json').read_bytes())
            self.assertEqual(calls, len(self.calls))

    async def test_fresh_prompt_audit_changes_only_prompt_identity_and_request_keys(self):
        async with self.backend() as backend:
            v1_dir = Path(self.temp.name) / 'v1'
            await audit_module.audit(self.bundle, expected_manifest_sha256=self.bundle['manifest_sha256'],
                backend=backend, directory=v1_dir, revision='audit', prompt=EVIDENCE_PROMPT)
            await self.execute(backend)
        old = json.loads((v1_dir / 'audit.json').read_text())['audit']
        new = json.loads((self.directory / 'audit.json').read_text())['audit']
        changed = {k for k in old['identity'] if old['identity'][k] != new['identity'][k]}
        self.assertEqual(changed, {'prompt'})
        self.assertEqual(list(old['counts']), list(new['counts']))
        for sid in old['counts']:
            self.assertNotEqual(old['counts'][sid]['request_key'], new['counts'][sid]['request_key'])


class EvidenceV2SelectionTests(unittest.TestCase):
    def test_fixed_namespace_and_exact_prompt_selection(self):
        self.assertEqual(audit_module.audit_prompt(EVIDENCE_PROMPT_V2.version), EVIDENCE_PROMPT_V2)
        self.assertEqual(audit_module.AUDIT_DIRECTORIES[EVIDENCE_PROMPT_V2.version],
                         'ragtruth-train-pilot-50-token-audit-evidence-prompt-v2')
        self.assertEqual(audit_module.AUDIT_DIRECTORIES[EVIDENCE_PROMPT.version],
                         'ragtruth-train-pilot-50-token-audit-evidence-v3')
        with self.assertRaises(ValueError):
            audit_module.audit_prompt('faithfulness-evidence-diagnostic-v999')

    def test_cli_rejects_old_and_custom_namespaces_before_reading_artifacts(self):
        for name in ('ragtruth-train-pilot-50-token-audit-evidence-v3', 'fresh-budget'):
            with self.subTest(name=name), patch('sys.argv', ['audit', '--expected-manifest-sha256', 'fixture',
                    '--prompt-version', EVIDENCE_PROMPT_V2.version, '--audit-id', name]), \
                 patch.object(audit_module, 'code_revision') as revision, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    audit_module.main()
                self.assertEqual(error.exception.code, 2)
                revision.assert_not_called()


if __name__ == '__main__':
    unittest.main()
