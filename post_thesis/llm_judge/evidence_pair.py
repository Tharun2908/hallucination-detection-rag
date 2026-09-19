"""Bounded paired synthetic comparison: fixed prompts, one attempt per case/arm."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .check_evidence_prompt_v2 import check as check_candidate
from .evidence_cases import examples as historical_examples
from .evidence_contract import EVIDENCE_PROMPT
from .evidence_diagnose import (EVIDENCE_V2_HALT_CODES, PrecountedBackend, load_plan as historical_plan,
                               validate_plan as validate_historical_plan, validate_v2_server)
from .evidence_prompt_v2 import EVIDENCE_PROMPT_V2, evidence_request_prompt_v2
from .evidence_prompt_v2_cases import CASE_VERSION, case_records, cases_sha256, examples
from .evidence_runner import run_evidence
from .evidence_schema_v3 import EVIDENCE_V3_FIELD_ORDER, EVIDENCE_V3_WIRE_SHA256, evidence_request_v3
from .judge import BackendError
from .prompts import content_hash
from .run_pilot import validate_server
from .runner import run_directory
from .serve import code_revision, resource_totals
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import VLLMBackend

PLAN_PATH = Path(__file__).with_name('configs') / 'evidence_prompt_pair_v1.json'
ARMS = ('v1', 'v2')
FACTORIES = {'v1': evidence_request_v3, 'v2': evidence_request_prompt_v2}
BUDGET_SECONDS = 600


def expected_plan():
    return {'study_stage': 'post_thesis', 'scope': 'paired_synthetic_development_only',
            'run_id': 'qwen3-evidence-prompt-pair-v1', 'case_version': CASE_VERSION,
            'cases_sha256': cases_sha256(), 'examples_per_arm': 26, 'max_generation_attempts': 52,
            'arms': {a: {'prompt_version': p.version, 'prompt_sha256': p.sha256}
                     for a, p in zip(ARMS, (EVIDENCE_PROMPT, EVIDENCE_PROMPT_V2))},
            'schema_version': 'v3', 'serialized_schema_sha256': EVIDENCE_V3_WIRE_SHA256,
            'expected_response_field_order': list(EVIDENCE_V3_FIELD_ORDER),
            'profile_name': 'evidence-v1',
            'profile_sha256': '4d7a4a3b0d534240e2d87c287faa97804b44320e40c2097525615975bdf2aa28',
            'structured_output_backend_policy': 'unchanged_vllm_default_auto',
            'case_order': 'frozen_fixture_order', 'arm_order': 'v1_then_v2_even_case_index_reverse_odd',
            'client_budget_seconds': BUDGET_SECONDS, 'generation_invocations': 1,
            'repeat_policy': 'cache_inspection_only_after_first_invocation',
            'unknown_window_policy': 'charge_full_600_seconds_and_disable_generation',
            'max_attempts_per_case_per_arm': 1, 'concurrency': 1, 'request_timeout_seconds': 60,
            'max_input_tokens_per_request': 4096, 'max_input_tokens_total': 212992,
            'max_output_tokens_per_request': 512, 'max_output_tokens_total': 26624,
            'max_tokenization_requests': 104, 'full_audit_before_generation': True,
            'truncation': 'none', 'halt_on_error_codes': list(EVIDENCE_V2_HALT_CODES)}


def load_plan():
    return json.loads(PLAN_PATH.read_text(encoding='utf-8'))


def validate_plan(plan, backend):
    check_candidate()
    # Preserve the existing schema-v3 compatibility evidence and serving policy.
    validate_historical_plan(historical_plan('v3'), backend, historical_examples(), 'v3')
    if (content_hash(plan) != content_hash(expected_plan())
            or content_hash(backend.profile) != plan['profile_sha256']
            or backend.config.max_output_tokens != 512 or backend.timeout_seconds != 60):
        raise RunConflict('changed paired plan, prompt, fixtures or serving profile')


def schedule():
    return [(index, arm) for index in range(26)
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]


def _save(path, state):
    atomic_json(path, {'execution_sha256': content_hash(state), 'execution': state})


def _validate_state(state, requests):
    try:
        status, audit = state['status'], state['input_audit']
        if status not in ('ready', 'started', 'finished') or audit['status'] not in ('pending', 'started', 'completed', 'failed'):
            raise ValueError
        counts = audit['counts']
        if not isinstance(counts, dict) or not set(counts) <= set(requests):
            raise ValueError
        for row in counts.values():
            if row['status'] not in ('started', 'ok', 'failed'):
                raise ValueError
            if row['status'] == 'ok' and (type(row['input_tokens']) is not int or not 1 <= row['input_tokens'] <= 4096):
                raise ValueError
        if audit['status'] == 'completed' and (set(counts) != set(requests) or any(r['status'] != 'ok' for r in counts.values())):
            raise ValueError
        if status == 'ready':
            if state['resources'] is not None or audit['status'] != 'pending' or counts:
                raise ValueError
        elif type(state['reserved_seconds']) is not int or state['reserved_seconds'] != BUDGET_SECONDS:
            raise ValueError
        if status == 'finished':
            resource_totals(state['resources']['wall_seconds'], 1)
            if state['outcome'] == 'invocation_completed' and audit['status'] != 'completed':
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RunConflict('invalid paired execution or audit state') from None


def comparisons(reports):
    indexed = {a: {p['sample_id']: p for p in reports[a]['predictions']} for a in ARMS}
    paired = []
    for case in case_records():
        row = {'sample_id': case['sample_id'], 'family': case['family'],
               'expected_verdict': case['expected_verdict'], 'expected_issue_type': case['expected_issue_type']}
        for arm in ARMS:
            p = indexed[arm][case['sample_id']]
            e = p['evidence']
            row[arm] = {'status': p['status'], 'verdict': p['verdict'],
                        'issue_type': e['issue_type'] if e else None,
                        'verdict_matches_expected': e['verdict'] == case['expected_verdict'] if e else None,
                        'issue_matches_expected': e['issue_type'] == case['expected_issue_type'] if e else None,
                        'semantic_review': 'pending' if e else 'not_available'}
        paired.append(row)
    return paired


def _summary(state, plan, reports, before, revision):
    paired = comparisons(reports)
    arm_summary = {}
    for arm in ARMS:
        insufficient = [p['evidence'] for p in reports[arm]['predictions']
                        if p['evidence'] and p['evidence']['issue_type'] == 'insufficient_support']
        arm_summary[arm] = {'valid': reports[arm]['examples_valid'], 'total': 26,
            'verdict_matches': sum(row[arm]['verdict_matches_expected'] is True for row in paired),
            'issue_matches': sum(row[arm]['issue_matches_expected'] is True for row in paired),
            'valid_insufficient_support': len(insufficient),
            'null_context_quotes_among_valid_insufficient_support': sum(e['context_quote'] is None for e in insufficient)}
    charged = (state['resources']['wall_seconds'] if state['status'] == 'finished'
               else BUDGET_SECONDS if state['status'] == 'started' else 0)
    return {'study_stage': 'post_thesis', 'kind': 'paired_synthetic_evidence_development',
            'code_revision': revision, 'execution_plan_sha256': content_hash(plan),
            'execution_status': state['status'], 'outcome': state.get('outcome'), 'error_code': state.get('error_code'),
            'charged_client_seconds': charged, 'generation_invocation_available': state['status'] == 'ready',
            'new_attempts_this_invocation': sum(r['attempts_total'] for r in reports.values()) - before,
            'input_audit': state['input_audit'], 'reports': reports, 'comparisons': paired, 'arm_summary': arm_summary,
            'semantic_review': 'pending', 'server_startup_and_idle_excluded': True, 'rental_cost': None}


async def execute(*, backend, revision, server_record=None, artifact_root=None,
                  inspection_only=False, clock=time.monotonic):
    plan = load_plan()
    validate_plan(plan, backend)
    cases = examples()
    requests_by_arm = {a: [FACTORIES[a](ex.item, backend.config) for ex in cases] for a in ARMS}
    requests = {r.key: r for rs in requests_by_arm.values() for r in rs}
    if len(cases) != 26 or len(requests) != 52:
        raise RunConflict('expected 26 distinct requests for each of two prompts')
    directory = run_directory(plan['run_id'], artifact_root)
    control = directory / 'execution'
    identity = {'plan': plan, 'code_revision': revision, 'config': asdict(backend.config)}
    with exclusive_run(control):
        path = control / 'budget.json'
        arm_runs = {a: plan['run_id'] + '-prompt-' + a for a in ARMS}
        journal_paths = [run_directory(arm_runs[a], artifact_root) / 'journal.sqlite3' for a in ARMS]
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            state = saved['execution']
            if saved['execution_sha256'] != content_hash(state) or state['identity'] != identity:
                raise RunConflict('changed or corrupt paired execution identity')
            _validate_state(state, requests)
            if state['status'] != 'ready' and not all(p.exists() for p in journal_paths):
                raise RunConflict('missing paired journal; preserve both arms')
        else:
            if any(p.exists() for p in journal_paths):
                raise RunConflict('missing paired budget; do not reset the experiment')
            state = {'identity': identity, 'status': 'ready', 'resources': None,
                     'input_audit': {'status': 'pending', 'counts': {}}}
            _save(path, state)
        guarded = PrecountedBackend(backend, state['input_audit']['counts'], EVIDENCE_V2_HALT_CODES)

        async def run(arm, cap):
            return await run_evidence(cases, run_id=arm_runs[arm], config=backend.config,
                backend=guarded, revision=revision, dataset_revision=CASE_VERSION, artifact_root=artifact_root,
                max_new_attempts=cap, halt_codes=EVIDENCE_V2_HALT_CODES, request_factory=FACTORIES[arm],
                expected_response_order=EVIDENCE_V3_FIELD_ORDER)

        reports = {a: await run(a, 0) for a in ARMS}
        before = sum(r['attempts_total'] for r in reports.values())
        if before and (state['status'] == 'ready' or state['input_audit']['status'] != 'completed'):
            raise RunConflict('attempts exist without a reserved invocation and complete audit')
        caught = None
        if state['status'] == 'ready' and not inspection_only:
            validate_server(server_record or {}, backend, revision)
            validate_v2_server(server_record or {}, 'v3')
            start = clock()
            state.update(status='started', reserved_seconds=BUDGET_SECONDS,
                         started_at=datetime.now(timezone.utc).isoformat(), server_session_snapshot=server_record)
            _save(path, state)

            async def work():
                state['server_preflight'] = await backend.preflight()
                state['input_audit']['status'] = 'started'
                _save(path, state)
                for index, arm in schedule():
                    request = requests_by_arm[arm][index]
                    row = {'status': 'started', 'input_tokens': None}
                    guarded.counts[request.key] = row
                    _save(path, state)
                    try:
                        count = await backend.count_input_tokens(request)
                        if type(count) is not int or count < 1:
                            raise BackendError('invalid_token_count')
                        row['input_tokens'] = count
                        if count > 4096 or count + request.config.max_output_tokens > backend.profile['max_model_len']:
                            raise BackendError('diagnostic_input_limit')
                    except BackendError as error:
                        row.update(status='failed', error_code=error.code)
                        state['input_audit']['status'] = 'failed'
                        _save(path, state)
                        raise
                    row['status'] = 'ok'
                    _save(path, state)
                state['input_audit']['status'] = 'completed'
                _save(path, state)
                # Balance which prompt goes first for each case; fixed before results.
                for index, arm in schedule():
                    if reports[arm]['attempts_total'] != index:
                        raise RunConflict('paired attempt order changed')
                    reports[arm] = await run(arm, 1)
                    if guarded.halt_code:
                        return

            outcome, error_code = 'invocation_completed', None
            try:
                await asyncio.wait_for(work(), timeout=BUDGET_SECONDS)
                if guarded.halt_code:
                    outcome, error_code = 'halted_on_serving_or_alignment_error', guarded.halt_code
            except asyncio.TimeoutError:
                outcome, error_code = 'deadline_reached', 'paired_diagnostic_deadline'
            except BackendError as error:
                outcome, error_code = 'preflight_or_audit_failed', error.code
            except BaseException as error:
                outcome, error_code = 'interrupted_or_error', type(error).__name__
                caught = error
            finally:
                try:
                    for a in ARMS:
                        reports[a] = await run(a, 0)
                finally:
                    state.update(status='finished', outcome=outcome, error_code=error_code,
                                 finished_at=datetime.now(timezone.utc).isoformat(),
                                 resources=resource_totals(clock() - start, 1))
                    _save(path, state)
        result = _summary(state, plan, reports, before, revision)
        atomic_json(directory / 'paired_summary.json', result)
        if caught is not None:
            raise caught
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-session-id')
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    revision = code_revision()
    record = None
    if args.server_session_id:
        if not args.server_session_id.startswith('server-'):
            parser.error('expected the full server-... directory name, including server-')
        record = json.loads((run_directory(args.server_session_id) / 'resources.json').read_text(encoding='utf-8'))

    async def start():
        async with VLLMBackend(profile_name='evidence-v1', api_key=os.environ.get('JUDGE_API_KEY')) as backend:
            return await execute(backend=backend, revision=revision, server_record=record,
                                 inspection_only=args.inspect_only)

    result = asyncio.run(start())
    print('Post-thesis paired synthetic evidence comparison; schema v3')
    print('Execution plan SHA256:', result['execution_plan_sha256'])
    print('Fixtures SHA256:', cases_sha256())
    print('Execution:', result['execution_status'], result['outcome'], result['error_code'])
    print('Input audit:', result['input_audit']['status'])
    print('New attempts:', result['new_attempts_this_invocation'])
    print('Client seconds charged:', result['charged_client_seconds'])
    for arm in ARMS:
        report = result['reports'][arm]
        prompt = load_plan()['arms'][arm]
        print('Prompt:', prompt['prompt_version'], prompt['prompt_sha256'])
        print('Arm', arm, json.dumps(result['arm_summary'][arm], sort_keys=True))
        print('Terminal failures / pending:', report['examples_terminal_failure'], '/', report['examples_pending'])
        print('Known token totals:', report['known_token_totals'])
        print('Unknown usage attempts:', report['attempts_with_unknown_tokens'])
        print('Raw response order:', json.dumps(report['response_order_summary'], sort_keys=True))
    print('Private paired summary:', run_directory(load_plan()['run_id']) / 'paired_summary.json')
    print('Manual semantic review pending. Synthetic development only; no benchmark data or probabilities.')
    return 0 if all(r['examples_valid'] == 26 and r['response_order_summary']['matches'] == 26
                    for r in result['reports'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
