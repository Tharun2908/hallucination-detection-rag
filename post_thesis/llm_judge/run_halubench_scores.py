"""Bounded frozen Qwen HaluBench TEST inference; no metrics or fitting."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import time

from .label_diagnose import LabelBackend, _load, validate_records
from .label_live import installed_source, known_usage, parse_response
from .prompts import content_hash
from .run_development_scores import GLOBAL_LOCK_ID, execute_prepared
from .run_pilot import charged_seconds
from .runner import run_directory
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run
from .halubench_scoring import (AUDIT_SHA256, COMPARISON_SHA256, FREEZE_SHA256,
    TEST_MANIFEST_SHA256, RUN_ID, load_plan, load_inputs, make_requests, validate_preparation)


def summarize(requests, records, ledger, *, new_attempts=0):
    validate_records(records,requests)
    plan=ledger['identity']['plan'];by_key={r['request_key']:r for r in records}
    predictions=[];totals={'input_tokens':0,'output_tokens':0};unknown=dict.fromkeys(totals,0)
    for request in requests:
        stored=by_key.get(request['key'])
        result=stored['result'] if stored and stored['state']=='finished' else None
        status=result['status'] if result else ('interrupted' if stored else 'pending')
        score=result.get('score') if result else None
        if status=='ok' and parse_response(result['response'],request)!=score:
            raise RunConflict('cached TEST score disagrees with raw response')
        predictions.append({'sample_id':request['sample_id'],'test_index':request['identity']['test_index'],
            'input_sha256':request['identity']['input_sha256'],'request_key':request['key'],
            'status':status,'score':score,'error':result.get('error') if result else None,
            'latency_seconds':result.get('latency_seconds') if result else None})
        if stored:
            usage=result.get('usage',{}) if result else {}
            if result and usage!=known_usage(result.get('response')):
                raise RunConflict('saved TEST usage disagrees with raw response')
            for key in totals:
                value=usage.get(key)
                if value is None: unknown[key]+=1
                elif type(value) is int and value>=0: totals[key]+=value
                else: raise RunConflict('invalid cached token usage')
    valid=sum(r['status']=='ok' for r in predictions);pending=sum(r['status']=='pending' for r in predictions)
    spent=charged_seconds(ledger['windows'],plan['client_budget_seconds'])
    report={'study_stage':'post_thesis','run_id':RUN_ID,'code_revision':ledger['identity']['code_revision'],
        'plan_sha256':content_hash(plan),'requests_sha256':ledger['identity']['requests_sha256'],
        'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
        'comparison_manifest_sha256':COMPARISON_SHA256,'audit_sha256':AUDIT_SHA256,
        'requests_total':len(requests),'valid_scores':valid,'pending':pending,
        'terminal_failures':len(records)-valid,'halt_reason':ledger.get('halt_reason'),
        'execution_complete':valid==plan['request_count'],'known_token_totals':totals,
        'unknown_usage_attempts':unknown,'charged_client_seconds':spent,
        'remaining_client_seconds':max(0,plan['client_budget_seconds']-spent),
        'predictions':predictions,'execution_windows':ledger['windows'],
        'preparation_windows':ledger.get('preparation_windows',[]),
        'score_space':'uncalibrated class-token margin and normalized two-label score',
        'calibration_applied':False,'calibration_fitted':False,'threshold_fitting':False,
        'TEST_metrics_computed':False,'HaluBench_read':True,'adaptation_training':False,'comparison_ready':False,
        'estimated_cost':None,'cost_basis':'unknown hourly rate; server lifetime in serving-session resources.json',
        'scope':'post_thesis_HaluBench_TEST_inference_only'}
    if new_attempts and (new_attempts%100==0 or pending==0 or ledger.get('halt_reason')):
        print(f'TEST progress: {valid}/{len(requests)} valid; {new_attempts} new attempts',flush=True)
    # Invocation-local new_attempts stays outside the immutable result hash.
    return {'report_sha256':content_hash(report),'report':report,'new_attempts':new_attempts}


async def execute(*, backend, revision, requests, preparation, server_record=None,
                  artifact_root=None, max_new_attempts=8000, preparation_seconds=0):
    with exclusive_run(run_directory(GLOBAL_LOCK_ID,artifact_root)):
        return await execute_prepared(plan=load_plan(),backend=backend,revision=revision,
            requests=requests,preparation=preparation,preparation_validator=validate_preparation,
            summary_builder=summarize,server_record=server_record,artifact_root=artifact_root,
            max_new_attempts=max_new_attempts,preparation_seconds=preparation_seconds,checkpoint_every=25)


def inspect_cached(revision, artifact_root=None):
    directory=run_directory(RUN_ID,artifact_root)
    if not (directory/'budget.json').is_file(): raise RunConflict('no cached TEST scoring run')
    with exclusive_run(run_directory(GLOBAL_LOCK_ID,artifact_root)), exclusive_run(directory):
        ledger=_load(directory/'budget.json');requests=_load(directory/'prepared.json');identity=ledger['identity']
        if (identity['code_revision']!=revision or identity['plan']!=load_plan()
                or identity['requests_sha256']!=content_hash(requests)):
            raise RunConflict('cached TEST identity differs; preserve the original run')
        validate_preparation(identity['preparation'],requests,identity['plan'])
        if not (directory/'journal.sqlite3').is_file(): raise RunConflict('missing TEST journal')
        journal=Journal(directory/'journal.sqlite3',identity)
        try:
            journal.recover();records=journal.records()
            if records and not ledger['windows']: raise RunConflict('attempts without budget reservation')
            result=summarize(requests,records,ledger)
            atomic_json(directory/'summary.json',result)
            return result
        finally: journal.close()


def prepare(examples, audited):
    from .check_label_tokenizer import load_tokenizer, package_versions, inspect_tokenizer, verify_reference
    versions=package_versions();source=installed_source()
    tokenizer,files=load_tokenizer(run_directory('label-tokenizer-cache-v1'),download=False)
    reference=verify_reference(inspect_tokenizer(tokenizer,files=files,versions=versions))
    requests=make_requests(examples,audited['audit'],tokenizer)
    preparation={'audited':audited,'comparison_manifest_sha256':COMPARISON_SHA256,
        'versions':versions,'tokenizer_files':files,'tokenizer_reference_sha256':reference,'source':source}
    validate_preparation(preparation,requests,load_plan())
    return requests,preparation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true',help='CPU preparation; no server or generation')
    mode.add_argument('--inspect',action='store_true',help='Journal replay only; no HTTP or tokenizer')
    parser.add_argument('--server-session-id')
    parser.add_argument('--max-new-attempts',type=int,default=8000)
    args=parser.parse_args()
    if not 0<=args.max_new_attempts<=8000: parser.error('max-new-attempts must be from 0 to 8000')
    revision=code_revision();directory=run_directory(RUN_ID)
    # Verify population and pinned baselines even during cache inspection.
    examples,audited,_=load_inputs()
    cached=inspect_cached(revision) if (directory/'budget.json').exists() else None
    if args.inspect or args.max_new_attempts==0:
        if cached is None: parser.error('no cached TEST run')
        result=cached
    elif cached and (cached['report']['pending']==0 or cached['report']['halt_reason']
                    or cached['report']['terminal_failures'] or cached['report']['remaining_client_seconds']<=0):
        result=cached
    else:
        started=time.monotonic();requests,preparation=prepare(examples,audited)
        preparation_seconds=time.monotonic()-started
        if args.prepare_only:
            print('Post-thesis TEST preparation passed; generation calls: 0; HTTP requests: 0')
            print('Examples / input tokens:',len(requests),'/',sum(len(r['payload']['prompt']) for r in requests))
            print('Plan SHA256:',content_hash(load_plan()))
            print('Requests SHA256:',content_hash(requests))
            print('Client budget seconds:',load_plan()['client_budget_seconds'])
            print('Build tools:',json.dumps({k:shutil.which(k) for k in ('gcc','g++','nvcc')}))
            print('CUDA 13 nvcc exists:',Path('/usr/local/cuda-13.0/bin/nvcc').is_file())
            print('HF_HOME:',os.environ.get('HF_HOME','not set'))
            print('Qwen server health and live inference not checked. Start the pinned label-score-v1 profile next.')
            return 0
        if not args.server_session_id or not args.server_session_id.startswith('server-'):
            parser.error('provide the full launcher server-... directory name')
        record=json.loads((run_directory(args.server_session_id)/'resources.json').read_text(encoding='utf-8'))
        async def run():
            async with LabelBackend() as backend:
                return await execute(backend=backend,revision=revision,requests=requests,preparation=preparation,
                    server_record=record,max_new_attempts=args.max_new_attempts,preparation_seconds=preparation_seconds)
        result=asyncio.run(run())
    report=result['report']
    print('Post-thesis frozen Qwen HaluBench TEST inference')
    print('Valid scores:',report['valid_scores'],'/',report['requests_total'])
    print('New attempts:',result['new_attempts'])
    print('Terminal failures / pending:',report['terminal_failures'],'/',report['pending'])
    print('Halt reason:',report['halt_reason'])
    print('Known token totals:',report['known_token_totals'])
    print('Unknown usage attempts:',report['unknown_usage_attempts'])
    print('Client seconds charged / remaining:',report['charged_client_seconds'],'/',report['remaining_client_seconds'])
    print('Report SHA256:',result['report_sha256'])
    print('Private summary:',directory/'summary.json')
    print('Raw scores only. No TEST metrics, fitting, threshold changes, target-domain adaptation or legacy-file changes.')
    return 0 if report['execution_complete'] else 1


if __name__=='__main__': raise SystemExit(main())
