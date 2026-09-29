"""Explicit post-thesis continuation of the pinned budget-exhausted HaluBench run."""
import argparse
from contextlib import closing
import asyncio
import json
from pathlib import Path
import sqlite3

from . import halubench_scoring as parent_contract
from . import run_halubench_scores as parent_runner
from .audit_halubench_tokens import validate_inputs
from .check_frozen_fit import load_freeze
from .halubench_comparison import SOURCES, align_sources, descriptor, COMPARISON_SHA256
from .freeze_comparison import read_report
from .label_diagnose import _load, LabelBackend, validate_records
from .label_live import installed_source
from .prepare_halubench import RUN_ID as INPUT_RUN_ID
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .run_development_scores import GLOBAL_LOCK_ID, execute_prepared
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'qwen3-halubench-test-label-score-continuation-v1'
PARENT_SHA256 = '2be55b823bb5b22e5bd5814ed578837b43e961ca2fe1a96bc0cf3f1bd0e95a3b'
PARENT_REVISION = 'd48cf349c9943a102e6db4780b44d9f5db3f729a'
PARENT_VALID = 7652
REQUEST_COUNT = 348
INPUT_TOKENS = 1589707
PLAN_PATH = Path(__file__).parent/'configs/halubench_label_continuation_v1.json'
IMPLEMENTATION_FILES = {'run_halubench_scores.py': '529161a1f19f0c084b5e6df93ae7c5e2d0a41f542ec42421a79833abf5a2d080', 'halubench_scoring.py': 'b09b74a28213eb0042eb0cda01359c33e966abced68c2ed767fc59f9b33712d5'}


def plan_descriptor():
    return {**parent_contract.load_plan(),
        'version':'halubench-label-budget-continuation-v1', 'run_id':RUN_ID,
        'scope':'explicit_budget_amendment_for_missing_canonical_HaluBench_scores',
        'parent_run_id':parent_contract.RUN_ID, 'parent_report_sha256':PARENT_SHA256,
        'parent_code_revision':PARENT_REVISION, 'parent_valid_scores_preserved':PARENT_VALID,
        'request_count':REQUEST_COUNT, 'max_attempts':REQUEST_COUNT,
        'max_output_tokens_total':REQUEST_COUNT, 'audited_input_tokens':INPUT_TOKENS,
        'client_budget_seconds':7200,
        'selection':'one interrupted request plus 347 never-attempted requests; original canonical order',
        'interrupted_parent_requests_reissued':1, 'automatic_retries':0,
        'parent_unknown_usage_preserved':True, 'parent_artifacts_rewritten':False,
        'selection_uses_labels_or_scores':False, 'parent_implementation_files':IMPLEMENTATION_FILES}


def load_plan():
    plan=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    if plan!=plan_descriptor(): raise RunConflict('continuation plan changed')
    return plan


def read_records(path, identity):
    """Read the existing journal without schema creation, recovery or updates."""
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT manifest FROM run WHERE singleton=1').fetchone()
        if row is None or json.loads(row['manifest'])!=identity:
            raise RunConflict('journal identity differs from preserved budget')
        rows=[]
        for stored in db.execute('SELECT * FROM attempts ORDER BY id'):
            row=dict(stored)
            if row['state']=='finished':
                result=json.loads(row['result'])
                if content_hash(result)!=row['result_hash']:
                    raise RunConflict('journal result checksum differs')
                row['result']=result
            elif row['state']!='interrupted' or row['result'] is not None or row['result_hash'] is not None:
                raise RunConflict('unrecovered or invalid journal record; original files preserved')
            rows.append(row)
    return rows


def select_remaining(requests, report):
    rows=report['predictions']
    if len(requests)!=8000 or len(rows)!=8000:
        raise RunConflict('complete canonical parent population required')
    selected=[];reissued=[]
    for index,(request,row) in enumerate(zip(requests,rows)):
        if (row['test_index']!=index or request['identity']['test_index']!=index
                or row['request_key']!=request['key'] or row['sample_id']!=request['sample_id']
                or row['input_sha256']!=request['identity']['input_sha256']):
            raise RunConflict('parent request/result alignment differs')
        expected='ok' if index<PARENT_VALID else 'interrupted' if index==PARENT_VALID else 'pending'
        if row['status']!=expected:
            raise RunConflict('unexpected parent status pattern; continuation not selected by score')
        if row['status']!='ok':
            selected.append(request)
            if row['status']=='interrupted': reissued.append(request['sample_id'])
    if len(selected)!=REQUEST_COUNT or sum(len(r['payload']['prompt']) for r in selected)!=INPUT_TOKENS:
        raise RunConflict('continuation request/token counts differ')
    return selected,reissued


def verify_parent(artifact_root=None):
    parent_contract.verify_implementation()
    for name,digest in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name))!=digest:
            raise RunConflict('parent implementation changed: '+name)
    directory=run_directory(parent_contract.RUN_ID,artifact_root)
    saved=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    report=saved['report']
    if saved.get('report_sha256')!=PARENT_SHA256 or content_hash(report)!=PARENT_SHA256:
        raise RunConflict('expected exact preserved partial parent summary')
    ledger=_load(directory/'budget.json');requests=_load(directory/'prepared.json');identity=ledger['identity']
    if (identity['code_revision']!=PARENT_REVISION or identity['plan']!=parent_contract.load_plan()
            or identity['requests_sha256']!=content_hash(requests)):
        raise RunConflict('parent revision, plan or prepared requests differ')
    parent_contract.validate_preparation(identity['preparation'],requests,identity['plan'])
    records=read_records(directory/'journal.sqlite3',identity)
    replay=parent_runner.summarize(requests,records,ledger)
    if replay['report_sha256']!=PARENT_SHA256 or replay['report']!=report:
        raise RunConflict('raw-response replay does not reproduce parent summary')
    if (report['valid_scores']!=PARENT_VALID or report['terminal_failures']!=1 or report['pending']!=347
            or report['halt_reason']!='TimeoutError' or report['remaining_client_seconds']!=0
            or report['charged_client_seconds']!=14405.70098266704
            or report['known_token_totals']!={'input_tokens':6204778,'output_tokens':7652}
            or report['unknown_usage_attempts']!={'input_tokens':1,'output_tokens':1}):
        raise RunConflict('parent exhaustion evidence differs')
    selected,reissued=select_remaining(requests,report)
    preparation={'source':identity['preparation']['source'],
        'parent_report_sha256':PARENT_SHA256,'parent_requests_sha256':identity['requests_sha256'],
        'selected_requests_sha256':content_hash(selected),'reissued_interrupted_sample_ids':reissued,
        'parent_known_token_totals':report['known_token_totals'],
        'parent_unknown_usage_attempts':report['unknown_usage_attempts'],
        'parent_charged_client_seconds':report['charged_client_seconds']}
    return selected,preparation


def verify_population():
    # Read-only: no parent locks, cache recovery, model loading or file rewrites.
    bundle=json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    validate_inputs(bundle,load_freeze())
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')}
             for r in bundle['manifest']['offline_rows']]
    align_sources({k:read_report(v) for k,v in SOURCES.items()},targets)
    if content_hash(descriptor())!=COMPARISON_SHA256:
        raise RunConflict('baseline comparison contract changed')


def validate_preparation(preparation,requests,plan):
    if plan!=load_plan(): raise RunConflict('unregistered continuation plan')
    selected,expected=verify_parent()
    if preparation!=expected or requests!=selected:
        raise RunConflict('continuation must use exact saved uncompleted requests')
    validate_records([],requests)


def summarize(requests,records,ledger,*,new_attempts=0):
    # Reuse unchanged raw-response/usage validation; replace only run bookkeeping.
    result=parent_runner.summarize(requests,records,ledger,new_attempts=new_attempts)
    report=result['report'];prep=ledger['identity']['preparation']
    report.update(run_id=RUN_ID,scope='post_thesis_HaluBench_explicit_budget_continuation',
        parent_report_sha256=PARENT_SHA256,parent_valid_scores_preserved=PARENT_VALID,
        canonical_examples_total=8000,combined_valid_scores=PARENT_VALID+report['valid_scores'],
        parent_artifacts_rewritten=False,explicit_parent_interrupted_requests_reissued=1,
        continuation_attempts=len(records),cumulative_attempts=PARENT_VALID+1+len(records),
        parent_unknown_usage_attempts=prep['parent_unknown_usage_attempts'],
        cumulative_known_token_totals={k:prep['parent_known_token_totals'][k]+v for k,v in report['known_token_totals'].items()},
        cumulative_unknown_usage_attempts={k:prep['parent_unknown_usage_attempts'][k]+v for k,v in report['unknown_usage_attempts'].items()},
        cumulative_charged_client_seconds=prep['parent_charged_client_seconds']+report['charged_client_seconds'])
    return {'report_sha256':content_hash(report),'report':report,'new_attempts':new_attempts}


async def execute(*,backend,revision,requests,preparation,server_record,artifact_root=None,max_new_attempts=348):
    with exclusive_run(run_directory(GLOBAL_LOCK_ID,artifact_root)):
        return await execute_prepared(plan=load_plan(),backend=backend,revision=revision,
            requests=requests,preparation=preparation,preparation_validator=validate_preparation,
            summary_builder=summarize,server_record=server_record,artifact_root=artifact_root,
            max_new_attempts=max_new_attempts,checkpoint_every=25)


def inspect_child(revision):
    directory=run_directory(RUN_ID)
    ledger=_load(directory/'budget.json');requests=_load(directory/'prepared.json');identity=ledger['identity']
    if (identity['code_revision']!=revision or identity['plan']!=load_plan()
            or identity['requests_sha256']!=content_hash(requests)):
        raise RunConflict('continuation cache identity differs')
    validate_preparation(identity['preparation'],requests,identity['plan'])
    records=read_records(directory/'journal.sqlite3',identity)
    replay=summarize(requests,records,ledger)
    saved=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if (saved.get('report_sha256')!=content_hash(saved['report'])
            or replay['report']!=saved['report']):
        raise RunConflict('child replay differs from saved summary; inspect completed invocation logs')
    return replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only',action='store_true',help='Read-only CPU validation; no third-party packages needed')
    modes.add_argument('--inspect',action='store_true',help='Read-only completed child replay; no HTTP')
    parser.add_argument('--server-session-id')
    parser.add_argument('--max-new-attempts',type=int,default=348)
    args=parser.parse_args()
    if not 0<=args.max_new_attempts<=348: parser.error('max-new-attempts must be 0..348')
    revision=code_revision();plan=load_plan();verify_population()
    requests,preparation=verify_parent()
    if args.prepare_only:
        print('Post-thesis continuation preparation passed; parent files opened read-only.')
        print('Parent valid scores preserved: 7652 / 8000')
        print('Continuation: 348 examples = 1 interrupted + 347 never attempted')
        print('Input tokens:',sum(len(r['payload']['prompt']) for r in requests))
        print('Reissued interrupted sample:',preparation['reissued_interrupted_sample_ids'])
        print('Selected requests SHA256:',preparation['selected_requests_sha256'])
        print('Plan SHA256:',content_hash(plan))
        print('Additional client budget seconds:',plan['client_budget_seconds'])
        print('Generation calls: 0; HTTP requests: 0; fitting calls: 0; files rewritten: 0')
        return 0
    if args.inspect or args.max_new_attempts==0:
        result=inspect_child(revision)
    else:
        if not args.server_session_id or not args.server_session_id.startswith('server-'):
            parser.error('provide full newly launched server-... directory name')
        if installed_source()!=preparation['source']:
            raise RunConflict('installed vLLM source changed')
        record=json.loads((run_directory(args.server_session_id)/'resources.json').read_text(encoding='utf-8'))
        async def run():
            async with LabelBackend() as backend:
                return await execute(backend=backend,revision=revision,requests=requests,
                    preparation=preparation,server_record=record,max_new_attempts=args.max_new_attempts)
        result=asyncio.run(run())
    r=result['report']
    print('Post-thesis HaluBench explicit continuation')
    print('Valid continuation scores:',r['valid_scores'],'/ 348')
    print('Combined valid scores:',r['combined_valid_scores'],'/ 8000')
    print('New attempts:',result['new_attempts'])
    print('Terminal failures / pending:',r['terminal_failures'],'/',r['pending'])
    print('Halt reason:',r['halt_reason'])
    print('Continuation token totals:',r['known_token_totals'])
    print('Cumulative unknown usage attempts:',r['cumulative_unknown_usage_attempts'])
    print('Client seconds charged / remaining:',r['charged_client_seconds'],'/',r['remaining_client_seconds'])
    print('Report SHA256:',result['report_sha256'])
    print('Private summary:',run_directory(RUN_ID)/'summary.json')
    print('Parent preserved. No TEST metrics, prompt changes, fitting or threshold changes.')
    return 0 if r['execution_complete'] else 1


if __name__=='__main__': raise SystemExit(main())
