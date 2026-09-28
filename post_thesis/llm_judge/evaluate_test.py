"""CPU-only application of frozen RAGTruth TEST metrics; no fitting or inference."""
import argparse
from importlib.metadata import version
import json
import math
import platform
from types import SimpleNamespace

from . import evaluation_math as metrics
from .audit_test_tokens import validate_inputs
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .fitting_inputs import read_journal
from .freeze_comparison import SOURCES, read_report, align_sources, RUN_ID as COMPARISON_RUN_ID
from .label_diagnose import _load, _validate_server
from .prepare_test_manifest import RUN_ID as INPUT_RUN_ID, save_once
from .prompts import content_hash
from .run_development_scores import GLOBAL_LOCK_ID
from .run_test_scores import summarize as replay_summary
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run
from .test_scoring import (AUDIT_SHA256, COMPARISON_SHA256, TEST_MANIFEST_SHA256,
    RUN_ID as JUDGE_RUN_ID, load_plan, validate_comparison, validate_preparation)

RUN_ID = 'ragtruth-frozen-four-system-evaluation-v1'
JUDGE_REPORT_SHA256 = 'ee7baf8a07368dddcce1bb80457d759dee7b1df953f8adb4f372b7327353438b'
SCORING_REVISION = '1a43d285ee8631d70aea7f2f6b2b1db18338d3c3'
REQUESTS_SHA256 = '03b6971ddea6334ed76328d6e299a163417bde925cbad6bac3a775b932d12da1'
PLAN_SHA256 = '3273a94ff8b7d079267fedb9386adba8a9baa3329d9addbd53168dc4e4d722ec'


def verify_judge(directory, targets):
    """Read-only raw-response replay under the original scoring revision."""
    bundle=json.loads((directory/'summary.json').read_text(encoding='utf-8'));report=bundle['report']
    if bundle.get('report_sha256')!=JUDGE_REPORT_SHA256 or content_hash(report)!=JUDGE_REPORT_SHA256:
        raise RunConflict('judge summary differs from completed run/replay')
    ledger=_load(directory/'budget.json');requests=_load(directory/'prepared.json')
    identity=ledger['identity'];plan=load_plan()
    if (identity['code_revision']!=SCORING_REVISION or identity['plan']!=plan
            or content_hash(plan)!=PLAN_SHA256
            or identity['requests_sha256']!=REQUESTS_SHA256 or content_hash(requests)!=REQUESTS_SHA256):
        raise RunConflict('judge scoring revision, plan or requests differ')
    validate_preparation(identity['preparation'],requests,plan)
    if len(requests)!=len(targets): raise RunConflict('judge coverage differs')
    for i,(request,target) in enumerate(zip(requests,targets)):
        if (request['sample_id']!=target['sample_id'] or request['identity']['test_index']!=i
                or request['identity']['input_sha256']!=target['input_sha256']):
            raise RunConflict('judge answer/context alignment differs')
    if not ledger['windows'] or ledger.get('halt_reason') is not None:
        raise RunConflict('missing or halted scoring execution')
    for window in ledger['windows']:
        if window['status']!='finished': raise RunConflict('unfinished scoring budget window')
        _validate_server(window['server_session_snapshot'],SimpleNamespace(profile=plan['profile']),
                         SCORING_REVISION,identity['preparation']['source'])
    records=read_journal(directory/'journal.sqlite3',identity)
    replay=replay_summary(requests,records,ledger)
    if replay['report']!=report or replay['report_sha256']!=JUDGE_REPORT_SHA256:
        raise RunConflict('judge report differs from raw response replay')
    if (len(records)!=plan['request_count'] or report['valid_scores']!=plan['request_count']
            or not report['execution_complete'] or report['pending'] or report['terminal_failures']
            or report['known_token_totals']!={'input_tokens':plan['audited_input_tokens'],'output_tokens':plan['request_count']}
            or any(report['unknown_usage_attempts'].values())
            or not 0<=report['charged_client_seconds']<=plan['client_budget_seconds']):
        raise RunConflict('judge completion, usage or budget mismatch')
    return report,{'report_sha256':JUDGE_REPORT_SHA256,'requests_sha256':REQUESTS_SHA256,
        'ledger_sha256':content_hash(ledger),'journal_records_sha256':content_hash(records),
        'scoring_code_revision':SCORING_REVISION,'raw_response_replay':'verified_read_only'}


def join_evaluation_rows(offline,judge,reports,comparison):
    """Only this offline stage joins labels; no metadata becomes a score feature."""
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')} for r in offline]
    if targets!=comparison['targets']: raise RunConflict('changed evaluation population/groups')
    projected=align_sources(reports,targets)
    if len(judge['predictions'])!=len(targets): raise RunConflict('judge score coverage differs')
    margins=[]
    for row,target in zip(judge['predictions'],targets):
        if row['status']!='ok' or any(row[k]!=target[k] for k in ('sample_id','test_index','input_sha256')):
            raise RunConflict('judge prediction order/content differs')
        value=row['score']['unsupported_log_odds']
        if type(value) not in (int,float) or not math.isfinite(value): raise RunConflict('invalid judge margin')
        margins.append(value)
    labels=[r['label'] for r in offline]
    if any(type(y) is not int or y not in (0,1) for y in labels): raise RunConflict('invalid offline labels')
    axes={'task':[r['metadata']['task_type'] for r in offline],
          'generator':[r['metadata']['model'] for r in offline]}
    if any(not isinstance(v,str) or not v for values in axes.values() for v in values):
        raise RunConflict('missing descriptive metadata')
    scores={'judge':margins,**{k:[r['score'] for r in projected[k]] for k in metrics.MAIN_SYSTEMS if k!='judge'}}
    provenance=dict(comparison['baselines'])
    # Baseline flags were verified by the pinned comparison loader; judge flags
    # below require verify_judge's raw-response replay and frozen-fit validation.
    provenance['judge']={'comparison_ready':True,**{k:True for k in metrics.PROVENANCE_FIELDS},
        'scope':'pinned frozen judge, exact TEST inputs, original server records and read-only raw-response replay',
        'report_sha256':JUDGE_REPORT_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
        'evidence_visibility':'complete answer/context; exact audited token IDs; no truncation'}
    return {'labels':labels,'scores':scores,'sample_ids':[r['sample_id'] for r in targets],
            'groups':[r['component_sha256'] for r in targets],'axes':axes,'provenance':provenance}


def efficiency(judge,reports):
    """Descriptive existing measurement scopes, not a controlled speed benchmark."""
    import numpy as np
    latencies=[r['latency_seconds'] for r in judge['predictions']]
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in latencies):
        raise RunConflict('invalid recorded judge request latency')
    n=len(latencies)
    return {'comparison':'descriptive_only_no_speedup_claim',
        'judge':{'examples':n,'client_seconds':judge['charged_client_seconds'],
            'request_latency_mean_seconds':float(np.mean(latencies)),
            'request_latency_median_seconds':float(np.median(latencies)),
            'request_latency_p95_seconds':float(np.percentile(latencies,95,method='linear')),
            'request_latency_sum_seconds':math.fsum(latencies),
            'client_seconds_per_example':judge['charged_client_seconds']/n,
            'known_token_totals':judge['known_token_totals'],'unknown_usage_attempts':judge['unknown_usage_attempts'],
            'scope':'sequential client requests; client total includes preflight/journal work, excludes preparation and server startup/idle',
            'warmup_policy':'no observations discarded; first request included',
            'server_lifetime_seconds':None,'monetary_cost':None,'cost_basis':'billable duration/rate not supplied'},
        'S4':{'recorded_timing':reports['S4']['timing'],'truncated_examples':reports['S4']['truncated_examples'],
              'scope':'shared GPU; batch and CUDA-forward sums; loading excluded'},
        'S2_dependency':{'recorded_timing':reports['S2_dependency']['timing'],
            'truncated_pairs':reports['S2_dependency']['truncated_pairs'],
            'scope':'shared GPU; example and CUDA-forward sums; loading excluded'},
        'MiniCheck_7B':{'recorded_timing':reports['MiniCheck_7B']['timing'],
            'known_token_totals':reports['MiniCheck_7B']['known_token_totals'],'workload':reports['MiniCheck_7B']['workload'],
            'scope':'per-example time; initialization recorded separately'},
        'S2_S4_metadata_free':{'fusion_application_seconds':None,
            'scope':'fresh S2/S4 measurements above; fusion application time not measured'},
        'limitations':['Native tokenizer token counts are not directly comparable compute units.',
            'No common warmup, concurrency or GPU-sharing experiment; no speedup ratios.',
            'GPU rental cost includes startup and idle time; unknown costs are not zero.']}


def compute(inputs):
    paired=metrics.paired_metrics(inputs['labels'],inputs['scores'],inputs['sample_ids'],
                                 inputs['groups'],inputs['provenance'])
    if paired.get('metrics_computed') is not True: raise RunConflict('comparison provenance gate blocked evaluation')
    slices={name:metrics.descriptive_slices(inputs['labels'],name,inputs['scores'][name],inputs['axes'])
            for name in metrics.MAIN_SYSTEMS}
    return {'paired':paired,'descriptive_slices':slices}


def evaluate_once(path,identity,inputs,observed_efficiency,limitations,compute_fn=compute):
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'));report=cached['report']
        if cached.get('report_sha256')!=content_hash(report) or report['identity']!=identity:
            raise RunConflict('changed or corrupt evaluation result; preserve the original')
        return cached,False
    result=compute_fn(inputs)
    report={'study_stage':'post_thesis','run_id':RUN_ID,'identity':identity,'results':result,
        'efficiency':observed_efficiency,'limitations':limitations,
        'generation_calls':0,'fitting_calls':0,'threshold_selection':False,
        'calibration_application':'frozen map; no refit','TEST_metrics_computed':True,
        'HaluBench_read':False,'source_files_rewritten':False,
        'implementation_timing':'benchmark loader added after inference; unchanged numerical contract frozen before inference'}
    bundle={'report_sha256':content_hash(report),'report':report};save_once(path,bundle)
    return bundle,True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision=code_revision();contract=load_contract();frozen=load_freeze()
    versions={'python':platform.python_version(),**{p:version(p) for p in contract['packages']}}
    if versions['python'].split('.')[:2]!=['3','12'] or any(versions[p]!=v for p,v in contract['packages'].items()):
        raise RunConflict('use the pinned CPU fitting environment: Python 3.12, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2')
    fit=json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8'));validate_fit(fit,frozen)
    manifest=json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    validate_inputs(manifest,frozen)
    offline=manifest['manifest']['offline_rows']
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')} for r in offline]
    reports={name:read_report(spec) for name,spec in SOURCES.items()}
    comparison_bundle=json.loads((run_directory(COMPARISON_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    comparison=validate_comparison(comparison_bundle,targets,reports)
    directory=run_directory(RUN_ID)
    with exclusive_run(run_directory(GLOBAL_LOCK_ID)), exclusive_run(run_directory(JUDGE_RUN_ID)), exclusive_run(directory):
        judge,judge_provenance=verify_judge(run_directory(JUDGE_RUN_ID),targets)
        inputs=join_evaluation_rows(offline,judge,reports,comparison)
        if len(inputs['labels'])!=2700 or len(set(inputs['groups']))!=450:
            raise RunConflict('canonical TEST population or groups changed')
        identity={'code_revision':revision,'versions':versions,'evaluation_contract_sha256':CONTRACT_SHA256,
            'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'comparison_manifest_sha256':COMPARISON_SHA256,
            'judge_freeze_sha256':FREEZE_SHA256,'fit_report_sha256':fit['report_sha256'],
            'judge':judge_provenance,'baselines':{k:v[2] for k,v in SOURCES.items()},
            'aligned_evaluation_inputs_sha256':content_hash(inputs)}
        print('All four systems aligned; frozen fit and raw judge responses verified.',flush=True)
        print('CPU evaluation: 2,000 paired group-bootstrap draws; no fitting.',flush=True)
        bundle,created=evaluate_once(directory/'report.json',identity,inputs,efficiency(judge,reports),comparison['limitations'])
    result=bundle['report']['results']['paired']
    print('Post-thesis RAGTruth TEST evaluation')
    print('New evaluation:',created)
    print('Shared examples / groups:',result['shared_n'],'/',len(result['bootstrap']['eligible_groups']))
    for name in metrics.MAIN_SYSTEMS:
        print(name,json.dumps(result['shared_points'][name]['metrics'],sort_keys=True))
    print('Paired differences: judge minus baseline; nominal 95% intervals')
    for name,values in result['paired_differences'].items():
        print(name,json.dumps({k:{f:values[k][f] for f in ('point_difference','lower','upper','valid_draws')}
                              for k in ('auroc','average_precision','f1','brier_calibrated','ece_calibrated')},sort_keys=True))
    print('Report SHA256:',bundle['report_sha256'])
    print('Private evaluation:',directory/'report.json')
    print('Frozen post-thesis TEST metrics only. No model calls, refitting, threshold changes or HaluBench access.')
    return 0


if __name__=='__main__': raise SystemExit(main())
