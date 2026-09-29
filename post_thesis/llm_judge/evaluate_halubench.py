"""CPU-only frozen HaluBench evaluation, joining the preserved run and continuation."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace

from . import continue_halubench_scores as continuation
from . import evaluation_math as metrics
from . import halubench_comparison as comparison
from .audit_halubench_tokens import validate_inputs, TEST_MANIFEST_SHA256
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .evaluate_test import compute, efficiency
from .freeze_comparison import read_report, LEGACY_PROVENANCE, FUSION_RECOVERY
from .label_diagnose import _load, _validate_server
from .prepare_halubench import RUN_ID as INPUT_RUN_ID
from .prepare_pilot import file_sha256
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'halubench-frozen-four-system-evaluation-v1'
CONTINUATION_SHA256 = 'b3bb26a1e568bab732bace6e31d55e3610d06b4bf86a57d386b4d648dd15b937'
CONTINUATION_REVISION = '883f74b60e1e623fda83456909b52a04fb383df4'
SELECTED_REQUESTS_SHA256 = 'b182ae9188e5f44cd67e39e2e1ccf07a36b9732f325f8b91d263891d1f145f9d'
CONTINUATION_PLAN_SHA256 = '25cbbff1398fdb58e07dc69f59c5fd94ce6e393471a829592fd8214739bd262c'
SOURCE_LEVELS = {'DROP','FinanceBench','covidQA','halueval','pubmedQA'}
# Numerical functions remain under evaluation_math_v1.json's pre-inference pins.
IMPLEMENTATION_FILES = {'continue_halubench_scores.py': '9134422da031d5cd49166ca8c85d2b0457e474581ce8493731ee1b0416ad3a39', 'evaluate_test.py': 'c18e7892682b4464f585da86032f3e8ae242a35710749cb00d4e43a4e049f91e', 'halubench_comparison.py': 'c032e5871bb399dea9b96826799ce284f10759bc8230764a4f40072a0167f38f'}


def verify_implementation():
    load_contract()
    for name,digest in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name))!=digest:
            raise RunConflict('reused evaluation/replay helper changed: '+name)


def merge_predictions(parent,child,targets):
    """Canonical exact-once merge; never treat the interrupted attempt as a score."""
    if len(targets)!=8000 or len({t['sample_id'] for t in targets})!=8000:
        raise RunConflict('full canonical 8k population required')
    if len(parent['predictions'])!=8000 or len(child['predictions'])!=348:
        raise RunConflict('parent/continuation populations differ')
    selected=[]
    for i,(row,target) in enumerate(zip(parent['predictions'],targets)):
        if target['test_index']!=i or any(row[k]!=target[k] for k in ('sample_id','test_index','input_sha256')):
            raise RunConflict('parent canonical input alignment differs')
        expected='ok' if i<7652 else 'interrupted' if i==7652 else 'pending'
        if row['status']!=expected: raise RunConflict('unexpected parent completion pattern')
        if expected=='ok': selected.append(row)
    for row,target in zip(child['predictions'],targets[7652:]):
        if row['status']!='ok' or any(row[k]!=target[k] for k in ('sample_id','test_index','input_sha256')):
            raise RunConflict('continuation canonical input alignment differs')
        selected.append(row)
    if len(selected)!=8000 or len({r['sample_id'] for r in selected})!=8000:
        raise RunConflict('duplicate or missing combined judge score')
    for row in selected:
        value=row['score']['unsupported_log_odds']
        if type(value) not in (int,float) or not math.isfinite(value):
            raise RunConflict('nonfinite or invalid judge margin')
    return selected


def verify_judge(targets):
    verify_implementation()
    requests,preparation=continuation.verify_parent()
    if (content_hash(requests)!=SELECTED_REQUESTS_SHA256
            or content_hash(continuation.load_plan())!=CONTINUATION_PLAN_SHA256):
        raise RunConflict('registered continuation requests or plan changed')
    child_bundle=continuation.inspect_child(CONTINUATION_REVISION)
    child=child_bundle['report']
    if (child_bundle['report_sha256']!=CONTINUATION_SHA256
            or content_hash(child)!=CONTINUATION_SHA256
            or not child['execution_complete'] or child['valid_scores']!=348
            or child['terminal_failures'] or child['pending'] or child['halt_reason'] is not None
            or child['known_token_totals']!={'input_tokens':1589707,'output_tokens':348}
            or any(child['unknown_usage_attempts'].values())
            or not 0<=child['charged_client_seconds']<=7200):
        raise RunConflict('completed continuation and identical replay required')
    parent_directory=run_directory(continuation.parent_contract.RUN_ID)
    parent_bundle=json.loads((parent_directory/'summary.json').read_text(encoding='utf-8'))
    parent=parent_bundle['report']
    if content_hash(parent)!=continuation.PARENT_SHA256:
        raise RunConflict('parent changed after read-only replay')
    provenance={}
    for name,run_id,revision in (
            ('parent',continuation.parent_contract.RUN_ID,continuation.PARENT_REVISION),
            ('continuation',continuation.RUN_ID,CONTINUATION_REVISION)):
        directory=run_directory(run_id);ledger=_load(directory/'budget.json');identity=ledger['identity']
        if identity['code_revision']!=revision or not ledger['windows']:
            raise RunConflict('original scoring identity or execution windows missing')
        for window in ledger['windows']:
            if window['status']!='finished': raise RunConflict('unclosed scoring execution window')
            _validate_server(window['server_session_snapshot'],SimpleNamespace(profile=identity['plan']['profile']),
                             revision,identity['preparation']['source'])
        records=continuation.read_records(directory/'journal.sqlite3',identity)
        provenance[name]={'run_id':run_id,'scoring_code_revision':revision,
            'ledger_sha256':content_hash(ledger),'journal_records_sha256':content_hash(records),
            'raw_response_replay':'verified_read_only'}
    if (child['cumulative_known_token_totals']!={'input_tokens':7794485,'output_tokens':8000}
            or child['cumulative_unknown_usage_attempts']!={'input_tokens':1,'output_tokens':1}
            or child['cumulative_attempts']!=8001 or child['combined_valid_scores']!=8000
            or child['cumulative_charged_client_seconds']!=parent['charged_client_seconds']+child['charged_client_seconds']):
        raise RunConflict('combined attempt, usage or time accounting differs')
    rows=merge_predictions(parent,child,targets)
    provenance.update(parent_report_sha256=continuation.PARENT_SHA256,
        continuation_report_sha256=CONTINUATION_SHA256,selected_requests_sha256=SELECTED_REQUESTS_SHA256,
        continuation_plan_sha256=CONTINUATION_PLAN_SHA256,combined_predictions_sha256=content_hash(rows),
        canonical_scores=8000,total_attempts=8001,original_interrupted_attempt_preserved=True)
    # This is an in-memory projection, never a replacement of either source summary.
    combined={'predictions':rows,'charged_client_seconds':child['cumulative_charged_client_seconds'],
        'known_token_totals':child['cumulative_known_token_totals'],
        'unknown_usage_attempts':child['cumulative_unknown_usage_attempts']}
    return combined,provenance


def verify_baselines(targets):
    reports={name:read_report(spec) for name,spec in comparison.SOURCES.items()}
    comparison.align_sources(reports,targets)
    expected=comparison.descriptor()
    saved=json.loads((run_directory(comparison.RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    if (saved.get('manifest_sha256')!=comparison.COMPARISON_SHA256
            or content_hash(saved['manifest'])!=comparison.COMPARISON_SHA256 or saved['manifest']!=expected):
        raise RunConflict('original fresh comparison manifest differs')
    legacy=read_report(LEGACY_PROVENANCE);recovery=read_report(FUSION_RECOVERY)
    for name,old in [('S4','S4'),('MiniCheck_7B','MC_7B')]:
        rule=load_contract()['operating_rules'][name];entry=legacy['threshold_reproduction'][old]
        if (entry['status']!='matches_historical_TRAIN_audit'
                or entry['reproduced']['threshold']!=rule['threshold'] or entry['comparator']!=rule['comparator']):
            raise RunConflict('historical TRAIN threshold changed')
    fusion=reports['S2_S4_metadata_free']
    if (fusion['full_model']!=recovery['results']['full_model']
            or fusion['full_model']['metadata_features'] is not False
            or fusion['threshold']['value']!=.45 or fusion['threshold']['comparator']!='>='
            or fusion['fusion_recovery_report_sha256']!=FUSION_RECOVERY[2]
            or fusion['fresh_source_report_sha256']!={'S2':comparison.SOURCES['S2_dependency'][2],'S4':comparison.SOURCES['S4'][2]}):
        raise RunConflict('frozen fusion lineage changed')
    return reports,expected


def join_evaluation_rows(offline,judge,reports,descriptor):
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')} for r in offline]
    comparison.align_sources(reports,targets)
    if len(judge['predictions'])!=8000: raise RunConflict('incomplete combined judge')
    for row,target in zip(judge['predictions'],targets):
        if row['status']!='ok' or any(row[k]!=target[k] for k in ('sample_id','test_index','input_sha256')):
            raise RunConflict('combined judge input alignment differs')
    labels=[r['label'] for r in offline];sources=[r['metadata']['source'] for r in offline]
    if any(type(y) is not int or y not in (0,1) for y in labels): raise RunConflict('invalid offline labels')
    if set(sources)!=SOURCE_LEVELS: raise RunConflict('canonical HaluBench source levels differ')
    scores={'judge':[r['score']['unsupported_log_odds'] for r in judge['predictions']]}
    provenance={}
    for name in metrics.MAIN_SYSTEMS:
        if name!='judge':
            spec=comparison.SOURCES[name];scores[name]=[r[spec[3]] for r in reports[name]['predictions']]
            if descriptor['baselines'][name]['comparison_ready'] is not True:
                raise RunConflict('baseline readiness is not established')
        provenance[name]={'comparison_ready':True,**{k:True for k in metrics.PROVENANCE_FIELDS},
            'scope':'verified fresh-runtime artifacts and exact canonical inputs; frozen RAGTruth policies',
            'historical_training_provenance_independently_verified':False,
            'evidence_visibility':'documented native preprocessing/truncation; not identical evidence windows'}
    return {'labels':labels,'sample_ids':[r['sample_id'] for r in offline],
        'groups':[r['component_sha256'] for r in offline], 'scores':scores,
        'axes':{'source':sources},'provenance':provenance}


def observed_efficiency(judge,reports):
    result=efficiency(judge,reports)
    result['judge'].update(total_client_attempts=8001,unique_valid_scores=8000,
        request_latency_scope='8000 successful requests only; interrupted request latency unavailable',
        client_total_scope='both parent and continuation, including interrupted window; preparation/server startup/idle excluded')
    result['S2_dependency']['empty_pair_examples']=1886
    result['limitations'].append('One original interrupted attempt retains unknown token usage; known totals are incomplete cost accounting.')
    return result


def evaluate_once(path,identity,inputs,timing,limitations,compute_fn=compute):
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'));report=saved['report']
        if saved.get('report_sha256')!=content_hash(report) or report['identity']!=identity:
            raise RunConflict('changed evaluation identity or corrupted report; preserve original')
        return saved,False
    result=compute_fn(inputs)
    report={'study_stage':'post_thesis','run_id':RUN_ID,'identity':identity,'results':result,
        'efficiency':timing,'limitations':limitations,'generation_calls':0,'fitting_calls':0,
        'threshold_selection':False,'calibration_application':'frozen RAGTruth map; no HaluBench refit',
        'TEST_metrics_computed':True,'HaluBench_read':True,'adaptation_training':False,
        'source_files_rewritten':False,'implementation_timing':'loader added after scoring; numerical contract frozen before inference'}
    bundle={'report_sha256':content_hash(report),'report':report};save_once(path,bundle)
    return bundle,True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',help='Read-only verification; no numerical packages, metrics or writes')
    args=parser.parse_args();revision=code_revision();verify_implementation();contract=load_contract();frozen=load_freeze()
    fit=json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8'));validate_fit(fit,frozen)
    manifest=json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'));validate_inputs(manifest,frozen)
    offline=manifest['manifest']['offline_rows']
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')} for r in offline]
    reports,desc=verify_baselines(targets);judge,judge_provenance=verify_judge(targets)
    inputs=join_evaluation_rows(offline,judge,reports,desc)
    if len(inputs['labels'])!=8000 or len(set(inputs['groups']))!=7198:
        raise RunConflict('canonical population/bootstrap groups changed')
    print('All four systems aligned: 8000 examples / 7198 bootstrap components.',flush=True)
    print('Parent + continuation raw responses verified; 8001 attempts / 8000 scores; one unknown-usage interruption preserved.',flush=True)
    if args.verify_only:
        print('Verification passed; generation calls: 0; fitting calls: 0; metrics computed: False; files rewritten: 0')
        return 0
    versions={'python':platform.python_version(),**{p:version(p) for p in contract['packages']}}
    if versions['python'].split('.')[:2]!=['3','12'] or any(versions[p]!=v for p,v in contract['packages'].items()):
        raise RunConflict('use pinned CPU environment: Python 3.12, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2')
    identity={'code_revision':revision,'versions':versions,'evaluation_contract_sha256':CONTRACT_SHA256,
        'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'comparison_manifest_sha256':comparison.COMPARISON_SHA256,
        'judge_freeze_sha256':FREEZE_SHA256,'fit_report_sha256':fit['report_sha256'],'judge':judge_provenance,
        'baselines':{k:v[2] for k,v in comparison.SOURCES.items()},'aligned_evaluation_inputs_sha256':content_hash(inputs)}
    limitations=desc['limitations']+[
        'Judge inference completed in a preserved budget-exhausted parent and an explicit 348-request continuation.',
        'The original interrupted attempt retains unknown usage; reported successful token totals are not complete billing.',
        'RAGTruth calibration and thresholds are transferred unchanged; no HaluBench calibration or selection.',
        'Source slices are descriptive, without separate bootstrap intervals or post-hoc tuning.']
    directory=run_directory(RUN_ID)
    with exclusive_run(directory):
        print('CPU evaluation: 2000 paired group-bootstrap draws; frozen calibration/thresholds, no fitting.',flush=True)
        bundle,created=evaluate_once(directory/'report.json',identity,inputs,observed_efficiency(judge,reports),limitations)
    result=bundle['report']['results']['paired']
    print('Post-thesis frozen HaluBench TEST evaluation; new evaluation:',created)
    print('Shared examples / groups:',result['shared_n'],'/',len(result['bootstrap']['eligible_groups']))
    for name in metrics.MAIN_SYSTEMS: print(name,json.dumps(result['shared_points'][name]['metrics'],sort_keys=True))
    print('Paired differences: judge minus baseline; nominal 95% intervals')
    for name,values in result['paired_differences'].items():
        print(name,json.dumps({k:{f:values[k][f] for f in ('point_difference','lower','upper','valid_draws')}
            for k in ('auroc','average_precision','f1','brier_calibrated','ece_calibrated')},sort_keys=True))
    print('Descriptive source metrics (no slice-specific fitting or intervals):')
    for name in metrics.MAIN_SYSTEMS:
        for source,row in bundle['report']['results']['descriptive_slices'][name]['source'].items():
            print(name,source,json.dumps({'n':row['n'],'metrics':row['metrics']},sort_keys=True))
    print('Report SHA256:',bundle['report_sha256']);print('Private evaluation:',directory/'report.json')
    print('Frozen post-thesis evaluation only. No model calls, fitting, target-domain adaptation or source rewrites.')
    return 0


if __name__=='__main__': raise SystemExit(main())
