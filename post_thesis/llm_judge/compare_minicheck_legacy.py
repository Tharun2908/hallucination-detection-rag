"""CPU-only fresh/legacy MiniCheck score agreement; no TEST label metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import load_freeze
from .minicheck_contract import SUPPORT_THRESHOLD
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'minicheck-fresh-legacy-agreement-v1'
FRESH_RUN_ID = 'minicheck-ragtruth-test-fresh-v1'
FRESH_SHA256 = '365f4d3b097ff1361bd1ccbea911c846d8b26560bd7882d1834801d79e9237f2'
LEGACY_SHA256 = '4e20383ab51a20c53934db5073d04e124e22079bc9a8bd99d5253eee7e320bcf'


def checked_fresh(bundle):
    report = bundle['report']
    if (bundle.get('report_sha256') != FRESH_SHA256 or content_hash(report) != FRESH_SHA256
            or report['run_id'] != FRESH_RUN_ID or report['study_stage'] != 'post_thesis'
            or report['identity']['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or report['valid_scores'] != 2700 or report['pending'] != 0
            or report['terminal_failures'] != 0 or report['initialization_failure']
            or report['examples_with_unknown_usage'] != 0
            or report['known_token_totals'] != {'input_tokens':13661600,'output_tokens':18935}
            or report['TEST_label_metrics_computed'] is not False):
        raise RunConflict('expected unchanged complete fresh MiniCheck report')
    return report


def probability(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise RunConflict('missing, nonfinite or invalid MiniCheck support score')
    return float(value)


def agreement(fresh, legacy, targets):
    """Project only indices/scores; never access legacy labels or task metadata."""
    if not targets or len(fresh) != len(targets) or len(legacy) != len(targets):
        raise RunConflict('complete fresh/legacy/target coverage required')
    old_by_index = {}
    for row in legacy:
        index = row['idx']
        if type(index) is not int or not 0 <= index < len(targets) or index in old_by_index:
            raise RunConflict('invalid or duplicate legacy MiniCheck index')
        old_by_index[index] = probability(row['minicheck_score'])
    changes = []
    seen = set()
    for index,(row,target) in enumerate(zip(fresh,targets)):
        if (target['sample_id'] in seen or target['test_index'] != index
                or row['status'] != 'ok' or row['test_index'] != index
                or row['sample_id'] != target['sample_id'] or row['input_sha256'] != target['input_sha256']):
            raise RunConflict('fresh MiniCheck input identity or order differs')
        seen.add(target['sample_id'])
        new,old = probability(row['support_score']),old_by_index[index]
        decision = new < SUPPORT_THRESHOLD
        if row['frozen_predicted_unsupported'] is not decision:
            raise RunConflict('saved MiniCheck decision disagrees with frozen TRAIN threshold')
        changes.append({'sample_id':target['sample_id'], 'test_index':index,
                        'fresh_support':new, 'legacy_support':old,
                        'absolute_difference':abs(new-old), 'signed_fresh_minus_legacy':new-old,
                        'fixed_threshold_decision_changed':decision != (old < SUPPORT_THRESHOLD)})
    differences = [r['absolute_difference'] for r in changes]
    decision_changes = sum(r['fixed_threshold_decision_changed'] for r in changes)
    return {'examples':len(changes), 'score_space':'support; larger means more supported',
            'matching_exactly':sum(d==0 for d in differences),
            'matching_after_rounding_fresh_to_4dp':sum(round(r['fresh_support'],4)==r['legacy_support'] for r in changes),
            'mean_absolute_difference':statistics.fmean(differences),
            'median_absolute_difference':statistics.median(differences),
            'maximum_absolute_difference':max(differences),
            'mean_signed_fresh_minus_legacy':statistics.fmean(r['signed_fresh_minus_legacy'] for r in changes),
            'differences_above_0_001':sum(d>.001 for d in differences),
            'differences_above_0_01':sum(d>.01 for d in differences),
            'differences_above_0_1':sum(d>.1 for d in differences),
            'fixed_threshold':{'value':SUPPORT_THRESHOLD, 'hex':SUPPORT_THRESHOLD.hex(), 'comparator':'<'},
            'fixed_threshold_decision_changes':decision_changes,
            'largest_differences':sorted(changes,key=lambda r:(-r['absolute_difference'],r['test_index']))[:20],
            'labels_used':False,'model_calls':0,'threshold_fitting':False,
            'historical_provenance_proven_by_agreement':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-file',type=Path,default=Path('/workspace/minicheck_results_test_7b.json'))
    args=parser.parse_args(); revision=code_revision()
    fresh=checked_fresh(json.loads((run_directory(FRESH_RUN_ID)/'summary.json').read_text(encoding='utf-8')))
    data=args.legacy_file.read_bytes()
    if hashlib.sha256(data).hexdigest()!=LEGACY_SHA256:
        raise RunConflict('expected the original audited legacy MiniCheck cache; do not overwrite it')
    legacy=json.loads(data)
    manifest=json.loads((run_directory(TEST_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples=validate_inputs(manifest,load_freeze())
    targets=[{'sample_id':ex.sample_id,'test_index':i,'input_sha256':content_hash(asdict(ex.item))}
             for i,ex in enumerate(examples)]
    result=agreement(fresh['predictions'],legacy,targets)
    report={'study_stage':'post_thesis','run_id':RUN_ID,'code_revision':revision,
            'fresh_report_sha256':FRESH_SHA256,'legacy_file_sha256':LEGACY_SHA256,
            'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'agreement':result,
            'TEST_label_metrics_computed':False,'model_calls':0,'fitting_calls':0,
            'source_files_rewritten':False,'HaluBench_read':False,
            'limitation':'Numerical agreement cannot prove historical input/checkpoint/runtime provenance; disagreement does not establish which system is correct.'}
    bundle={'report_sha256':content_hash(report),'report':report}
    directory=run_directory(RUN_ID)
    with exclusive_run(directory): created=save_once(directory/'report.json',bundle)
    print('Post-thesis MiniCheck fresh/legacy score agreement')
    print('New report:',created)
    print(json.dumps(result,indent=2))
    print('Report SHA256:',bundle['report_sha256'])
    print('Private report:',directory/'report.json')
    print('No labels used for comparison, model calls, fitting, TEST performance metrics or source rewrites.')
    return 0


if __name__=='__main__': raise SystemExit(main())
