"""Apply the frozen metadata-free fusion to fresh S2/S4 TEST features; no fit."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

from .audit_test_tokens import (FUSION_REPORT_SHA256, TEST_MANIFEST_SHA256,
                                validate_inputs, validate_recovery)
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .recover_fusion import (RUN_ID as RECOVERY_RUN_ID, VERSIONS, checked_bundle,
                             features, predict_record)
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'ragtruth-fresh-metadata-free-fusion-v1'
SOURCES = {
    'S2': ('s2-ragtruth-test-fresh-v1', '8960c4a8c5d2c873b6f68241bb3c05810d6848fab392b9e8cf5407cdda4a013b'),
    'S4': ('s4-ragtruth-test-fresh-v1', '4f502fc775f15332eb78aa5070ef2d056015241d91a67f5c1ceb254ba2f2dea6'),
}
COEF = [-1.3590096505049223, 3.155528234679687]
INTERCEPT = -1.4168645142264538
THRESHOLD = float.fromhex('0x1.ccccccccccccdp-2')
ROUNDING = 'Python round(float, 4) on raw S2 min and S4 probability before fixed TRAIN normalization'


def aligned_features(s2, s4, targets):
    """Input hash/ID alignment only: labels and metadata are never projected."""
    if not targets or len(s2) != len(targets) or len(s4) != len(targets):
        raise RunConflict('complete S2/S4/target coverage required')
    left, right, projected = [], [], []
    seen = set()
    for index, (r2, r4, target) in enumerate(zip(s2, s4, targets)):
        sid, digest = target['sample_id'], target['input_sha256']
        if sid in seen or target['test_index'] != index:
            raise RunConflict('duplicate or reordered fusion target')
        seen.add(sid)
        for row in (r2, r4):
            if (row['status'] != 'ok' or row['sample_id'] != sid
                    or row['test_index'] != index or row['input_sha256'] != digest):
                raise RunConflict('fresh feature identity, order or coverage differs')
        raw, score = r2['raw_min_relevance'], r4['unsupported_score']
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (raw, score))
                or not 0 <= score <= 1):
            raise RunConflict('nonfinite, missing or invalid fresh feature')
        rounded_raw, rounded_score = round(float(raw), 4), round(float(score), 4)
        left.append({'idx': index, 'raw_min_relevance': rounded_raw})
        right.append({'idx': index, 'signal4_score': rounded_score})
        projected.append({'sample_id': sid, 'test_index': index, 'input_sha256': digest,
                          's2_raw_min_rounded': rounded_raw, 's4_probability_rounded': rounded_score})
    return left, right, projected


def fixed_model(recovery):
    result = recovery['results']
    model, selected = result['full_model'], result['threshold']['selected']
    if (model['feature_order'] != ['s2', 's4'] or model['metadata_features'] is not False
            or model['classes'] != [0, 1] or model['coef'] != COEF or model['intercept'] != INTERCEPT
            or model['s2_normalization'] != {'min': -11.43, 'max': 10.641, 'clip': [0., 1.]}
            or result['threshold']['comparator'] != '>='
            or selected['threshold'] != THRESHOLD or selected['threshold_hex'] != THRESHOLD.hex()):
        raise RunConflict('frozen fusion model or operating threshold differs')
    return model


def apply(s2, s4, targets, recovery):
    model = fixed_model(recovery)
    left, right, projected = aligned_features(s2, s4, targets)
    x = features(left, right)
    probabilities = predict_record(model, x)
    predictions = []
    for row, vector, value in zip(projected, x, probabilities):
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RunConflict('invalid frozen fusion prediction')
        predictions.append({**row, 'status': 'ok', 's2_normalized': float(vector[0]),
                            'unsupported_score': score, 'predicted_unsupported': score >= THRESHOLD})
    return predictions


def legacy_agreement(predictions, legacy):
    """Score/decision agreement only; no ground-truth labels or metric selection."""
    if len(predictions) != len(legacy): raise RunConflict('legacy fusion coverage differs')
    differences, decision_changes = [], 0
    for new, old in zip(predictions, legacy):
        if (new['sample_id'] != old['sample_id'] or new['test_index'] != old['test_index']
                or new['input_sha256'] != old['target_manifest_input_sha256']):
            raise RunConflict('legacy fusion alignment target differs')
        value = old['unsupported_score']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RunConflict('invalid preserved legacy fusion probability')
        differences.append(abs(new['unsupported_score'] - value))
        decision_changes += new['predicted_unsupported'] != (value >= THRESHOLD)
    return {'examples': len(predictions), 'maximum_absolute_difference': max(differences),
            'mean_absolute_difference': sum(differences) / len(differences),
            'fixed_threshold_decision_changes': decision_changes, 'labels_used': False,
            'historical_provenance_proven_by_agreement': False}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    versions = {name: version(name) for name in VERSIONS}
    if versions != VERSIONS: raise RunConflict('use the existing pinned .venv-judge-fit environment')
    manifest_bundle = json.loads((run_directory(TEST_RUN_ID) / 'manifest.json').read_text(encoding='utf-8'))
    validate_inputs(manifest_bundle, load_freeze())
    manifest = manifest_bundle['manifest']
    recovery_bundle = json.loads((run_directory(RECOVERY_RUN_ID) / 'report.json').read_text(encoding='utf-8'))
    validate_recovery(recovery_bundle, manifest)
    recovery = recovery_bundle['report']
    reports = {}
    for name, (run_id, digest) in SOURCES.items():
        report = checked_bundle(run_directory(run_id) / 'summary.json', 'report', 'report_sha256', digest)
        if (report['study_stage'] != 'post_thesis' or report['run_id'] != run_id
                or report['identity']['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
                or report['valid_scores'] != 2700 or report['pending'] != 0 or report['terminal_failures'] != 0):
            raise RunConflict('expected complete pinned fresh baseline report')
        reports[name] = report
    # Project only alignment identifiers. Offline target labels are not passed to application.
    targets = [{key: row[key] for key in ('sample_id', 'test_index', 'input_sha256')}
               for row in manifest['offline_rows']]
    predictions = apply(reports['S2']['predictions'], reports['S4']['predictions'], targets, recovery)
    report = {
        'study_stage': 'post_thesis', 'run_id': RUN_ID, 'code_revision': revision, 'versions': versions,
        'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'judge_freeze_sha256': FREEZE_SHA256,
        'fusion_recovery_report_sha256': FUSION_REPORT_SHA256,
        'fresh_source_report_sha256': {key: value[1] for key, value in SOURCES.items()},
        'feature_representation': ROUNDING, 'full_model': recovery['results']['full_model'],
        'threshold': {'value': THRESHOLD, 'hex': THRESHOLD.hex(), 'comparator': '>=',
                      'selection_data': 'preserved historical TRAIN meta-OOF'},
        'valid_scores': len(predictions), 'predictions': predictions,
        'legacy_score_agreement': legacy_agreement(predictions, recovery['results']['TEST_predictions']),
        'inference_input_alignment_verified': True, 'fresh_feature_provenance_linked': True,
        'comparison_ready': False, 'historical_training_provenance_verified': False,
        'fitting_calls': 0, 'transformer_calls': 0, 'judge_calls': 0,
        'TEST_label_metrics_computed': False, 'legacy_files_rewritten': False, 'HaluBench_read': False,
        'limitations': recovery['limitations'] + [
            'Fresh TEST feature provenance does not repair legacy TRAIN feature/checkpoint provenance.',
            'S2 sentence filtering and S4 truncation limit evidence visibility; inputs differ from the full-context judge.',
            'No independent probability calibration is fitted for this fusion.',
            'Final comparison readiness requires explicit treatment of these baseline limitations.'],
    }
    bundle = {'report_sha256': content_hash(report), 'report': report}
    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        created = save_once(directory / 'report.json', bundle)
    print('Post-thesis frozen fusion applied to fresh S2/S4 TEST features')
    print('New output:', created)
    print('Valid scores:', len(predictions), '/ 2700')
    print('Feature representation:', ROUNDING)
    print('Fixed TRAIN threshold: >=', THRESHOLD)
    print('Legacy score agreement:', json.dumps(report['legacy_score_agreement']))
    print('Report SHA256:', bundle['report_sha256'])
    print('Private report:', directory / 'report.json')
    print('No fitting, transformer/judge calls, TEST label metrics or legacy-file changes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
