"""Apply the frozen RAGTruth fusion to canonical HaluBench features on CPU."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from .apply_fresh_fusion import apply, fixed_model, ROUNDING, THRESHOLD
from .audit_halubench_tokens import TEST_MANIFEST_SHA256, TEST_ROWS, validate_inputs
from .audit_test_tokens import FUSION_REPORT_SHA256
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .prepare_halubench import RUN_ID as INPUT_RUN_ID, SPLIT_SHA256
from .prepare_pilot import file_sha256
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .recover_fusion import RUN_ID as RECOVERY_RUN_ID, VERSIONS, checked_bundle
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'halubench-fresh-metadata-free-fusion-v1'
SOURCES = {
    'S2': ('s2-halubench-test-fresh-v1', 'e71c27c5c62f2b45f8332ee76c8a9bc6ba9d15915970921178a10daa47b9bb35'),
    'S4': ('s4-halubench-test-fresh-v1', '2d68d0657d983db0df84b146264652eea00034734e82d09a42126c56e4ba0f2c'),
}
IMPLEMENTATION_FILES = {
    'apply_fresh_fusion.py': 'fada182aea529684ccf97349ec986a63cb7f811d616bb9a3f560ea9504ed12c8',
    'recover_fusion.py': 'e8cae0d39620f5ef08077ee446b2a4ac47c374fa23211febf33e427a1a899494',
}


def verify_implementation():
    for name, digest in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise RunConflict('frozen fusion implementation changed: ' + name)
    return dict(IMPLEMENTATION_FILES)


def validate_source(report, name):
    if (report['study_stage'] != 'post_thesis' or report['run_id'] != SOURCES[name][0]
            or report['identity']['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or report['identity']['judge_freeze_sha256'] != FREEZE_SHA256
            or report['examples_total'] != TEST_ROWS or report['valid_scores'] != TEST_ROWS
            or len(report['predictions']) != TEST_ROWS
            or report['pending'] != 0 or report['terminal_failures'] != 0
            or report['HaluBench_read'] is not True or report['adaptation_training'] is not False):
        raise RunConflict('expected complete pinned zero-shot HaluBench ' + name + ' report')
    if name == 'S2':
        empty = [row for row in report['predictions'] if row['pairs'] == 0]
        if (report['empty_pair_examples'] != 1886 or len(empty) != 1886
                or any(row['raw_min_relevance'] != 0.0 for row in empty)):
            raise RunConflict('original S2 empty-pair raw-zero policy changed')


def build_report(manifest, s2, s4, recovery, *, revision, versions, implementation):
    """Only identifiers enter alignment; no labels or metadata enter fusion math."""
    validate_source(s2, 'S2')
    validate_source(s4, 'S4')
    fixed_model(recovery)
    targets = [{key: row[key] for key in ('sample_id', 'test_index', 'input_sha256')}
               for row in manifest['offline_rows']]
    if len(targets) != TEST_ROWS:
        raise RunConflict('all canonical HaluBench targets required')
    predictions = apply(s2['predictions'], s4['predictions'], targets, recovery)
    report = {
        'study_stage': 'post_thesis', 'run_id': RUN_ID, 'code_revision': revision,
        'versions': versions, 'implementation_files': implementation,
        'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'canonical_split_sha256': SPLIT_SHA256,
        'judge_freeze_sha256': FREEZE_SHA256, 'fusion_recovery_report_sha256': FUSION_REPORT_SHA256,
        'fresh_source_report_sha256': {key: value[1] for key, value in SOURCES.items()},
        'feature_representation': ROUNDING, 'full_model': recovery['results']['full_model'],
        'threshold': {'value': THRESHOLD, 'hex': THRESHOLD.hex(), 'comparator': '>=',
                      'selection_data': 'preserved historical RAGTruth TRAIN meta-OOF'},
        'examples_total': TEST_ROWS, 'valid_scores': len(predictions), 'predictions': predictions,
        'S2_empty_pair_examples': s2['empty_pair_examples'],
        'S2_empty_pair_policy': 'raw_min_relevance=0.0 before fixed TRAIN normalization; not a probability',
        'S2_truncated_pairs': s2['truncated_pairs'], 'S4_truncated_examples': s4['truncated_examples'],
        'inference_input_alignment_verified': True, 'fresh_feature_provenance_linked': True,
        'comparison_ready': False, 'historical_training_provenance_verified': False,
        'zero_shot_transfer': True, 'adaptation_training': False, 'canonical_membership_changed': False,
        'fitting_calls': 0, 'transformer_calls': 0, 'judge_calls': 0, 'http_requests': 0,
        'TEST_label_metrics_computed': False, 'legacy_files_rewritten': False, 'HaluBench_read': True,
        'limitations': recovery['limitations'] + [
            'Fresh HaluBench inference does not repair historical TRAIN feature/checkpoint provenance.',
            'S2 filtering yields 1886 empty-pair raw-zero fallbacks; S4 truncates 850 contexts.',
            'The feature policy is the frozen fresh RAGTruth pipeline, not a new fit to HaluBench.',
            'Fusion output is not independently calibrated on HaluBench.',
            'Canonical outer groups are disjoint, but exact passages link 994 TEST rows to adaptation components; no adaptation training is used.',
            'Final comparison readiness awaits MiniCheck and judge artifacts.'],
    }
    return {'report_sha256': content_hash(report), 'report': report}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    implementation = verify_implementation()
    versions = {name: version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RunConflict('use the existing pinned .venv-judge-fit environment')
    manifest_bundle = json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    validate_inputs(manifest_bundle, load_freeze())
    recovery = checked_bundle(run_directory(RECOVERY_RUN_ID)/'report.json',
                              'report', 'report_sha256', FUSION_REPORT_SHA256)
    reports = {name: checked_bundle(run_directory(run_id)/'summary.json', 'report', 'report_sha256', digest)
               for name, (run_id, digest) in SOURCES.items()}
    bundle = build_report(manifest_bundle['manifest'], reports['S2'], reports['S4'], recovery,
                          revision=revision, versions=versions, implementation=implementation)
    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        created = save_once(directory/'report.json', bundle)
    print('Post-thesis frozen RAGTruth fusion applied to fresh HaluBench S2/S4 features')
    print('New output:', created)
    print('Valid scores:', bundle['report']['valid_scores'], '/ 8000')
    print('Feature representation:', ROUNDING)
    print('S2 empty-pair examples: 1886; original raw-zero fallback preserved')
    print('Fixed RAGTruth TRAIN threshold: >=', THRESHOLD)
    print('Report SHA256:', bundle['report_sha256'])
    print('Private report:', directory/'report.json')
    print('No fitting, model calls, TEST label metrics, target-domain adaptation or legacy-file changes.')
    return 0


if __name__ == '__main__': raise SystemExit(main())
