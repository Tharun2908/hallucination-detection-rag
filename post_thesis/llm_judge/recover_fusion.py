"""Post-thesis CPU reconstruction of the historical metadata-free S2+S4 fusion."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import warnings

from . import audit_baseline_provenance as audit
from .check_frozen_fit import FIT_RUN_ID, load_freeze, validate_fit
from .prepare_pilot import TRAIN_SHA256, _records, file_sha256, read_train
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'ragtruth-metadata-free-fusion-recovery-v1'
PROVENANCE_SHA256 = '2e85cc46aeb4ade600c3c70680c200338f580dd20875a1a122d71f0988575e7c'
PROVENANCE_REVISION = '84a9b48158476da32844d9679bce52b9e28cdbbd'
REFERENCE_SHA256 = '7360563c16c5de872148537d253688e6ac3b5e4f8cefd2f95af44569386cc5fb'
SOURCE_SHA256 = 'af0b30a3397754e00b877a8fb06d5e5b5351e1ab2f6e8d878502e175a050ff2d'
VERSIONS = {'numpy': '1.26.4', 'scipy': '1.14.1', 'scikit-learn': '1.5.2'}
S2_MIN, S2_MAX = -11.430, 10.641
CACHE_NAMES = ('relevance_results_train_v2.json', 'signal4_results_train_oof.json',
               'relevance_results_test_v2.json', 'signal4_results_test.json')
# The original sklearn 1.5.2 defaults, made explicit for reconstruction.
LR_PARAMS = dict(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                 intercept_scaling=1, l1_ratio=None, max_iter=1000,
                 multi_class='deprecated', n_jobs=None, penalty='l2',
                 random_state=42, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)


def checked_bundle(path, payload_key, hash_key, digest):
    bundle = json.loads(Path(path).read_text(encoding='utf-8'))
    if bundle.get(hash_key) != digest or content_hash(bundle[payload_key]) != digest:
        raise RunConflict('pinned artifact identity mismatch: ' + str(path))
    return bundle[payload_key]


def features(s2_rows, s4_rows):
    """Two columns only. Row ordering/identity is validated before this function."""
    import numpy as np
    if not s2_rows or len(s2_rows) != len(s4_rows):
        raise RunConflict('fusion requires complete paired score rows')
    result = []
    for index, (r2, r4) in enumerate(zip(s2_rows, s4_rows)):
        if r2['idx'] != index or r4['idx'] != index:
            raise RunConflict('features require canonical contiguous indices')
        raw, s4 = r2['raw_min_relevance'], r4['signal4_score']
        if (type(raw) not in (int, float) or not math.isfinite(raw)
                or type(s4) not in (int, float) or not math.isfinite(s4) or not 0 <= s4 <= 1):
            raise RunConflict('fusion requires finite nonmissing scores')
        s2 = float(max(0.0, min(1.0, (float(raw) - S2_MIN) / (S2_MAX - S2_MIN))))
        result.append([s2, float(s4)])
    return np.array(result, dtype=float)


def select_threshold(probs, labels):
    import numpy as np
    from sklearn.metrics import f1_score
    candidates = []
    best = None
    for threshold in np.arange(0.05, 0.96, 0.05):
        predictions = (probs >= threshold).astype(int)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        item = {'threshold': float(threshold), 'threshold_hex': float(threshold).hex(),
                'f1': f1, 'tp': int(((predictions == 1) & (labels == 1)).sum()),
                'fp': int(((predictions == 1) & (labels == 0)).sum()),
                'fn': int(((predictions == 0) & (labels == 1)).sum())}
        candidates.append(item)
        if best is None or f1 > best['f1']:
            best = item
    return {'selected': best, 'comparator': '>=', 'space': 'fusion_unsupported_probability',
            'selection_data': 'TRAIN_meta_OOF', 'tie_rule': 'first_maximum_in_ascending_grid',
            'candidates': candidates}


def model_record(model):
    return {'feature_order': ['s2', 's4'], 'metadata_features': False,
            's2_normalization': {'min': S2_MIN, 'max': S2_MAX, 'clip': [0.0, 1.0]},
            'classes': model.classes_.tolist(), 'coef': model.coef_[0].tolist(),
            'intercept': float(model.intercept_[0]), 'iterations': model.n_iter_.tolist(),
            'sklearn_parameters': model.get_params()}


def predict_record(record, x):
    import numpy as np
    from scipy.special import expit
    if record['classes'] != [0, 1] or record['feature_order'] != ['s2', 's4'] or record['metadata_features']:
        raise RunConflict('unexpected saved fusion model contract')
    return expit(x @ np.array(record['coef']) + record['intercept'])


def reconstruct(x_train, labels, x_test, reference):
    """TEST labels cannot enter this interface. No TEST metrics or tuning."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof = np.empty(len(labels), dtype=float)
    folds = []
    fit_calls = 0
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold, (training, heldout) in enumerate(StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels), 1):
            model = LogisticRegression(**LR_PARAMS)
            model.fit(x_train[training], labels[training])
            fit_calls += 1
            oof[heldout] = model.predict_proba(x_train[heldout])[:, 1]
            folds.append({'fold': fold, 'heldout_indices': heldout.tolist(),
                          'model': model_record(model)})
        threshold = select_threshold(oof, labels)
        selected = threshold['selected']
        matches = (round(selected['threshold'], 4) == reference['train_threshold']
                   and round(selected['f1'], 4) == reference['oof_train_f1'])
        result = {'status': 'historical_TRAIN_reference_mismatch', 'fusion_fit_calls': fit_calls,
                  'historical_TRAIN_reference': reference, 'rounded_TRAIN_reference_matches': matches,
                  'threshold': threshold, 'folds': folds, 'TRAIN_meta_oof_scores': oof.tolist(),
                  'full_model': None, 'TEST_scores': []}
        if not matches:
            return result  # Preserve mismatch; do not tune or emit final TEST predictions.
        model = LogisticRegression(**LR_PARAMS)
        model.fit(x_train, labels)
        record = model_record(model)
        probs = predict_record(record, x_test)
        if not np.allclose(probs, model.predict_proba(x_test)[:, 1], rtol=0, atol=1e-14):
            raise RunConflict('JSON coefficient replay disagrees with sklearn predictions')
        result.update(status='reconstructed_legacy_baseline_with_provenance_limits',
                      fusion_fit_calls=fit_calls + 1, full_model=record, TEST_scores=probs.tolist())
        return result


def recover_once(path, identity, compute):
    """Caller owns the run lock. A completed record, including mismatch, is immutable."""
    if path.exists():
        bundle = json.loads(path.read_text(encoding='utf-8'))
        report = bundle['report']
        if bundle.get('report_sha256') != content_hash(report) or report['identity'] != identity:
            raise RunConflict('existing fusion reconstruction differs or is corrupt; preserve it')
        return bundle, False
    report = compute()
    report['identity'] = identity
    bundle = {'report_sha256': content_hash(report), 'report': report}
    save_once(path, bundle)
    return bundle, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--train-parquet', type=Path, required=True)
    args = parser.parse_args()
    revision = code_revision()
    versions = {p: version(p) for p in VERSIONS}
    if versions != VERSIONS:
        raise RunConflict('install requirements-baseline-audit.txt in the CPU environment')
    frozen = load_freeze()
    validate_fit(json.loads((run_directory(FIT_RUN_ID) / 'fit.json').read_text(encoding='utf-8')), frozen)
    provenance = checked_bundle(run_directory(audit.RUN_ID) / 'report.json', 'report', 'report_sha256', PROVENANCE_SHA256)
    if provenance['code_revision'] != PROVENANCE_REVISION:
        raise RunConflict('unexpected provenance audit revision')
    manifest = checked_bundle(run_directory(TEST_RUN_ID) / 'manifest.json', 'manifest', 'manifest_sha256', audit.TEST_MANIFEST_SHA256)
    train, _ = _records(read_train(args.train_parquet))
    expected = {'train': train, 'test': manifest['offline_rows']}
    caches = {}
    for name in CACHE_NAMES:
        digest, split, field, probability = audit.FILES[name]
        path = args.baseline_dir / name
        if file_sha256(path) != digest or provenance['cache_checks'][name]['sha256'] != digest:
            raise RunConflict('legacy fusion cache changed: ' + name)
        rows, checks = audit.align(json.loads(path.read_text(encoding='utf-8')), expected[split], field, probability)
        if checks['missing_scores']:
            raise RunConflict('complete fusion rows required; no silent row dropping')
        caches[name] = rows
    audit.oof_check(caches[CACHE_NAMES[1]])
    reference_path = audit.REPO_ROOT / 'results/fusion/fusion_decomposition_review_results.json'
    source_path = audit.REPO_ROOT / 'fusion/fusion_decomposition_review.py'
    if file_sha256(reference_path) != REFERENCE_SHA256 or file_sha256(source_path) != SOURCE_SHA256:
        raise RunConflict('historical fusion reference/source changed')
    # Access only TRAIN fields. Historical TEST metrics are never selection targets.
    historical = json.loads(reference_path.read_text(encoding='utf-8'))['results']['S2_S4']['meta_oof_threshold']
    reference = {k: historical[k] for k in ('train_threshold', 'oof_train_f1')}
    x_train = features(caches[CACHE_NAMES[0]], caches[CACHE_NAMES[1]])
    x_test = features(caches[CACHE_NAMES[2]], caches[CACHE_NAMES[3]])
    import numpy as np
    labels = np.array([r['label'] for r in train], dtype=int)
    identity = {'version': 'metadata-free-fusion-recovery-v1', 'code_revision': revision,
                'versions': versions, 'provenance_report_sha256': PROVENANCE_SHA256,
                'TEST_manifest_sha256': audit.TEST_MANIFEST_SHA256, 'TRAIN_parquet_sha256': TRAIN_SHA256,
                'judge_freeze_sha256': content_hash(frozen), 'source_sha256': SOURCE_SHA256,
                'historical_reference_sha256': REFERENCE_SHA256,
                'cache_sha256': {name: audit.FILES[name][0] for name in CACHE_NAMES}}

    def compute():
        result = reconstruct(x_train, labels, x_test, reference)
        result['TEST_predictions'] = [
            {'sample_id': row['sample_id'], 'test_index': row['test_index'],
             'target_manifest_input_sha256': row['input_sha256'], 'unsupported_score': score}
            for row, score in zip(manifest['offline_rows'], result.pop('TEST_scores'))]
        return {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'results': result,
                'generation_calls': 0, 'http_requests': 0, 'judge_fitting_calls': 0,
                'TEST_labels_used_for_alignment_only': True, 'TEST_metrics_computed': False,
                'source_files_rewritten': False, 'metadata_features': False, 'comparison_ready': False,
                'historical_serialized_model_recovered': False,
                'limitations': ['CPU reconstruction from original cached features; not historical weights.',
                    'TRAIN reference match is rounded to four decimals, not proof of identical historical predictions.',
                    'Target manifest input hashes describe alignment targets, not historical inference inputs.',
                    'Legacy cache input identity, checkpoint linkage and evidence visibility remain unverified.',
                    'Original full TRAIN baseline fitting includes judge development rows; training exposure differs.',
                    'Meta OOF is not fully nested end-to-end cross-validation of the S4 training pipeline.',
                    'Stored S4 fold tags do not independently establish checkpoint training exclusion.']}

    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        bundle, created = recover_once(directory / 'report.json', identity, compute)
    result = bundle['report']['results']
    print('Post-thesis metadata-free S2+S4 CPU reconstruction')
    print('New reconstruction:', created)
    print('New fusion fits:', result['fusion_fit_calls'] if created else 0)
    print('Status:', result['status'])
    print('Rounded historical TRAIN reference matches:', result['rounded_TRAIN_reference_matches'])
    print('TRAIN meta-OOF threshold:', json.dumps(result['threshold']['selected']))
    print('Full model:', json.dumps(result['full_model'], indent=2))
    print('Saved TEST predictions:', len(result['TEST_predictions']))
    print('Report SHA256:', bundle['report_sha256'])
    print('Code revision:', revision)
    print('Private report:', directory / 'report.json')
    print('No TEST metrics, transformer calls or judge refitting. Legacy provenance limitations remain.')
    return 0 if result['rounded_TRAIN_reference_matches'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
