"""Post-thesis evaluation mathematics. No file loading, inference, fitting or tuning."""
import math

from .check_frozen_fit import load_freeze
from .fitting_math import reliability
from .prompts import content_hash
from .storage import RunConflict

MAIN_SYSTEMS = ('judge', 'S4', 'MiniCheck_7B', 'S2_S4_metadata_free')
SEED = 20260920
REPLICATES = 2000
MIN_VALID_DRAWS = 1900
ECE_EDGES = [i / 10 for i in range(11)]
PROVENANCE_FIELDS = ('input_content_verified', 'checkpoint_provenance_verified',
                     'threshold_provenance_verified', 'evidence_visibility_documented')


def _binary(values):
    import numpy as np
    values = list(values)
    if any(type(v) not in (int, bool) or v not in (0, 1) for v in values):
        raise RunConflict('labels must be explicit binary integers or booleans')
    return np.array(values, dtype=int)


def system_arrays(name, scores):
    """Input is judge raw margins, MC support probabilities, other unsupported scores.

    None is the only missing-value marker. Do not coerce invalid outputs to scores.
    Returned arrays are internal: all reported metrics use the explicit valid mask.
    """
    import numpy as np
    from scipy.special import expit
    if name not in MAIN_SYSTEMS:
        raise RunConflict('unregistered system; metadata-aware fusion is not a substitute')
    scores = list(scores)
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in scores):
        raise RunConflict('scores must be finite numeric values or explicit None')
    if name != 'judge' and any(v is not None and not 0 <= v <= 1 for v in scores):
        raise RunConflict('baseline probability outside [0,1]')
    raw = np.array([np.nan if v is None else v for v in scores], dtype=np.float64)
    valid = np.isfinite(raw)
    decision = np.full(len(raw), np.nan)
    if name == 'judge':
        frozen = load_freeze()
        a, b = (frozen['calibration']['parameters'][k] for k in ('a', 'b'))
        threshold = frozen['operating_threshold']['margin']
        rank = raw.copy()
        probabilities = {'raw': expit(raw), 'calibrated': expit(a * raw + b)}
        decision[valid] = raw[valid] >= threshold
        rule = {'space': 'raw_unsupported_log_odds', 'comparator': '>=', 'threshold': threshold}
    elif name == 'MiniCheck_7B':
        # Negation preserves support-score ranking without introducing subtraction ties.
        rank = -raw
        probabilities = {'raw': 1 - raw}
        threshold = 0.20000000000000004
        decision[valid] = raw[valid] < threshold
        rule = {'space': 'support_probability', 'comparator': '<', 'threshold': threshold}
    else:
        rank = raw.copy()
        probabilities = {'raw': raw.copy()}
        threshold = .55 if name == 'S4' else .45
        decision[valid] = raw[valid] >= threshold
        rule = {'space': 'unsupported_probability', 'comparator': '>=', 'threshold': threshold}
    return {'name': name, 'raw_input': raw, 'rank': rank, 'probabilities': probabilities,
            'predictions': decision, 'valid': valid,
            'rule': {**rule, 'threshold_hex': float(rule['threshold']).hex()}}


def _point(y, system, indices):
    """All indices must refer to valid scores; repetition implements bootstrap weights."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = y[indices]
    predictions = system['predictions'][indices]
    n = len(y)
    tp = int(((y == 1) & (predictions == 1)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    both = set(y.tolist()) == {0, 1}
    rank_reason = None if both else ('empty_valid_subset' if not n else 'one_class_subset')
    metrics = {
        'auroc': float(roc_auc_score(y, system['rank'][indices])) if both else None,
        'average_precision': float(average_precision_score(y, system['rank'][indices])) if both else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'accuracy': (tp + tn) / n if n else None,
    }
    calibration = {}
    for view, probabilities in system['probabilities'].items():
        values = reliability(probabilities[indices], y, ECE_EDGES) if n else None
        calibration[view] = values
        for key in ('brier', 'ece'):
            metrics[key + '_' + view] = None if values is None else values[key]
    return {'n': n, 'positives': int(y.sum()), 'negatives': n - int(y.sum()),
            'prevalence': float(y.mean()) if n else None,
            'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'metrics': metrics, 'undefined_ranking_reason': rank_reason,
            'reliability': calibration}


def single_metrics(labels, name, scores):
    """Descriptive valid-subset metrics with the original denominator preserved."""
    import numpy as np
    y = _binary(labels)
    system = system_arrays(name, scores)
    if len(y) != len(system['valid']):
        raise RunConflict('label/score length mismatch')
    indices = np.flatnonzero(system['valid'])
    report = _point(y, system, indices)
    report.update(attempted_n=len(y), missing_n=len(y) - len(indices),
                  coverage=len(indices) / len(y) if len(y) else None, rule=system['rule'])
    if name == 'judge':
        reference = dict(system)
        reference['predictions'] = np.where(system['valid'], system['raw_input'] >= 0, np.nan)
        point = _point(y, reference, indices)
        report['fixed_zero_reference_secondary'] = {'confusion': point['confusion'],
            'metrics': {key: point['metrics'][key] for key in ('f1', 'precision', 'recall', 'accuracy')},
            'comparator': '>=', 'raw_margin_threshold': 0.0}
    return report


def descriptive_slices(labels, name, scores, axes):
    """All supplied axis levels are reported, including all-missing/one-class slices."""
    y, scores = list(labels), list(scores)
    if len(y) != len(scores):
        raise RunConflict('unaligned descriptive inputs')
    output = {}
    for axis, values in axes.items():
        values = list(values)
        if len(values) != len(y) or any(not isinstance(v, str) or not v for v in values):
            raise RunConflict('complete named slice metadata required')
        output[axis] = {}
        for level in sorted(set(values)):
            indices = [i for i, value in enumerate(values) if value == level]
            output[axis][level] = single_metrics([y[i] for i in indices], name, [scores[i] for i in indices])
    return output


def comparison_gate(provenance):
    """Checks caller-validated evidence status; does not create provenance evidence."""
    missing = {}
    for name in MAIN_SYSTEMS:
        source = provenance.get(name, {})
        fields = [key for key in PROVENANCE_FIELDS if source.get(key) is not True]
        if source.get('comparison_ready') is not True:
            fields.append('comparison_ready')
        if fields:
            missing[name] = fields
    return {'status': 'blocked' if missing else 'ready', 'missing_requirements': missing}


def cluster_draws(groups, eligible):
    """K full groups per draw; retain multiplicity and original row order in groups."""
    import numpy as np
    group_names = sorted({groups[int(i)] for i in eligible})
    members = [np.array([i for i in eligible if groups[int(i)] == name], dtype=int) for name in group_names]
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, len(members), size=(REPLICATES, len(members))) if members else np.empty((REPLICATES, 0), dtype=int)
    return group_names, members, draws


def interval(values):
    import numpy as np
    valid = [v for v in values if v is not None]
    enough = len(valid) >= MIN_VALID_DRAWS
    return {'lower': float(np.percentile(valid, 2.5, method='linear')) if enough else None,
            'upper': float(np.percentile(valid, 97.5, method='linear')) if enough else None,
            'valid_draws': len(valid), 'invalid_draws': len(values) - len(valid),
            'reason': None if enough else 'fewer_than_1900_valid_draws', 'confidence_level': .95}


def _delta_key(metric):
    # Both judge reliability views compare to the same raw baseline probabilities.
    return metric.replace('_calibrated', '_raw')


def paired_metrics(labels, score_sets, sample_ids, groups, provenance):
    """Registered four-system shared intersection. Stops before metrics if unready.

    Callers must validate artifact identities/content evidence before setting ready.
    This numerical core is not a benchmark loader or a provenance attestation.
    """
    import numpy as np
    if set(score_sets) != set(MAIN_SYSTEMS):
        raise RunConflict('all four registered systems required; no pairwise subset substitution')
    labels = list(labels)
    score_sets = {name: list(values) for name, values in score_sets.items()}
    y = _binary(labels)
    sample_ids, groups = list(sample_ids), list(groups)
    if (len(sample_ids) != len(y) or len(groups) != len(y) or len(set(sample_ids)) != len(y)
            or any(not isinstance(v, str) or not v for v in sample_ids + groups)):
        raise RunConflict('complete unique IDs and nonempty frozen group IDs required')
    systems = {name: system_arrays(name, score_sets[name]) for name in MAIN_SYSTEMS}
    if any(len(s['valid']) != len(y) for s in systems.values()):
        raise RunConflict('unaligned system scores')
    gate = comparison_gate(provenance)
    if gate['status'] != 'ready':
        return {'study_stage': 'post_thesis', 'comparison': gate, 'metrics_computed': False}
    shared = np.flatnonzero(np.logical_and.reduce([s['valid'] for s in systems.values()]))
    group_names, members, draws = cluster_draws(groups, shared)
    points = {name: _point(y, s, shared) for name, s in systems.items()}
    metrics = {name: list(point['metrics']) for name, point in points.items()}
    distributions = {name: {metric: [] for metric in metrics[name]} for name in MAIN_SYSTEMS}
    differences = {name: {metric: [] for metric in metrics['judge']} for name in MAIN_SYSTEMS if name != 'judge'}
    for draw in draws:
        indices = np.concatenate([members[int(i)] for i in draw]) if len(draw) else np.array([], dtype=int)
        values = {name: _point(y, s, indices)['metrics'] for name, s in systems.items()}
        for name in MAIN_SYSTEMS:
            for metric, value in values[name].items():
                # No observations is not evidence for a zero performance interval.
                distributions[name][metric].append(value if len(indices) else None)
        for name in differences:
            for metric in differences[name]:
                a, b = values['judge'][metric], values[name][_delta_key(metric)]
                differences[name][metric].append(a - b if len(indices) and a is not None and b is not None else None)
    intervals = {name: {m: interval(v) for m, v in scores.items()} for name, scores in distributions.items()}
    delta_intervals = {}
    for name, scores in differences.items():
        delta_intervals[name] = {}
        for metric, values in scores.items():
            a, b = points['judge']['metrics'][metric], points[name]['metrics'][_delta_key(metric)]
            delta_intervals[name][metric] = {
                'judge_metric': metric, 'baseline_metric': _delta_key(metric), 'direction': 'judge_minus_baseline',
                'larger_is_better': not metric.startswith(('brier', 'ece')),
                'point_difference': a - b if len(shared) and a is not None and b is not None else None,
                **interval(values)}
    return {'study_stage': 'post_thesis', 'comparison': gate, 'metrics_computed': True,
            'attempted_n': len(y), 'shared_n': len(shared), 'shared_positive_n': int(y[shared].sum()),
            'shared_negative_n': len(shared) - int(y[shared].sum()),
            'shared_coverage': len(shared) / len(y) if len(y) else None,
            'shared_sample_ids': [sample_ids[int(i)] for i in shared],
            'individual': {name: single_metrics(labels, name, score_sets[name]) for name in MAIN_SYSTEMS},
            'shared_points': points, 'intervals': intervals, 'paired_differences': delta_intervals,
            'bootstrap': {'replicates': REPLICATES, 'minimum_valid_draws': MIN_VALID_DRAWS,
                          'seed': SEED, 'rng': 'Generator(PCG64)', 'eligible_groups': group_names,
                          'draws_sha256': content_hash(draws.tolist()),
                          'eligible_membership_sha256': content_hash({name: [sample_ids[int(i)] for i in indices]
                                                                      for name, indices in zip(group_names, members)}),
                          'percentile_method': 'linear', 'unit': 'whole_groups',
                          'refitting': False, 'nominal_unadjusted_intervals': True}}
