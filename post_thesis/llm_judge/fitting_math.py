"""Numerical implementation of the preregistered development fitting rules."""

import math

from .storage import RunConflict


def arrays(margins, labels, *, require_both=True):
    import numpy as np
    x, y = np.asarray(margins, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if (x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isin(y, [0, 1]).all() or (require_both and set(y) != {0, 1})):
        raise RunConflict("finite aligned scores and both binary classes are required")
    return x, y


def objective(parameters, margins, labels, penalty):
    import numpy as np
    from scipy.special import expit
    a, b = parameters
    z = a * margins + b
    residual = expit(z) - labels
    value = np.mean(np.logaddexp(0.0, (1 - 2 * labels) * z)) + penalty / 2 * ((a - 1)**2 + b**2)
    gradient = np.array([np.mean(residual * margins) + penalty * (a - 1),
                         np.mean(residual) + penalty * b], dtype=np.float64)
    return float(value), gradient


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def fit_calibration(margins, labels, config):
    import numpy as np
    from scipy.optimize import minimize
    x, y = arrays(margins, labels)
    initial = np.array([config['initial_parameters'][k] for k in ('a', 'b')], dtype=np.float64)
    bounds = [config['bounds'][k] for k in ('a', 'b')]
    acceptance = config['acceptance']
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            initial_value, _ = objective(initial, x, y, config['lambda'])
            result = minimize(objective, initial, args=(x, y, config['lambda']), jac=True,
                              method=config['optimizer']['method'], bounds=bounds,
                              options=config['optimizer']['options'])
            parameters = np.asarray(result.x, dtype=np.float64)
            value, gradient = objective(parameters, x, y, config['lambda'])
        finite = bool(np.isfinite(parameters).all() and np.isfinite(value) and np.isfinite(gradient).all())
        distances = [min(float(v) - lo, hi - float(v)) for v, (lo, hi) in zip(parameters, bounds)]
        reasons = []
        if not result.success: reasons.append('optimizer_unsuccessful')
        if not finite: reasons.append('nonfinite_fit')
        if any(d <= acceptance['min_absolute_distance_from_each_bound_exclusive'] for d in distances):
            reasons.append('boundary_reached')
        if not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > acceptance['max_abs_gradient']:
            reasons.append('gradient_tolerance_failed')
        if not math.isfinite(value) or value > initial_value + acceptance['max_objective_increase_from_initial']:
            reasons.append('objective_not_accepted')
        return {
            'status': 'failed' if reasons else 'accepted',
            'parameters': None if reasons else {k: float(v) for k, v in zip(('a', 'b'), parameters)},
            'failure_reasons': reasons,
            'diagnostics': {'candidate_parameters_not_for_deployment': {k: _number(v) for k, v in zip(('a', 'b'), parameters)},
                            'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
                            'optimizer_message': str(result.message), 'iterations': int(result.nit),
                            'function_evaluations': int(result.nfev), 'initial_objective': _number(initial_value),
                            'final_objective': _number(value), 'gradient': [_number(v) for v in gradient],
                            'minimum_bound_distances': [_number(v) for v in distances]},
        }
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as error:
        return {'status': 'failed', 'parameters': None, 'failure_reasons': ['numerical_fit_error'],
                'diagnostics': {'error_type': type(error).__name__, 'message': str(error)}}


def confusion(margins, labels, threshold):
    x, y = arrays(margins, labels, require_both=False)
    predictions = x >= threshold if threshold is not None else x != x  # all-negative sentinel
    tp = int(((y == 1) & predictions).sum()); fp = int(((y == 0) & predictions).sum())
    fn = int(((y == 1) & ~predictions).sum()); tn = int(((y == 0) & ~predictions).sum())
    numerator, denominator = 2 * tp, 2 * tp + fp + fn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'f1_numerator': numerator, 'f1_denominator': denominator,
            'f1': numerator / denominator if denominator else 0.0,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'accuracy': (tp + tn) / len(y)}


def select_threshold(margins, labels):
    x, y = arrays(margins, labels)
    candidates, best, best_fraction = [], None, None
    for threshold in sorted(set(float(v) for v in x)) + [None]:
        row = {'kind': 'above_max' if threshold is None else 'finite_margin',
               'margin': threshold, 'margin_hex': None if threshold is None else threshold.hex(),
               **confusion(x, y, threshold)}
        candidates.append(row)
        numerator, denominator = row['f1_numerator'], row['f1_denominator']
        fraction = (numerator, denominator) if denominator else (0, 1)
        # Ascending traversal + >= resolves exact F1 ties toward the largest t.
        if best_fraction is None or fraction[0] * best_fraction[1] >= best_fraction[0] * fraction[1]:
            best, best_fraction = row, fraction
    return {'status': 'selected', 'space': 'raw_unsupported_log_odds', 'comparator': '>=',
            'selected': best, 'candidate_table': candidates,
            'fixed_zero_reference': confusion(x, y, 0.0),
            'interpretation': 'operating_development_selection_performance_not_heldout'}


def reliability(probabilities, labels, edges):
    import numpy as np
    p, y = arrays(probabilities, labels, require_both=False)
    if np.any(p < 0) or np.any(p > 1):
        raise RunConflict('probability outside [0,1]')
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) != 11 or not np.array_equal(edges, np.array([i / 10 for i in range(11)])):
        raise RunConflict('expected preregistered ten equal-width bins')
    indices = np.minimum(np.searchsorted(edges, p, side='right') - 1, 9)
    bins, ece = [], 0.0
    for index in range(10):
        mask = indices == index; n = int(mask.sum())
        mean_p = float(p[mask].mean()) if n else None
        mean_y = float(y[mask].mean()) if n else None
        contribution = n / len(y) * abs(mean_p - mean_y) if n else 0.0
        ece += contribution
        bins.append({'left': float(edges[index]), 'right': float(edges[index + 1]),
                     'right_inclusive': index == 9, 'count': n, 'mean_probability': mean_p,
                     'positive_fraction': mean_y, 'ece_contribution': contribution})
    return {'brier': float(np.mean((p - y)**2)), 'ece': float(ece), 'bins': bins}


def fit_arms(arms, protocol):
    from scipy.special import expit
    calibration = fit_calibration(arms['calibration']['margins'], arms['calibration']['labels'], protocol['calibration'])
    threshold = select_threshold(arms['operating_threshold']['margins'], arms['operating_threshold']['labels'])
    diagnostics = {}
    for arm in ('calibration', 'operating_threshold'):
        x, y = arrays(arms[arm]['margins'], arms[arm]['labels'])
        raw = reliability(expit(x), y, protocol['calibration_diagnostics']['ece_edges'])
        calibrated = None
        if calibration['status'] == 'accepted':
            pars = calibration['parameters']
            calibrated = reliability(expit(pars['a'] * x + pars['b']), y, protocol['calibration_diagnostics']['ece_edges'])
        diagnostics[arm] = {'raw_uncalibrated': raw, 'calibrated': calibrated,
                            'interpretation': 'in_sample_fit_diagnostics' if arm == 'calibration'
                                              else 'development_diagnostics_not_independent_pipeline_validation'}
    threshold['descriptive_calibrated_threshold'] = None
    if calibration['status'] == 'accepted' and threshold['selected']['margin'] is not None:
        pars = calibration['parameters']
        threshold['descriptive_calibrated_threshold'] = float(expit(pars['a'] * threshold['selected']['margin'] + pars['b']))
    return {'calibration': calibration, 'operating_threshold': threshold, 'reliability': diagnostics,
            'deployment_rule': 'compare_raw_margin_to_selected_raw_threshold_even_when_calibrated'}
