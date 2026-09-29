"""Offline, post-evaluation disagreement counts and a blinded manual review pack."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import platform

from . import evaluate_test as rag
from . import evaluate_halubench as halu
from . import evaluation_math as metrics
from .check_frozen_fit import load_freeze, validate_fit, FIT_RUN_ID
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'frozen-test-disagreement-review-v1'
EVALUATIONS = {
    'ragtruth': (rag.RUN_ID, '7bdf3e9c5de63e585a07044f7e33b14f960490c01b1e682f08494da9148d3aac'),
    'halubench': (halu.RUN_ID, '6919ae8e3d137fc40803a351a15216628d8df2b89fd3c9b2d5908a1fd2e571d0'),
}
REVIEW_BASELINES = ('MiniCheck_7B', 'S2_S4_metadata_free')
STATES = ('both_correct', 'judge_only_correct', 'baseline_only_correct', 'both_wrong')
CATEGORIES = ('unsupported_addition', 'contradiction', 'partial_support', 'absence_claim',
              'numeric_claim', 'paraphrase', 'long_context_grounding', 'source_specific', 'other')
POLICY = {
    'version': 'frozen-test-disagreement-review-v1', 'study_stage': 'post_thesis',
    'timing': 'defined after aggregate TEST results; descriptive error analysis, not confirmatory validation',
    'seed': 'post-thesis-review-20260929-v1', 'per_stratum': 2,
    'strata': 'benchmark x task/source x baseline x correctness state',
    'baselines': list(REVIEW_BASELINES), 'states': list(STATES),
    'selection': 'ascending SHA256(seed, benchmark, slice, baseline, state, sample_id); one row per component within each stratum',
    'shortfall': 'take all available distinct components; no replacement or quota redistribution',
    'deduplication': 'union by benchmark and sample_id across strata; retain all selection memberships',
    'review_order': 'ascending opaque review_id; no sorting by score or correctness',
    'masking': 'packet excludes labels, scores, system names, source/task names and selection strata; text can reveal domain',
    'limit': 128, 'model_calls': 0, 'fitting_calls': 0, 'threshold_changes': False,
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_evaluation(bundle, expected_hash, inputs):
    report = bundle['report']
    if (bundle.get('report_sha256') != expected_hash or content_hash(report) != expected_hash
            or report['identity']['aligned_evaluation_inputs_sha256'] != content_hash(inputs)):
        raise RunConflict('completed evaluation or aligned inputs changed')
    return report


def load_population(benchmark):
    """Reuse immutable evaluation loaders; replay responses without recomputing metrics."""
    module = rag if benchmark == 'ragtruth' else halu
    frozen = load_freeze()
    fit = read_json(run_directory(FIT_RUN_ID) / 'fit.json')
    validate_fit(fit, frozen)
    bundle = read_json(run_directory(module.INPUT_RUN_ID) / 'manifest.json')
    module.validate_inputs(bundle, frozen)
    manifest = bundle['manifest']; offline = manifest['offline_rows']
    targets = [{k: row[k] for k in ('sample_id', 'test_index', 'input_sha256', 'component_sha256')}
               for row in offline]
    if benchmark == 'ragtruth':
        reports = {name: rag.read_report(spec) for name, spec in rag.SOURCES.items()}
        descriptor = rag.validate_comparison(read_json(run_directory(rag.COMPARISON_RUN_ID) / 'manifest.json'), targets, reports)
        judge, _ = rag.verify_judge(run_directory(rag.JUDGE_RUN_ID), targets)
    else:
        reports, descriptor = halu.verify_baselines(targets)
        judge, _ = halu.verify_judge(targets)
    inputs = module.join_evaluation_rows(offline, judge, reports, descriptor)
    run_id, digest = EVALUATIONS[benchmark]
    report = verify_evaluation(read_json(run_directory(run_id) / 'report.json'), digest, inputs)
    return manifest, inputs, report


def correctness_state(judge, baseline, label):
    if judge == label:
        return 'both_correct' if baseline == label else 'judge_only_correct'
    return 'baseline_only_correct' if baseline == label else 'both_wrong'


def decision_rows(benchmark, manifest, inputs, evaluation):
    """Use the same frozen decision functions and check all saved confusion counts."""
    n = len(inputs['sample_ids'])
    if (len(set(inputs['sample_ids'])) != n or len(manifest['model_inputs']) != n
            or len(manifest['offline_rows']) != n or len(inputs['labels']) != n
            or len(inputs['groups']) != n):
        raise RunConflict('duplicate, missing or unaligned population')
    arrays = {name: metrics.system_arrays(name, inputs['scores'][name]) for name in metrics.MAIN_SYSTEMS}
    if any(len(a['valid']) != n or not a['valid'].all() for a in arrays.values()):
        raise RunConflict('complete finite scores required for descriptive review')
    axis = 'task' if benchmark == 'ragtruth' else 'source'
    if len(inputs['axes'][axis]) != n:
        raise RunConflict('slice population differs')
    rows = []
    for i, (text, offline) in enumerate(zip(manifest['model_inputs'], manifest['offline_rows'])):
        sample_id = inputs['sample_ids'][i]; label = inputs['labels'][i]
        if (text['sample_id'] != sample_id or offline['sample_id'] != sample_id
                or offline['test_index'] != i or offline['component_sha256'] != inputs['groups'][i]
                or offline['label'] != label or type(label) is not int or label not in (0, 1)
                or content_hash({k: text[k] for k in ('answer', 'context')}) != offline['input_sha256']):
            raise RunConflict('review text/label/group alignment differs')
        rows.append({'benchmark': benchmark, 'sample_id': sample_id, 'component': inputs['groups'][i],
            'slice': inputs['axes'][axis][i], 'label': label, 'input_sha256': offline['input_sha256'],
            'answer': text['answer'], 'context': text['context'],
            'scores': {name: inputs['scores'][name][i] for name in metrics.MAIN_SYSTEMS},
            'predictions': {name: int(arrays[name]['predictions'][i]) for name in metrics.MAIN_SYSTEMS}})
    for name in metrics.MAIN_SYSTEMS:
        confusion = dict.fromkeys(('tp', 'fp', 'fn', 'tn'), 0)
        for row in rows:
            label, predicted = row['label'], row['predictions'][name]
            confusion[('t' if predicted == label else 'f') + ('p' if predicted else 'n')] += 1
        if confusion != evaluation['results']['paired']['shared_points'][name]['confusion']:
            raise RunConflict('review decisions differ from completed evaluation: ' + name)
    return rows


def count_rows(rows):
    patterns = Counter(''.join(str(row['predictions'][s]) for s in metrics.MAIN_SYSTEMS) for row in rows)
    pairs = {}
    for baseline in metrics.MAIN_SYSTEMS[1:]:
        counts = Counter(correctness_state(r['predictions']['judge'], r['predictions'][baseline], r['label']) for r in rows)
        pairs[baseline] = {state: counts[state] for state in STATES}
    all_correct = sum(all(v == r['label'] for v in r['predictions'].values()) for r in rows)
    all_wrong = sum(all(v != r['label'] for v in r['predictions'].values()) for r in rows)
    return {'n': len(rows), 'pair_correctness': pairs,
        'all_four_correct': all_correct, 'all_four_wrong': all_wrong,
        'mixed_decisions': len(rows) - all_correct - all_wrong,
        'pattern_system_order': list(metrics.MAIN_SYSTEMS), 'decision_patterns': dict(sorted(patterns.items()))}


def build_review(rows):
    """Deterministic, stratified diagnostic selection; no representativeness claim."""
    keys = [(r['benchmark'], r['sample_id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise RunConflict('duplicate review input')
    summaries, selected, strata = {}, {}, []
    for benchmark in sorted({r['benchmark'] for r in rows}):
        population = [r for r in rows if r['benchmark'] == benchmark]
        slices = sorted({r['slice'] for r in population})
        summaries[benchmark] = {'overall': count_rows(population), 'slices': {}}
        for source in slices:
            subset = [r for r in population if r['slice'] == source]
            summaries[benchmark]['slices'][source] = count_rows(subset)
            for baseline in REVIEW_BASELINES:
                for state in STATES:
                    candidates = [r for r in subset if correctness_state(r['predictions']['judge'], r['predictions'][baseline], r['label']) == state]
                    rank = lambda r: content_hash([POLICY['seed'], benchmark, source, baseline, state, r['sample_id']])
                    candidates.sort(key=lambda r: (rank(r), r['sample_id']))
                    used = set(); chosen = []
                    membership = {'benchmark': benchmark, 'slice': source, 'baseline': baseline, 'state': state}
                    for row in candidates:
                        if row['component'] in used: continue
                        used.add(row['component']); chosen.append(row['sample_id'])
                        key = (benchmark, row['sample_id'])
                        if key not in selected: selected[key] = {'row': row, 'memberships': []}
                        selected[key]['memberships'].append(membership)
                        if len(chosen) == POLICY['per_stratum']: break
                    strata.append({**membership, 'eligible_rows': len(candidates),
                        'eligible_components': len({r['component'] for r in candidates}),
                        'selected_ids': chosen, 'shortfall': POLICY['per_stratum'] - len(chosen)})
    packet, key_rows, annotations = [], [], []
    for key, item in selected.items():
        row = item['row']; review_id = content_hash([POLICY['seed'], 'review', *key])
        packet.append({'review_id': review_id, 'answer': row['answer'], 'context': row['context']})
        key_rows.append({'review_id': review_id, **{k: row[k] for k in ('benchmark', 'sample_id', 'slice', 'component', 'input_sha256', 'label', 'scores', 'predictions')},
                         'selection_memberships': item['memberships']})
        annotations.append({'review_id': review_id, 'reviewer': None, 'reviewed_at': None,
            'independent_verdict': None, 'categories': [], 'answer_quote': None, 'context_quote': None,
            'rationale': None, 'ambiguity_notes': None})
    for items in (packet, key_rows, annotations): items.sort(key=lambda r: r['review_id'])
    if len(packet) > POLICY['limit']: raise RunConflict('review quota exceeded')
    return {'summaries': summaries, 'selection_strata': strata, 'selected_unique_examples': len(packet)}, packet, key_rows, annotations


def write_outputs(directory, payloads):
    """Validate every existing file before writing; never edit reviewers' copies."""
    for name, value in payloads.items():
        path = directory / name
        if path.exists() and read_json(path) != value:
            raise RunConflict('existing review artifact differs; preserve it: ' + name)
    return {name: save_once(directory / name, value) for name, value in payloads.items()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    contract = rag.load_contract(); halu.verify_implementation()
    versions = {'python': platform.python_version(), **{p: version(p) for p in contract['packages']}}
    if versions['python'].split('.')[:2] != ['3', '12'] or any(versions[p] != v for p, v in contract['packages'].items()):
        raise RunConflict('use the existing pinned CPU fitting environment')
    rows = []
    for benchmark in EVALUATIONS:
        print('Verifying frozen inputs, predictions and evaluation:', benchmark, flush=True)
        manifest, inputs, report = load_population(benchmark)
        rows.extend(decision_rows(benchmark, manifest, inputs, report))
    result, packet, key, annotations = build_review(rows)
    report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'policy': POLICY,
        'policy_sha256': content_hash(POLICY), 'input_rows_sha256': content_hash(rows),
        'evaluations': {k: v[1] for k, v in EVALUATIONS.items()}, 'results': result,
        'packet_sha256': content_hash(packet), 'key_sha256': content_hash(key),
        'annotation_template_sha256': content_hash(annotations), 'model_calls': 0, 'fitting_calls': 0,
        'threshold_changes': False, 'manual_review_completed': False,
        'limitations': ['Correctness means agreement with benchmark labels, not an adjudicated semantic truth.',
            'Post-evaluation stratified review is diagnostic; selected-case frequencies are not population error rates.',
            'One component per stratum; components may recur across strata. Selected examples are deduplicated.',
            'Binary systems cannot all disagree pairwise; report unanimous correct/wrong and mixed decisions.',
            'Manual review must inspect full context; quote membership alone does not establish support.',
            'Existing native evidence windows, S2 empty-pair fallback and historical provenance limits remain.']}
    bundle = {'report_sha256': content_hash(report), 'report': report}
    payloads = {'report.json': bundle, 'review_packet.json': packet, 'review_key.json': key,
                'annotation_template.json': annotations}
    directory = run_directory(RUN_ID)
    with exclusive_run(directory):
        # Code/version receipt is separate: later HEAD changes never rewrite the original pack.
        receipt = directory / 'creation.json'
        if not receipt.exists(): save_once(receipt, {'code_revision': code_revision(), 'versions': versions})
        changed = write_outputs(directory, payloads)
    print('Post-thesis frozen disagreement review; new files:', sum(changed.values()))
    for benchmark, entry in result['summaries'].items():
        print(benchmark, json.dumps(entry['overall'], sort_keys=True))
    print('Unique review examples:', result['selected_unique_examples'], '/ maximum', POLICY['limit'])
    print('Report SHA256:', bundle['report_sha256']); print('Private review directory:', directory)
    print('No inference, fitting or threshold changes. Manual review pending; read DISAGREEMENT_REVIEW.md.')
    return 0


if __name__ == '__main__': raise SystemExit(main())
