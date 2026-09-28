"""Fresh post-thesis S2 TEST features; fixed sentence pairs, resumable per example."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import load_freeze
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID
from .prompts import content_hash
from .runner import run_directory
from .run_s4_test import VERSIONS
from .s2_inference import MODEL, REVISION, FILES, S2Backend, aggregate, reference
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 's2-ragtruth-test-fresh-v1'
PLAN = {'study_stage': 'post_thesis', 'version': RUN_ID, 'examples': 2700,
        'sentence_pairs': 362918, 'maximum_forward_batches': 20576,
        'batch_size': 32, 'batch_boundary': 'within each answer sentence, original context order',
        'attempts_per_example': 1, 'max_invocation_seconds': 7200,
        'deadline_check': 'before each example; active example is not preempted',
        'model': MODEL, 'revision': REVISION,
        'model_files': {k: {'bytes': v[0], 'sha256': v[1]} for k,v in FILES.items()},
        'device': 'H200:0', 'dtype': 'float32', 'tf32': False, 'attention_implementation': 'eager',
        'max_length': 512, 'truncation': 'longest_first', 'padding': 'longest_in_batch',
        'forward_fields': ['input_ids', 'attention_mask', 'token_type_ids'],
        'activation': 'Identity; raw logits', 'fusion_feature': 'min of per-answer context maxima',
        'empty_sentence_policy': 'original zero raw min/mean; empty maxima list',
        'TRAIN_normalization_fitting': False, 'TEST_metrics': False, 'judge_calls': 0}


def validate_result(result, ref):
    if result.get('status') == 'error':
        if result.get('pair_batches') != [] or not isinstance(result.get('error'), str):
            raise RunConflict('invalid S2 error record')
        return
    if result.get('status') != 'ok': raise RunConflict('unknown S2 result status')
    computed = aggregate(result['pair_batches'], ref['n_answer_sentences'], ref['n_context_sentences'])
    if any(result[key] != value for key, value in computed.items()):
        raise RunConflict('S2 aggregate differs from saved raw pair logits')
    for key in ('example_seconds', 'cuda_forward_seconds'):
        v = result[key]
        if type(v) not in (float, int) or not math.isfinite(v) or v < 0: raise RunConflict('invalid S2 timing')
    for batch in result['pair_batches']:
        digest = batch['forward_tensors_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RunConflict('invalid S2 tensor hash')
        if len(batch['coverage']) != len(batch['logits']): raise RunConflict('S2 pair coverage length differs')
        for pair in batch['coverage']:
            if set(pair) != {'answer', 'context'}: raise RunConflict('missing S2 visibility record')
            for counts in pair.values():
                if len(counts) != 2 or any(type(v) is not int for v in counts) or not 0 <= counts[1] <= counts[0]:
                    raise RunConflict('invalid S2 full/kept token counts')
            if sum(c[1] for c in pair.values()) > 509: raise RunConflict('S2 pair exceeds token budget')


def summarize(refs, records, identity):
    keys = [content_hash(ref) for ref in refs]
    expected = dict(zip(keys, refs))
    by_key, predictions, runtimes = {}, [], []
    timing = {'known_example_seconds': 0., 'known_cuda_forward_seconds': 0., 'unknown_timing_examples': 0}
    truncated = {'answer_pairs': 0, 'context_pairs': 0}
    completed_batches = completed_pairs = 0
    for record in records:
        key = record['request_key']
        if key not in expected or key in by_key or record['ordinal'] != 1 or record['state'] not in ('finished', 'interrupted'):
            raise RunConflict('unexpected or repeated S2 journal attempt')
        by_key[key] = record
        if record['state'] == 'finished': validate_result(record['result'], expected[key])
    for key, ref in zip(keys, refs):
        stored = by_key.get(key)
        result = stored['result'] if stored and stored['state'] == 'finished' else None
        status = result['status'] if result else ('interrupted' if stored else 'pending')
        if status == 'ok':
            timing['known_example_seconds'] += result['example_seconds']
            timing['known_cuda_forward_seconds'] += result['cuda_forward_seconds']
            completed_batches += ref['forward_batches']; completed_pairs += ref['pairs']
            for batch in result['pair_batches']:
                for pair in batch['coverage']:
                    for side in ('answer', 'context'):
                        truncated[side + '_pairs'] += pair[side][1] < pair[side][0]
            if result['runtime'] not in runtimes: runtimes.append(result['runtime'])
            predictions.append({**ref, 'status': 'ok', **{k: result[k] for k in
                                ('raw_min_relevance', 'raw_mean_relevance', 'per_answer_best_logits')}})
        else:
            if stored: timing['unknown_timing_examples'] += 1
            predictions.append({**ref, 'status': status, 'raw_min_relevance': None,
                                'raw_mean_relevance': None, 'per_answer_best_logits': None})
    valid = sum(p['status'] == 'ok' for p in predictions)
    pending = sum(p['status'] == 'pending' for p in predictions)
    report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'identity': identity,
              'examples_total': len(refs), 'valid_scores': valid, 'terminal_failures': len(refs)-valid-pending,
              'pending': pending, 'completed_pairs': completed_pairs, 'completed_forward_batches': completed_batches,
              'truncated_pairs': truncated, 'timing': timing, 'runtimes': runtimes, 'predictions': predictions,
              'pair_provenance_location': 'journal.sqlite3: committed result for each example',
              'legacy_cache_provenance_verified': False, 'legacy_files_rewritten': False,
              'comparison_ready': False, 'TEST_label_metrics_computed': False,
              'judge_calls': 0, 'fitting_calls': 0, 'HaluBench_read': False}
    return {'report_sha256': content_hash(report), 'report': report}


def execute(examples, *, directory, identity, backend_factory, max_new_examples=2700):
    if type(max_new_examples) is not int or not 0 <= max_new_examples <= PLAN['examples']:
        raise ValueError('max-new-examples must be an integer from 0 to 2700')
    refs = [reference(ex, i) for i, ex in enumerate(examples)]
    if (not refs or len({r['sample_id'] for r in refs}) != len(refs) or len(refs) > PLAN['examples']
            or sum(r['pairs'] for r in refs) > PLAN['sentence_pairs']
            or sum(r['forward_batches'] for r in refs) > PLAN['maximum_forward_batches']):
        raise RunConflict('S2 inputs exceed the fixed count budget or have invalid IDs')
    directory = Path(directory)
    with exclusive_run(directory):
        journal = Journal(directory / 'journal.sqlite3', {'identity': identity, 'references': refs})
        try:
            journal.recover()
            summary = summarize(refs, journal.records(), identity)
            seen = {r['request_key'] for r in journal.records()}
            new, backend, started = 0, None, time.perf_counter()
            for ex, ref in zip(examples, refs):
                key = content_hash(ref)
                if key in seen: continue
                if new >= max_new_examples or time.perf_counter()-started >= PLAN['max_invocation_seconds']: break
                if backend is None: backend = backend_factory()
                if time.perf_counter()-started >= PLAN['max_invocation_seconds']: break
                attempt = journal.start(key, 1); new += 1
                try:
                    result = backend.score(ex)
                    validate_result(result, ref)
                except Exception as exc:
                    result = {'status': 'error', 'pair_batches': [], 'error': type(exc).__name__ + ': ' + str(exc)}
                journal.finish(attempt, result)
                if new % 25 == 0 or result['status'] != 'ok':
                    summary = summarize(refs, journal.records(), identity)
                    atomic_json(directory / 'summary.json', summary)
                    print(f"S2: {summary['report']['valid_scores']}/{len(examples)} valid; new examples {new}", flush=True)
                if result['status'] != 'ok':
                    print('S2 error:', result['error'], flush=True)
                    break
            summary = summarize(refs, journal.records(), identity)
            path = directory / 'summary.json'
            if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != summary:
                atomic_json(path, summary)
            return summary, new
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path, default=Path('/root/llm-judge-hf-cache'))
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--max-new-examples', type=int, default=2700)
    args = parser.parse_args()
    revision = code_revision()
    versions = {'python': platform.python_version(), **{p: version(p) for p in VERSIONS}}
    if platform.python_version_tuple()[:2] != ('3', '12') or any(versions[p] != v for p,v in VERSIONS.items()):
        raise RunConflict('use the unchanged .venv-judge-serving environment')
    bundle = json.loads((run_directory(TEST_RUN_ID) / 'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, load_freeze())
    refs = [reference(ex, i) for i,ex in enumerate(examples)]
    if (len(refs), sum(r['pairs'] for r in refs), sum(r['forward_batches'] for r in refs)) != (2700, 362918, 20576):
        raise RunConflict('canonical S2 sentence workload changed')
    identity = {'code_revision': revision, 'plan': PLAN, 'versions': versions,
                'TEST_manifest_sha256': TEST_MANIFEST_SHA256,
                'prepared_references_sha256': content_hash(refs)}
    print('S2 fixed workload: 2700 examples / 362918 sentence pairs / 20576 forward batches', flush=True)
    result, new = execute(examples, directory=run_directory(RUN_ID), identity=identity,
                          backend_factory=lambda: S2Backend(args.model_cache, download=args.download_model),
                          max_new_examples=args.max_new_examples)
    report = result['report']
    print('Post-thesis fresh S2 TEST inference')
    print('Valid scores:', report['valid_scores'], '/', report['examples_total'])
    print('New attempted examples:', new)
    print('Terminal failures / pending:', report['terminal_failures'], '/', report['pending'])
    print('Completed sentence pairs / forward batches:', report['completed_pairs'], '/', report['completed_forward_batches'])
    print('Truncated pairs:', json.dumps(report['truncated_pairs']))
    print('Timing (shared GPU; loading excluded):', json.dumps(report['timing']))
    print('Report SHA256:', result['report_sha256'])
    print('Private summary:', run_directory(RUN_ID) / 'summary.json')
    print('Raw S2 features only. No TEST label metrics, normalization fitting, fusion fitting, judge calls or legacy-file changes.')
    return 0 if report['terminal_failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
