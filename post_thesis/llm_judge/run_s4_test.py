"""Bounded, resumable fresh S4 scores for the 2,700 canonical TEST inputs."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .check_s4_checkpoint import RUN_ID as COMPAT_RUN_ID, verify_checkpoint
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID
from .prompts import content_hash
from .runner import run_directory
from .s4_inference import S4Backend, TOKENIZER_SHA256
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 's4-ragtruth-test-fresh-v1'
COMPAT_SHA256 = '504dccf973f8c9995a02128332dc21a83d97c80c594cd1cbaee8396b11653077'
COMPAT_REVISION = '4581a26d854c3008a3c2e1f3c6ea8899ea934316'
VERSIONS = {'huggingface-hub': '1.32.0', 'safetensors': '0.8.0', 'tokenizers': '0.23.2',
            'torch': '2.13.0+cu130', 'transformers': '5.17.0'}
PLAN = {'study_stage': 'post_thesis', 'version': 's4-ragtruth-test-fresh-v1',
        'examples': 2700, 'batch_size': 16, 'maximum_batches': 169, 'attempts_per_batch': 1,
        'device': 'H200:0', 'dtype': 'float32', 'tf32': False, 'attention_implementation': 'eager',
        'max_length': 512, 'truncation': 'longest_first', 'padding': 'max_length',
        'pair_order': ['answer', 'context'], 'forward_fields': ['input_ids', 'attention_mask'],
        'unsupported_class': 1, 'cpu_threads': 2, 'seed': 42,
        'max_invocation_seconds': 1800, 'deadline_check': 'before each batch; no preemption of an active batch',
        'frozen_historical_threshold': {'comparator': '>=', 'value': 0.55},
        'threshold_application': 'deferred to registered evaluation',
        'benchmark_metrics': False, 'judge_calls': 0, 'training': False}


def validate_compatibility(bundle):
    report = bundle['report']
    if (bundle.get('report_sha256') != COMPAT_SHA256 or content_hash(report) != COMPAT_SHA256
            or report['identity']['code_revision'] != COMPAT_REVISION
            or report['check']['status'] != 'passed'
            or report['check']['tokenizer_backend_sha256'] != TOKENIZER_SHA256):
        raise RunConflict('expected the completed S4 CPU compatibility report')
    return report


def batch_plan(examples):
    if not examples or len({ex.sample_id for ex in examples}) != len(examples):
        raise RunConflict('expected unique nonempty S4 inputs')
    batches = []
    for start in range(0, len(examples), PLAN['batch_size']):
        group = examples[start:start + PLAN['batch_size']]
        refs = [{'sample_id': ex.sample_id, 'test_index': start + i,
                 'input_sha256': content_hash(asdict(ex.item))} for i, ex in enumerate(group)]
        batches.append({'key': content_hash(refs), 'references': refs, 'examples': group})
    return batches


def validate_result(result, refs):
    if result.get('status') == 'error':
        if result.get('rows') != [] or not isinstance(result.get('error'), str):
            raise RunConflict('invalid S4 failure record')
        return
    if result.get('status') != 'ok' or len(result.get('rows', [])) != len(refs):
        raise RunConflict('invalid S4 batch result')
    for name in ('batch_seconds', 'cuda_forward_seconds'):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RunConflict('invalid S4 timing')
    for row, ref in zip(result['rows'], refs):
        if row['sample_id'] != ref['sample_id'] or row['input_sha256'] != ref['input_sha256']:
            raise RunConflict('S4 result does not match its exact input')
        score, logits = row['unsupported_score'], row['logits']
        if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or len(logits) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
            raise RunConflict('invalid S4 score or logits')
        margin = logits[1] - logits[0]
        exp_term = math.exp(-abs(margin))
        expected = 1 / (1 + exp_term) if margin >= 0 else exp_term / (1 + exp_term)
        if abs(score - expected) > 1e-6:
            raise RunConflict('S4 score disagrees with saved logits')
        digest = row['forward_tensors_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RunConflict('invalid S4 effective-input hash')
        if type(row['nonpadding_tokens']) is not int or not 1 <= row['nonpadding_tokens'] <= 512:
            raise RunConflict('invalid S4 input length')
        coverage = row['coverage']
        if set(coverage) != {'answer', 'context'}:
            raise RunConflict('missing S4 evidence visibility')
        for counts in coverage.values():
            if (set(counts) != {'full_tokens', 'kept_tokens'}
                    or any(type(v) is not int for v in counts.values())
                    or not 0 <= counts['kept_tokens'] <= counts['full_tokens']):
                raise RunConflict('invalid S4 token coverage')
        if sum(v['kept_tokens'] for v in coverage.values()) >= row['nonpadding_tokens']:
            raise RunConflict('S4 input must include special tokens')


def summarize(batches, records, identity):
    planned = {b['key']: b for b in batches}
    seen, predictions, runtimes = set(), [], []
    timing = {'known_batch_seconds': 0., 'known_cuda_forward_seconds': 0., 'unknown_timing_batches': 0}
    for record in records:
        key = record['request_key']
        if key not in planned or key in seen or record['ordinal'] != 1 or record['state'] not in ('finished', 'interrupted'):
            raise RunConflict('unrecognized, repeated or unfinished S4 batch')
        seen.add(key)
        if record['state'] == 'finished':
            validate_result(record['result'], planned[key]['references'])
    by_key = {r['request_key']: r for r in records}
    for batch in batches:
        stored = by_key.get(batch['key'])
        result = stored['result'] if stored and stored['state'] == 'finished' else None
        status = result['status'] if result else ('interrupted' if stored else 'pending')
        if status == 'ok':
            timing['known_batch_seconds'] += result['batch_seconds']
            timing['known_cuda_forward_seconds'] += result['cuda_forward_seconds']
            if result['runtime'] not in runtimes:
                runtimes.append(result['runtime'])
            predictions.extend({**ref, 'status': 'ok', **row} for ref, row in zip(batch['references'], result['rows']))
        else:
            if stored: timing['unknown_timing_batches'] += 1
            predictions.extend({**ref, 'status': status, 'unsupported_score': None} for ref in batch['references'])
    valid = [p for p in predictions if p['status'] == 'ok']
    pending = sum(p['status'] == 'pending' for p in predictions)
    report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'identity': identity,
              'examples_total': len(predictions), 'valid_scores': len(valid), 'pending': pending,
              'terminal_failures': len(predictions) - len(valid) - pending, 'attempted_batches': len(records),
              'truncated_examples': {side: sum(p['coverage'][side]['kept_tokens'] < p['coverage'][side]['full_tokens']
                                               for p in valid) for side in ('answer', 'context')},
              'timing': timing, 'runtimes': runtimes, 'predictions': predictions,
              'legacy_files_rewritten': False, 'historical_training_provenance_verified': False,
              'legacy_cache_to_checkpoint_link_verified': False, 'comparison_ready': False,
              'benchmark_metrics_computed': False, 'judge_calls': 0, 'fitting_calls': 0,
              'HaluBench_read': False, 'scope': 'fresh_post_thesis_S4_TEST_inference'}
    return {'report_sha256': content_hash(report), 'report': report}


def execute(examples, *, directory, identity, backend_factory, max_new_batches=169):
    if type(max_new_batches) is not int or not 0 <= max_new_batches <= PLAN['maximum_batches']:
        raise ValueError('max-new-batches must be an integer from 0 to 169')
    batches = batch_plan(examples)
    if len(examples) > PLAN['examples'] or len(batches) > PLAN['maximum_batches']:
        raise RunConflict('S4 run exceeds the registered count budget')
    manifest = {'identity': identity, 'batches': [{k: b[k] for k in ('key', 'references')} for b in batches]}
    directory = Path(directory)
    with exclusive_run(directory):
        journal = Journal(directory / 'journal.sqlite3', manifest)
        try:
            journal.recover()
            records = journal.records()
            summary = summarize(batches, records, identity)
            seen = {r['request_key'] for r in records}
            new, backend, started = 0, None, time.perf_counter()
            for batch in batches:
                if batch['key'] in seen: continue
                if new >= max_new_batches or time.perf_counter() - started >= PLAN['max_invocation_seconds']: break
                if backend is None: backend = backend_factory()
                if time.perf_counter() - started >= PLAN['max_invocation_seconds']: break
                attempt = journal.start(batch['key'], 1)
                new += 1
                try:
                    result = backend.score(batch['examples'])
                    validate_result(result, batch['references'])
                except Exception as exc:
                    result = {'status': 'error', 'rows': [], 'error': type(exc).__name__ + ': ' + str(exc)}
                journal.finish(attempt, result)
                summary = summarize(batches, journal.records(), identity)
                atomic_json(directory / 'summary.json', summary)
                print(f"S4: {summary['report']['valid_scores']}/{len(examples)} valid; new batches {new}", flush=True)
                if result['status'] != 'ok':
                    print('S4 batch error:', result['error'], flush=True)
                    break
            summary = summarize(batches, journal.records(), identity)
            path = directory / 'summary.json'
            if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != summary:
                atomic_json(path, summary)
            return summary, new
        finally:
            # Process death or Ctrl-C leaves the durable reservation pending; the
            # next invocation marks it interrupted and never repeats that batch.
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/signal4_model'))
    parser.add_argument('--max-new-batches', type=int, default=169)
    args = parser.parse_args()
    revision = code_revision()
    versions = {'python': platform.python_version(), **{p: version(p) for p in VERSIONS}}
    if platform.python_version_tuple()[:2] != ('3', '12') or any(versions[p] != v for p, v in VERSIONS.items()):
        raise RunConflict('use the existing, unchanged .venv-judge-serving environment')
    compatibility = validate_compatibility(json.loads((run_directory(COMPAT_RUN_ID) / 'report.json').read_text(encoding='utf-8')))
    files = verify_checkpoint(args.checkpoint)
    if files != compatibility['identity']['checkpoint_files']:
        raise RunConflict('checkpoint changed after CPU compatibility check')
    frozen = load_freeze()
    bundle = json.loads((run_directory(TEST_RUN_ID) / 'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, frozen)
    identity = {'code_revision': revision, 'plan': PLAN, 'versions': versions,
                'checkpoint_files': files, 'compatibility_report_sha256': COMPAT_SHA256,
                'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'judge_freeze_sha256': FREEZE_SHA256}
    result, new = execute(examples, directory=run_directory(RUN_ID), identity=identity,
                          backend_factory=lambda: S4Backend(args.checkpoint.resolve(strict=True)),
                          max_new_batches=args.max_new_batches)
    report = result['report']
    print('Post-thesis fresh S4 TEST inference')
    print('Valid scores:', report['valid_scores'], '/', report['examples_total'])
    print('New attempted batches:', new)
    print('Terminal failures / pending:', report['terminal_failures'], '/', report['pending'])
    print('Truncated examples:', json.dumps(report['truncated_examples']))
    print('Timing (shared GPU; model loading excluded):', json.dumps(report['timing']))
    print('Report SHA256:', result['report_sha256'])
    print('Private summary:', run_directory(RUN_ID) / 'summary.json')
    print('Fresh scores only. No TEST metrics, judge calls, fitting, HaluBench reads or legacy-file changes.')
    return 0 if report['terminal_failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
