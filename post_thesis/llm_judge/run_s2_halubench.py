"""Fresh S2 zero-shot features for the canonical HaluBench 8k; no fitting."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from . import run_s2_test as prior
from .audit_halubench_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .prepare_halubench import RUN_ID as INPUT_RUN_ID
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .runner import run_directory
from .s2_inference import S2Backend, reference
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 's2-halubench-test-fresh-v1'
S4_RUN_ID = 's4-halubench-test-fresh-v1'
S4_REPORT_SHA256 = '2d68d0657d983db0df84b146264652eea00034734e82d09a42126c56e4ba0f2c'
S4_REVISION = '80ac88e02aac163db33bc885ab441fae14eedfa6'
REFERENCES_SHA256 = '92ce62a176b1963a87e49c260f081c6fe573e915debd8070a6594f018134959d'
IMPLEMENTATION_FILES = {
    'run_s2_test.py': 'ad3325e5bffa474f95a94dfdbc00dd5d6478d48858c5db68d57e934b0427d2d8',
    's2_inference.py': 'a948f994e9c9583a89032d53400b5b4461af41bcb8125abd985c303dcb67fcfa',
    'check_s4_checkpoint.py': 'fa121cb77b024ea5ad3af9e9f8b3731cbf899eb3292204443a425fd7dd2edd82',
}
PLAN = {**prior.PLAN, 'version': RUN_ID, 'dataset': 'canonical_HaluBench_8k_TEST',
        'examples': 8000, 'sentence_pairs': 81107, 'maximum_forward_batches': 8327,
        'empty_pair_examples': 1886, 'prepared_references_sha256': REFERENCES_SHA256,
        'zero_shot_transfer': True, 'adaptation_training': False,
        'implementation_files': IMPLEMENTATION_FILES}
validate_result = prior.validate_result


def verify_implementation():
    for name, expected in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise RunConflict('registered S2 implementation changed: ' + name)
    return dict(IMPLEMENTATION_FILES)


def validate_s4(bundle, examples):
    report = bundle['report']
    if (bundle.get('report_sha256') != S4_REPORT_SHA256 or content_hash(report) != S4_REPORT_SHA256
            or report['run_id'] != S4_RUN_ID or report['identity']['code_revision'] != S4_REVISION
            or report['identity']['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or report['identity']['judge_freeze_sha256'] != FREEZE_SHA256
            or report['examples_total'] != len(examples) or report['valid_scores'] != len(examples)
            or report['terminal_failures'] != 0 or report['pending'] != 0
            or report['HaluBench_read'] is not True or report['adaptation_training'] is not False):
        raise RunConflict('expected completed unchanged fresh HaluBench S4 report')
    if len(report['predictions']) != len(examples):
        raise RunConflict('incomplete S4 prediction alignment')
    for i, (row, ex) in enumerate(zip(report['predictions'], examples)):
        if (row['sample_id'] != ex.sample_id or type(row['test_index']) is not int
                or row['test_index'] != i or row['status'] != 'ok'
                or row['input_sha256'] != content_hash(asdict(ex.item))):
            raise RunConflict('S4 prediction input alignment changed')
    return S4_REPORT_SHA256


def summarize(refs, records, identity):
    report = prior.summarize(refs, records, identity)['report']
    report.update({'run_id': RUN_ID, 'HaluBench_read': True, 'RAGTruth_inputs_read': False,
                   'adaptation_training': False,
                   'empty_pair_examples': sum(r['pairs'] == 0 for r in refs),
                   'scope': 'fresh_post_thesis_S2_HaluBench_zero_shot_features'})
    return {'report_sha256': content_hash(report), 'report': report}


def execute(examples, *, directory, identity, backend_factory, max_new_examples=8000):
    if type(max_new_examples) is not int or not 0 <= max_new_examples <= PLAN['examples']:
        raise ValueError('max-new-examples must be an integer from 0 to 8000')
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
    parser.add_argument('--download-model', action='store_true', help='Allow only the pinned S2 model files if missing')
    parser.add_argument('--max-new-examples', type=int, default=8000)
    args = parser.parse_args()
    if not 0 <= args.max_new_examples <= PLAN['examples']:
        parser.error('--max-new-examples must be from 0 to 8000')
    revision = code_revision()
    implementation = verify_implementation()
    versions = {'python': platform.python_version(), **{p: version(p) for p in prior.VERSIONS}}
    if platform.python_version_tuple()[:2] != ('3', '12') or any(versions[p] != v for p,v in prior.VERSIONS.items()):
        raise RunConflict('use the unchanged .venv-judge-serving environment')
    bundle = json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, load_freeze())
    s4_hash = validate_s4(json.loads((run_directory(S4_RUN_ID)/'summary.json').read_text(encoding='utf-8')), examples)
    refs = [reference(ex, i) for i,ex in enumerate(examples)]
    counts = (len(refs),sum(r['pairs'] for r in refs),sum(r['forward_batches'] for r in refs),sum(r['pairs']==0 for r in refs))
    if counts != (8000,81107,8327,1886) or content_hash(refs) != REFERENCES_SHA256:
        raise RunConflict('canonical HaluBench S2 sentence workload changed')
    identity = {'code_revision': revision, 'plan': PLAN, 'versions': versions,
                'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'judge_freeze_sha256': FREEZE_SHA256,
                'prepared_references_sha256': REFERENCES_SHA256, 'S4_report_sha256': s4_hash,
                'implementation_files': implementation}
    print('S2 fixed workload: 8000 examples / 81107 sentence pairs / 8327 forward batches', flush=True)
    print('Empty-pair examples: 1886; original zero raw-feature policy preserved.', flush=True)
    result, new = execute(examples, directory=run_directory(RUN_ID), identity=identity,
                          backend_factory=lambda: S2Backend(args.model_cache, download=args.download_model),
                          max_new_examples=args.max_new_examples)
    report = result['report']
    print('Post-thesis fresh S2 HaluBench zero-shot TEST inference')
    print('Valid scores:', report['valid_scores'], '/', report['examples_total'])
    print('New attempted examples:', new)
    print('Terminal failures / pending:', report['terminal_failures'], '/', report['pending'])
    print('Completed sentence pairs / forward batches:', report['completed_pairs'], '/', report['completed_forward_batches'])
    print('Empty-pair examples (fixed preprocessing):', report['empty_pair_examples'])
    print('Truncated pairs:', json.dumps(report['truncated_pairs']))
    print('Timing (shared GPU; loading excluded):', json.dumps(report['timing']))
    print('Report SHA256:', result['report_sha256'])
    print('Private summary:', run_directory(RUN_ID)/'summary.json')
    print('Raw S2 features only. No TEST label metrics, target-domain training, normalization/fusion fitting, judge calls or legacy-file changes.')
    return 0 if report['terminal_failures'] == 0 else 1


if __name__ == '__main__': raise SystemExit(main())
