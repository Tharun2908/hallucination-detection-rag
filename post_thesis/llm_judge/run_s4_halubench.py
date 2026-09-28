"""Bounded fresh S4 zero-shot inference on the unchanged canonical HaluBench 8k."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from . import run_s4_test as prior
from .audit_halubench_tokens import TEST_MANIFEST_SHA256, RUN_ID as TOKEN_RUN_ID, validate_inputs
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .check_s4_checkpoint import RUN_ID as COMPAT_RUN_ID, verify_checkpoint
from .prepare_halubench import RUN_ID as INPUT_RUN_ID
from .prepare_pilot import file_sha256
from .prompts import content_hash
from .runner import run_directory
from .s4_inference import S4Backend
from .serve import code_revision
from .storage import Journal, RunConflict, atomic_json, exclusive_run

RUN_ID = 's4-halubench-test-fresh-v1'
TOKEN_AUDIT_SHA256 = '7173c30e3a197aabb61c8ffd8c0e145cc556affa9aad20d7dcd43bc9cb24782a'
TOKEN_AUDIT_REVISION = 'cbc9cca82ebeb46daa42e82722fe8c5cdff141ab'
BATCHES_SHA256 = 'eca3ad65ea7d1bbe05d588ca13d410afa39d86451e28d7974d289fb8f6d1db03'
IMPLEMENTATION_FILES = {
    'run_s4_test.py': '59dd85194ca369d3dc9b5efccbb60d1da4a9618c12b5323bd1968f64bc6613a1',
    's4_inference.py': 'fc4f61ac1ce947c2e313a1e89093644747a0b9d9779b6a16000cd7e6937b9af7',
    'check_s4_checkpoint.py': 'fa121cb77b024ea5ad3af9e9f8b3731cbf899eb3292204443a425fd7dd2edd82',
}
# Reuse the already exercised runtime, not a new checkpoint or target-domain fit.
PLAN = {**prior.PLAN, 'version': RUN_ID, 'dataset': 'canonical_HaluBench_8k_TEST',
        'examples': 8000, 'maximum_batches': 500, 'zero_shot_transfer': True,
        'batch_references_sha256': BATCHES_SHA256,
        'adaptation_training': False, 'implementation_files': IMPLEMENTATION_FILES}
batch_plan = prior.batch_plan
validate_result = prior.validate_result


def verify_implementation():
    for name, expected in IMPLEMENTATION_FILES.items():
        if file_sha256(Path(__file__).with_name(name)) != expected:
            raise RunConflict('registered S4 implementation changed: ' + name)
    return dict(IMPLEMENTATION_FILES)


def validate_token_audit(bundle):
    audit = bundle['audit']
    if (bundle.get('audit_sha256') != TOKEN_AUDIT_SHA256
            or content_hash(audit) != TOKEN_AUDIT_SHA256
            or audit['status'] != 'completed'
            or audit['identity']['code_revision'] != TOKEN_AUDIT_REVISION
            or audit['identity']['TEST_manifest_sha256'] != TEST_MANIFEST_SHA256
            or audit['identity']['judge_freeze_sha256'] != FREEZE_SHA256
            or audit['summary']['examples_counted'] != 8000
            or audit['summary']['total_input_tokens'] != 7794485
            or audit['summary']['all_inputs_fit'] is not True):
        raise RunConflict('expected completed unchanged HaluBench judge token audit')
    return TOKEN_AUDIT_SHA256


def summarize(batches, records, identity):
    # Pure validation/aggregation is shared with the exercised RAGTruth runner.
    # Retag only this new report; never mutate a historical artifact.
    report = prior.summarize(batches, records, identity)['report']
    report.update({'run_id': RUN_ID, 'HaluBench_read': True, 'RAGTruth_inputs_read': False,
                   'adaptation_training': False, 'scope': 'fresh_post_thesis_S4_HaluBench_zero_shot_TEST_inference'})
    return {'report_sha256': content_hash(report), 'report': report}


def execute(examples, *, directory, identity, backend_factory, max_new_batches=500):
    if type(max_new_batches) is not int or not 0 <= max_new_batches <= PLAN['maximum_batches']:
        raise ValueError('max-new-batches must be an integer from 0 to 500')
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
    parser.add_argument('--max-new-batches', type=int, default=500)
    args = parser.parse_args()
    if not 0 <= args.max_new_batches <= PLAN['maximum_batches']:
        parser.error('--max-new-batches must be from 0 to 500')
    revision = code_revision()
    implementation = verify_implementation()
    versions = {'python': platform.python_version(), **{p: version(p) for p in prior.VERSIONS}}
    if platform.python_version_tuple()[:2] != ('3', '12') or any(versions[p] != v for p,v in prior.VERSIONS.items()):
        raise RunConflict('use the existing unchanged .venv-judge-serving environment')
    compatibility = prior.validate_compatibility(json.loads((run_directory(COMPAT_RUN_ID)/'report.json').read_text(encoding='utf-8')))
    files = verify_checkpoint(args.checkpoint)
    if files != compatibility['identity']['checkpoint_files']:
        raise RunConflict('checkpoint changed after CPU compatibility check')
    frozen = load_freeze()
    bundle = json.loads((run_directory(INPUT_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    examples = validate_inputs(bundle, frozen)
    planned = [{k: b[k] for k in ('key','references')} for b in batch_plan(examples)]
    if content_hash(planned) != BATCHES_SHA256:
        raise RunConflict('canonical HaluBench batch references differ')
    audit_hash = validate_token_audit(json.loads((run_directory(TOKEN_RUN_ID)/'audit.json').read_text(encoding='utf-8')))
    identity = {'code_revision': revision, 'plan': PLAN, 'versions': versions,
                'checkpoint_files': files, 'compatibility_report_sha256': prior.COMPAT_SHA256,
                'implementation_files': implementation, 'token_audit_sha256': audit_hash,
                'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'judge_freeze_sha256': FREEZE_SHA256}
    result, new = execute(examples, directory=run_directory(RUN_ID), identity=identity,
                          backend_factory=lambda: S4Backend(args.checkpoint.resolve(strict=True)),
                          max_new_batches=args.max_new_batches)
    report = result['report']
    print('Post-thesis fresh S4 HaluBench zero-shot TEST inference')
    print('Valid scores:', report['valid_scores'], '/', report['examples_total'])
    print('New attempted batches:', new)
    print('Terminal failures / pending:', report['terminal_failures'], '/', report['pending'])
    print('Truncated examples:', json.dumps(report['truncated_examples']))
    print('Timing (shared GPU; model loading excluded):', json.dumps(report['timing']))
    print('Report SHA256:', result['report_sha256'])
    print('Private summary:', run_directory(RUN_ID)/'summary.json')
    print('Fresh canonical 8k scores only. No TEST metrics, target-domain training, judge calls, fitting or legacy-file changes.')
    return 0 if report['terminal_failures'] == 0 else 1


if __name__ == '__main__': raise SystemExit(main())
