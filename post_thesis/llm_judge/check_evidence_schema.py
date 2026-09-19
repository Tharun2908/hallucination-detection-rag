"""CPU-only check of installed XGrammar against a versioned evidence schema.

No model/tokenizer downloads, HTTP requests, vLLM server or generation. This
checks native schema compilation/matching, not end-to-end serving compatibility.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from .evidence_schema_v2 import EVIDENCE_SCHEMA_V2_JSON
from .prompts import canonical_json, content_hash
from .runner import run_directory
from .schema_v2_checks import checks_sha256, compare_acceptance
from .serve import code_revision
from .storage import atomic_json


def package_versions():
    result = {'python': platform.python_version()}
    for name in ('xgrammar', 'vllm', 'torch'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def schema_spec(schema_version):
    if schema_version == 'v2':
        return EVIDENCE_SCHEMA_V2_JSON, checks_sha256()
    if schema_version == 'v3':
        from .evidence_schema_v3 import EVIDENCE_SCHEMA_V3_JSON
        from .schema_v3_checks import checks_sha256 as v3_hash
        return EVIDENCE_SCHEMA_V3_JSON, v3_hash()
    raise ValueError('unknown evidence schema version')


def native_check(schema_version='v2'):
    schema, fixture_hash = schema_spec(schema_version)
    import xgrammar as xgr
    # Fixed tiny byte vocabulary, not the model tokenizer. accept_string tests
    # complete serialized JSON strings; no token logits or GPU tensors are used.
    tokenizer = xgr.TokenizerInfo([bytes([i]) for i in range(128)] + [b'<eos>'],
                                  vocab_type=xgr.VocabType.RAW, stop_token_ids=[128])
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1, cache_enabled=False)
    compiled = compiler.compile_json_schema(schema)

    def accepts_text(text):
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        accepted = matcher.accept_string(text)
        return accepted and matcher.is_terminated()

    if schema_version == 'v2':
        results = compare_acceptance(lambda payload: accepts_text(canonical_json(payload)))
    else:
        from .schema_v3_checks import compare_acceptance as compare_v3
        results = compare_v3(accepts_text)
    return {'status': 'passed' if all(r['matches_expected'] for r in results) else 'failed',
            'cases': results, 'cases_total': len(results),
            'checks_sha256': fixture_hash, 'schema_version': schema_version,
            'schema_sha256': content_hash(json.loads(schema)),
            'serialized_schema_sha256': hashlib.sha256(schema.encode('utf-8')).hexdigest(),
            'method': 'native XGrammar compilation and complete-string matching with fixed ASCII byte vocabulary'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema-version', choices=('v2', 'v3'), default='v2')
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output is not None:
        try:
            result = native_check(args.schema_version)
        except Exception as error:
            result = {'status': 'blocked', 'error_type': type(error).__name__}
            # Local environment/API details stay in the private worker log.
            print(f'{type(error).__name__}: {error}', file=sys.stderr)
        atomic_json(args.worker_output, result)
        return 0 if result['status'] == 'passed' else 1

    schema, fixture_hash = schema_spec(args.schema_version)
    revision = code_revision()
    directory = run_directory('evidence-schema-' + args.schema_version + '-compatibility-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    worker_output = directory / 'native_result.json'
    report = {'study_stage': 'post_thesis', 'kind': 'CPU_schema_compatibility',
              'code_revision': revision, 'versions': package_versions(),
              'schema_version': args.schema_version,
              'schema_sha256': content_hash(json.loads(schema)),
              'serialized_schema_sha256': hashlib.sha256(schema.encode('utf-8')).hexdigest(),
              'checks_sha256': fixture_hash, 'generation_calls': 0,
              'model_tokenizer_checked': False, 'vllm_server_checked': False,
              'status': 'started', 'started_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(directory / 'report.json', report)
    with (directory / 'worker.log').open('w', encoding='utf-8') as log:
        try:
            completed = subprocess.run([sys.executable, '-m', 'post_thesis.llm_judge.check_evidence_schema',
                                        '--schema-version', args.schema_version, '--worker-output', str(worker_output)],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False)
            result = json.loads(worker_output.read_text(encoding='utf-8')) if worker_output.exists() else {'status': 'blocked'}
            report.update(status=result['status'], native_result=result, worker_returncode=completed.returncode)
            if completed.returncode != 0 and report['status'] == 'passed':
                report['status'] = 'blocked'
        except subprocess.TimeoutExpired:
            report.update(status='blocked', error_type='timeout', timeout_seconds=60)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(directory / 'report.json', report)
    print('Versions:', json.dumps(report['versions'], sort_keys=True))
    print('Schema version:', args.schema_version)
    print('Schema SHA256 (canonical):', report['schema_sha256'])
    print('Serialized schema SHA256:', report['serialized_schema_sha256'])
    print('Fixture SHA256:', report['checks_sha256'])
    print('Native grammar check:', report['status'])
    if 'native_result' in report and 'cases' in report['native_result']:
        cases = report['native_result']['cases']
        print('Matching acceptance checks:', sum(c['matches_expected'] for c in cases), '/', len(cases))
        for case in cases:
            if not case['matches_expected']:
                print('Mismatch:', json.dumps(case, sort_keys=True))
    print('Private compatibility record:', directory / 'report.json')
    print('Generation calls: 0. Model tokenizer and live vLLM compatibility remain unverified.')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
