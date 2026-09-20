"""Fit the preregistered post-thesis calibrator and raw-margin threshold offline."""
import argparse
from importlib.metadata import version
import json
import platform
import time

from .audit_development_tokens import FIT_SHA256, MANIFEST_SHA256, load_fit_protocol
from .fitting_inputs import load_arms
from .fitting_math import fit_arms
from .prompts import content_hash
from .run_development_scores import GLOBAL_LOCK_ID
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run

RUN_ID = 'qwen3-ragtruth-development-fit-v1'


def software_versions(protocol):
    versions = {'python': platform.python_version(), **{k: version(k) for k in ('numpy', 'scipy')}}
    expected = protocol['software_for_future_fit']
    if ('.'.join(versions['python'].split('.')[:2]) != expected['python']
            or any(versions[k] != expected[k] for k in ('numpy', 'scipy'))):
        raise RunConflict('use Python 3.12 and the exact requirements-fit.txt environment')
    return versions


def save_fit(arms, protocol, identity, directory):
    path = directory / 'fit.json'
    with exclusive_run(directory):
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            digest = saved.pop('report_sha256')
            if content_hash(saved) != digest or saved['identity'] != identity:
                raise RunConflict('existing fit identity/checksum differs; preserve the artifact')
            saved['report_sha256'] = digest
            return saved, False
        started = time.monotonic()
        results = fit_arms(arms, protocol)
        report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'identity': identity,
                  'results': results, 'cpu_client_wall_seconds': time.monotonic() - started,
                  'generation_calls': 0, 'http_requests': 0, 'TEST_read': False, 'HaluBench_read': False,
                  'scope': 'reserved_TRAIN_development_only', 'strict_document_disjointness_established': False}
        report['report_sha256'] = content_hash(report)
        atomic_json(path, report)
        return report, True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    protocol = load_fit_protocol()
    versions = software_versions(protocol)
    revision = code_revision()
    # Prevent a scoring client from changing journals during read-only replay.
    with exclusive_run(run_directory(GLOBAL_LOCK_ID)):
        arms, inputs = load_arms(protocol)
        identity = {'fitter_code_revision': revision, 'versions': versions,
                    'fit_protocol_sha256': FIT_SHA256, 'development_manifest_sha256': MANIFEST_SHA256,
                    'verified_inputs': inputs}
        directory = run_directory(RUN_ID)
        report, fitted = save_fit(arms, protocol, identity, directory)
    result = report['results']
    print('Input replay: both complete reserved arms verified')
    print('New fitting invocation:', fitted)
    print('Calibration:', json.dumps(result['calibration'], indent=2))
    print('Operating threshold (TRAIN development selection):', json.dumps(result['operating_threshold']['selected'], indent=2))
    for arm, values in result['reliability'].items():
        print(arm, values['interpretation'])
        for kind in ('raw_uncalibrated', 'calibrated'):
            value = values[kind]
            print(kind, None if value is None else {k: value[k] for k in ('brier', 'ece')})
    print('Private fit:', directory / 'fit.json')
    print('Report SHA256:', report['report_sha256'])
    print('Post-thesis TRAIN only. No model calls or TEST evaluation. Development metrics are not test performance.')
    return 0 if result['calibration']['status'] == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
