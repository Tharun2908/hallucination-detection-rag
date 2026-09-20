"""Validate the frozen evaluation core using artificial examples only."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from . import evaluation_math as metrics
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .prepare_pilot import file_sha256
from .prepare_test_manifest import save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID = 'evaluation-math-synthetic-check-v1'
CONTRACT_PATH = Path(__file__).parent / 'configs' / 'evaluation_math_v1.json'
CONTRACT_SHA256 = 'c092deea43883442914500fedb0627b7938c513ce5a77bc730079161b4425373'
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if content_hash(contract) != CONTRACT_SHA256 or contract['judge_freeze_sha256'] != FREEZE_SHA256:
        raise RunConflict('evaluation contract changed')
    load_freeze()
    for path, digest in contract['source_sha256'].items():
        if file_sha256(REPO_ROOT / path) != digest:
            raise RunConflict('evaluation implementation changed: ' + path)
    if (contract['main_systems'] != list(metrics.MAIN_SYSTEMS)
            or contract['bootstrap']['replicates'] != metrics.REPLICATES
            or contract['bootstrap']['minimum_valid_draws'] != metrics.MIN_VALID_DRAWS
            or contract['bootstrap']['seed'] != metrics.SEED
            or contract['ece_edges'] != metrics.ECE_EDGES):
        raise RunConflict('evaluation constants disagree with contract')
    return contract


def synthetic_fixture():
    return {'labels': [0, 1] * 6,
            'sample_ids': ['synthetic-' + str(i) for i in range(12)],
            'groups': [name for name in ('a', 'b', 'c', 'd', 'e', 'f') for _ in range(2)],
            'scores': {'judge': [-3., 1., 1., -2., .5, 0.] * 2,
                       'S4': [.1, .9, .8, .2, .3, .5] * 2,
                       'MiniCheck_7B': [.9, .1, .2, .7, .8, .3] * 2,
                       'S2_S4_metadata_free': [.1, .9, .8, .2, .3, .5] * 2}}


def evaluate_fixture():
    fixture = synthetic_fixture()
    # These flags apply exclusively to constructed arrays, never to legacy caches.
    synthetic_ready = {name: {'comparison_ready': True, **{key: True for key in metrics.PROVENANCE_FIELDS}}
                       for name in metrics.MAIN_SYSTEMS}
    result = metrics.paired_metrics(fixture['labels'], fixture['scores'], fixture['sample_ids'],
                                    fixture['groups'], synthetic_ready)
    if (result['shared_n'] != 12 or result['bootstrap']['replicates'] != 2000
            or result['shared_points']['judge']['confusion'] != {'tp': 4, 'fp': 4, 'fn': 2, 'tn': 2}
            or any(result['intervals'][name]['auroc']['valid_draws'] != 2000 for name in metrics.MAIN_SYSTEMS)):
        raise RunConflict('synthetic evaluation regression')
    blocked = metrics.paired_metrics(fixture['labels'], fixture['scores'], fixture['sample_ids'], fixture['groups'], {})
    if blocked['metrics_computed'] is not False:
        raise RunConflict('comparison provenance gate regression')
    return result


def check_once(directory, revision, versions):
    contract = load_contract()
    identity = {'code_revision': revision, 'versions': versions,
                'contract_sha256': content_hash(contract), 'fixtures_sha256': content_hash(synthetic_fixture())}
    path = Path(directory) / 'report.json'
    with exclusive_run(Path(directory)):
        if path.exists():
            bundle = json.loads(path.read_text(encoding='utf-8'))
            if (bundle['report_sha256'] != content_hash(bundle['report'])
                    or bundle['report']['identity'] != identity):
                raise RunConflict('existing synthetic evaluation check differs; preserve it')
            return bundle, False
        result = evaluate_fixture()
        report = {'study_stage': 'post_thesis', 'run_id': RUN_ID, 'identity': identity,
                  'scope': 'artificial_examples_only', 'status': 'passed',
                  'bootstrap_result': result, 'generation_calls': 0, 'fitting_calls': 0,
                  'benchmark_predictions_read': False, 'benchmark_metrics_computed': False,
                  'legacy_baseline_readiness_changed': False, 'scoring_allowance': 0}
        bundle = {'report_sha256': content_hash(report), 'report': report}
        save_once(path, bundle)
        return bundle, True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    contract = load_contract()
    versions = {'python': platform.python_version(), **{p: version(p) for p in contract['packages']}}
    if ('.'.join(versions['python'].split('.')[:2]) != '3.12'
            or any(versions[p] != expected for p, expected in contract['packages'].items())):
        raise RunConflict('use the existing pinned CPU baseline/fitting environment')
    bundle, created = check_once(run_directory(RUN_ID), revision, versions)
    result = bundle['report']['bootstrap_result']
    print('Post-thesis evaluation mathematics: artificial examples only')
    print('Status:', bundle['report']['status'])
    print('New synthetic evaluation:', created)
    print('Shared synthetic examples:', result['shared_n'])
    print('Registered bootstrap replicates:', result['bootstrap']['replicates'])
    print('Valid AUROC draws per system:', json.dumps({name: result['intervals'][name]['auroc']['valid_draws'] for name in metrics.MAIN_SYSTEMS}))
    print('Unready comparison blocked:', True)
    print('Contract SHA256:', CONTRACT_SHA256)
    print('Report SHA256:', bundle['report_sha256'])
    print('Code revision:', revision)
    print('Private report:', run_directory(RUN_ID) / 'report.json')
    print('No benchmark predictions read, benchmark metrics, inference, fitting or scoring allowance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
