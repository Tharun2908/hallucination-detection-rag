"""Verify the completed post-thesis fit without refitting or model calls."""
import argparse
import json
from pathlib import Path

from .audit_development_tokens import FIT_SHA256, MANIFEST_SHA256
from .fitting_inputs import REPORT_HASHES, SCORING_REVISION
from .label_score_contract import LABEL_PROMPT
from .prompts import content_hash
from .runner import run_directory
from .storage import RunConflict
from .vllm_backend import load_profile

FREEZE_PATH = Path(__file__).parent / 'configs' / 'frozen_judge_v1.json'
FREEZE_SHA256 = '90a70a63c550e87ad0f3fa41f11e1de99d9a08fc85c47de23f6ffafd3314fde8'
FIT_RUN_ID = 'qwen3-ragtruth-development-fit-v1'


def load_freeze():
    frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    if (content_hash(frozen) != FREEZE_SHA256
            or frozen['fit_protocol_sha256'] != FIT_SHA256
            or frozen['development_manifest_sha256'] != MANIFEST_SHA256
            or frozen['prompt_version'] != LABEL_PROMPT.version
            or frozen['prompt_sha256'] != LABEL_PROMPT.sha256
            or frozen['serving_profile'] != load_profile('label-score-v1')):
        raise RunConflict('frozen judge identity has changed')
    return frozen


def validate_fit(report, frozen):
    expected_hash = frozen['source_fit_report_sha256']
    if (report.get('report_sha256') != expected_hash
            or content_hash({k:v for k,v in report.items() if k != 'report_sha256'}) != expected_hash):
        raise RunConflict('private fit does not match the completed fit hash')
    identity = report['identity']
    if (report['study_stage'] != 'post_thesis' or report['run_id'] != FIT_RUN_ID
            or identity['fitter_code_revision'] != frozen['source_fitter_revision']
            or identity['fit_protocol_sha256'] != frozen['fit_protocol_sha256']
            or identity['development_manifest_sha256'] != frozen['development_manifest_sha256']
            or report['generation_calls'] != 0 or report['http_requests'] != 0
            or report['TEST_read'] is not False or report['HaluBench_read'] is not False):
        raise RunConflict('fit provenance or scope mismatch')
    sources = identity['verified_inputs']
    if set(sources) != {'calibration','operating_threshold'}:
        raise RunConflict('expected both independently reserved fit arms')
    for arm, source in sources.items():
        if (source['summary_sha256'] not in REPORT_HASHES[arm]
                or source['scoring_code_revision'] != SCORING_REVISION or source['rows'] != 600):
            raise RunConflict('fit scoring inputs differ from the completed reserved runs')
    result = report['results']
    calibration = result['calibration']
    expected = frozen['calibration']
    if (calibration['status'] != 'accepted' or calibration['failure_reasons']
            or calibration['parameters'] != expected['parameters']
            or {k:float(v).hex() for k,v in calibration['parameters'].items()} != expected['parameter_hex']):
        raise RunConflict('calibration differs from the accepted frozen map')
    threshold = result['operating_threshold']
    selected = threshold['selected']
    if (threshold['status'] != 'selected' or threshold['space'] != 'raw_unsupported_log_odds'
            or threshold['comparator'] != '>=' or selected['kind'] != 'finite_margin'
            or selected['margin'] != frozen['operating_threshold']['margin']
            or selected['margin_hex'] != frozen['operating_threshold']['margin_hex']
            or float(selected['margin']).hex() != selected['margin_hex']):
        raise RunConflict('operating threshold differs from the frozen raw-margin rule')
    return {'fit_report_sha256':expected_hash,'freeze_sha256':content_hash(frozen),
            'calibration_parameters':calibration['parameters'],
            'raw_margin_threshold':selected['margin'],'comparator':'>=',
            'generation_calls':0,'fitting_calls':0,'source_files_rewritten':False}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    frozen = load_freeze()
    path = run_directory(FIT_RUN_ID) / 'fit.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps(validate_fit(report, frozen), indent=2))
    print('Frozen fit verified. Post-thesis only; no refit or model calls.')
    print('Next: benchmark input/baseline alignment and bounded execution manifest.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
