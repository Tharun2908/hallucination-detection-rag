"""CPU-only decision freeze checks; fixtures do not claim real fit provenance."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import check_frozen_fit as check
from post_thesis.llm_judge.fitting_inputs import SCORING_REVISION, REPORT_HASHES
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    frozen = check.load_freeze()
    report = {'study_stage':'post_thesis','run_id':check.FIT_RUN_ID,
              'identity':{'fitter_code_revision':frozen['source_fitter_revision'],
                          'fit_protocol_sha256':frozen['fit_protocol_sha256'],
                          'development_manifest_sha256':frozen['development_manifest_sha256'],
                          'verified_inputs':{arm:{'summary_sha256':hashes[1], 'scoring_code_revision':SCORING_REVISION,
                                                   'rows':600} for arm,hashes in REPORT_HASHES.items()}},
              'generation_calls':0,'http_requests':0,'TEST_read':False,'HaluBench_read':False,
              'results':{'calibration':{'status':'accepted','failure_reasons':[], 'parameters':deepcopy(frozen['calibration']['parameters'])},
                         'operating_threshold':{'status':'selected','space':'raw_unsupported_log_odds','comparator':'>=',
                                               'selected':{'kind':'finite_margin','margin':-1.25,'margin_hex':(-1.25).hex()}}}}
    return sign(report,frozen)


def sign(report,frozen):
    report.pop('report_sha256',None)
    report['report_sha256']=content_hash(report)
    frozen['source_fit_report_sha256']=report['report_sha256']
    return report,frozen


class FreezeTests(unittest.TestCase):
    def test_current_prompt_profile_and_freeze_identity(self):
        frozen=check.load_freeze()
        self.assertEqual(content_hash(frozen),check.FREEZE_SHA256)
        self.assertEqual(frozen['generation_allowance'],0)
        self.assertIsNone(frozen['test_scoring_plan'])

    def test_valid_artificial_report_and_corrupt_hash(self):
        report,frozen=fixture()
        self.assertEqual(check.validate_fit(report,frozen)['raw_margin_threshold'],-1.25)
        report['generation_calls']=1
        with self.assertRaises(RunConflict): check.validate_fit(report,frozen)

    def test_changed_map_threshold_scope_or_source_rejected_even_with_consistent_hash(self):
        for kind in ('map','threshold','comparator','scope','source','rows','failed','revision'):
            report,frozen=fixture()
            if kind=='map': report['results']['calibration']['parameters']['a']=.5
            if kind=='threshold': report['results']['operating_threshold']['selected']['margin']=-1.0
            if kind=='comparator': report['results']['operating_threshold']['comparator']='>'
            if kind=='scope': report['TEST_read']=True
            if kind=='source': report['identity']['verified_inputs']['calibration']['summary_sha256']=REPORT_HASHES['operating_threshold'][1]
            if kind=='rows': report['identity']['verified_inputs']['calibration']['rows']=599
            if kind=='failed': report['results']['calibration']['status']='failed'
            if kind=='revision': report['identity']['fitter_code_revision']='other'
            sign(report,frozen)
            with self.subTest(kind=kind),self.assertRaises(RunConflict): check.validate_fit(report,frozen)

    def test_checker_replay_does_not_write_files_or_refit(self):
        report,frozen=fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);path=directory/'fit.json';path.write_text(json.dumps(report))
            before=path.read_bytes();mtime=path.stat().st_mtime_ns
            with patch.object(check,'load_freeze',return_value=frozen), patch.object(check,'run_directory',return_value=directory),patch('sys.argv',['check_frozen_fit']),redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(),0);self.assertEqual(check.main(),0)
            self.assertEqual(path.read_bytes(),before);self.assertEqual(path.stat().st_mtime_ns,mtime)
            self.assertEqual(list(directory.iterdir()),[path])


if __name__=='__main__': unittest.main()
