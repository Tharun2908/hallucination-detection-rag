"""Verify fresh canonical HaluBench baselines without computing label metrics."""
import math

from .audit_halubench_tokens import TEST_MANIFEST_SHA256
from .check_frozen_fit import FREEZE_SHA256
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .freeze_comparison import read_report, LEGACY_PROVENANCE, FUSION_RECOVERY, LIMITATIONS
from .prompts import content_hash
from .prepare_test_manifest import save_once
from .runner import run_directory
from .storage import RunConflict, exclusive_run

RUN_ID = 'halubench-fresh-comparison-manifest-v1'
SOURCES = {
    'S4': ('s4-halubench-test-fresh-v1', 'summary.json', '2d68d0657d983db0df84b146264652eea00034734e82d09a42126c56e4ba0f2c', 'unsupported_score'),
    'S2_dependency': ('s2-halubench-test-fresh-v1', 'summary.json', 'e71c27c5c62f2b45f8332ee76c8a9bc6ba9d15915970921178a10daa47b9bb35', 'raw_min_relevance'),
    'S2_S4_metadata_free': ('halubench-fresh-metadata-free-fusion-v1', 'report.json', '2dbd25f11f167cc7e0ec74c1557450dcf2407cb9ea89db53e876d84f2acea415', 'unsupported_score'),
    'MiniCheck_7B': ('minicheck-halubench-test-fresh-v1', 'summary.json', '18a157b4caf96fba23b35b8adb9f961dc5df45fd847a3e026305df7289e181b0', 'support_score'),
}


def descriptor():
    contract = load_contract()
    return {'study_stage': 'post_thesis', 'run_id': RUN_ID,
        'scope': 'fresh_runtime_zero_shot_canonical_HaluBench_TEST_baselines',
        'TEST_manifest_sha256': TEST_MANIFEST_SHA256, 'judge_freeze_sha256': FREEZE_SHA256,
        'evaluation_contract_sha256': CONTRACT_SHA256, 'examples': 8000, 'bootstrap_components': 7198,
        'targets_bound_by': 'canonical input manifest; all baseline IDs/order/input hashes verified on every load',
        'source_reports': {k: {'run_id':v[0], 'filename':v[1], 'report_sha256':v[2], 'score_field':v[3]} for k,v in SOURCES.items()},
        'baselines': {k: {'comparison_ready':True, 'variant':'fresh_post_thesis_runtime',
                         'operating_rule':contract['operating_rules'][k],
                         'historical_training_provenance_independently_verified':False}
                      for k in ('S4','MiniCheck_7B','S2_S4_metadata_free')},
        'historical_threshold_provenance_sha256': LEGACY_PROVENANCE[2],
        'fusion_recovery_report_sha256': FUSION_RECOVERY[2],
        'limitations': LIMITATIONS + [
            'All operating policies transfer from RAGTruth; none are fitted or selected on HaluBench.',
            'S2 has 1886 empty-pair raw-zero features and 35 truncated context pairs; S4 truncates 850 contexts.',
            'MiniCheck has 9158 sentence requests with no truncated prompts; Qwen consumes the full answer/context.',
            'The unchanged canonical split has 7838 TEST groups merged to 7198 exact-passage bootstrap components.',
            'Exact passages link 994 TEST rows to adaptation components; no adaptation training is used.',
            'RAGTruth fresh/legacy MiniCheck drift is documented; no HaluBench legacy runtime is selected by performance.'],
        'judge_predictions_pending':True, 'paired_metrics_ready':False,
        'model_calls':0, 'fitting_calls':0, 'TEST_metrics_computed':False, 'adaptation_training':False}


COMPARISON_SHA256 = content_hash(descriptor())


def align_sources(reports, targets):
    if (len(targets) != 8000 or len({t['sample_id'] for t in targets}) != 8000
            or len({t['component_sha256'] for t in targets}) != 7198):
        raise RunConflict('canonical HaluBench population or bootstrap groups differ')
    for name,spec in SOURCES.items():
        r=reports[name];identity=r if name=='S2_S4_metadata_free' else r['identity']
        if (r['study_stage']!='post_thesis' or r['run_id']!=spec[0]
                or identity['TEST_manifest_sha256']!=TEST_MANIFEST_SHA256
                or r['valid_scores']!=8000 or len(r['predictions'])!=8000
                or r.get('pending',0)!=0 or r.get('terminal_failures',0)!=0
                or r['HaluBench_read'] is not True or r['adaptation_training'] is not False):
            raise RunConflict('incomplete or non-zero-shot HaluBench baseline: '+name)
        for i,(row,target) in enumerate(zip(r['predictions'],targets)):
            value=row[spec[3]]
            if (row['status']!='ok' or row['test_index']!=i or target['test_index']!=i
                    or any(row[k]!=target[k] for k in ('sample_id','input_sha256'))
                    or type(value) not in (int,float) or not math.isfinite(value)
                    or (name!='S2_dependency' and not 0<=value<=1)):
                raise RunConflict('baseline input alignment or score differs: '+name)


def verify_and_save(targets):
    reports={name:read_report(spec) for name,spec in SOURCES.items()}
    align_sources(reports,targets)
    legacy=read_report(LEGACY_PROVENANCE);recovery=read_report(FUSION_RECOVERY)
    contract=load_contract()
    for name,old in [('S4','S4'),('MiniCheck_7B','MC_7B')]:
        rule=contract['operating_rules'][name];entry=legacy['threshold_reproduction'][old]
        if (entry['status']!='matches_historical_TRAIN_audit'
                or entry['reproduced']['threshold']!=rule['threshold'] or entry['comparator']!=rule['comparator']):
            raise RunConflict('historical TRAIN operating policy changed')
    fusion=reports['S2_S4_metadata_free']
    if (fusion['full_model']!=recovery['results']['full_model']
            or fusion['full_model']['metadata_features'] is not False
            or fusion['threshold']['value']!=.45 or fusion['threshold']['comparator']!='>='
            or fusion['fusion_recovery_report_sha256']!=FUSION_RECOVERY[2]
            or fusion['fresh_source_report_sha256']!={'S2':SOURCES['S2_dependency'][2],'S4':SOURCES['S4'][2]}):
        raise RunConflict('frozen fusion lineage changed')
    manifest=descriptor()
    if content_hash(manifest)!=COMPARISON_SHA256: raise RunConflict('comparison descriptor changed')
    directory=run_directory(RUN_ID)
    with exclusive_run(directory):
        save_once(directory/'manifest.json',{'manifest_sha256':COMPARISON_SHA256,'manifest':manifest})
    return manifest
