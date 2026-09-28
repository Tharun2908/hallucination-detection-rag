"""Freeze aligned fresh baseline artifacts and limitations; no TEST metrics."""
import argparse
import json
import math

from .audit_test_tokens import TEST_MANIFEST_SHA256, validate_inputs
from .check_evaluation_math import CONTRACT_SHA256, load_contract
from .check_frozen_fit import FREEZE_SHA256, load_freeze
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID='ragtruth-fresh-comparison-manifest-v1'
SOURCES={
 'S4':('s4-ragtruth-test-fresh-v1','summary.json','4f502fc775f15332eb78aa5070ef2d056015241d91a67f5c1ceb254ba2f2dea6','unsupported_score'),
 'S2_dependency':('s2-ragtruth-test-fresh-v1','summary.json','8960c4a8c5d2c873b6f68241bb3c05810d6848fab392b9e8cf5407cdda4a013b','raw_min_relevance'),
 'S2_S4_metadata_free':('ragtruth-fresh-metadata-free-fusion-v1','report.json','931476d6815f8959d4073ef13abf595a077bf89da31ff82f9b72571e17296e3d','unsupported_score'),
 'MiniCheck_7B':('minicheck-ragtruth-test-fresh-v1','summary.json','365f4d3b097ff1361bd1ccbea911c846d8b26560bd7882d1834801d79e9237f2','support_score'),
}
LEGACY_PROVENANCE=('ragtruth-legacy-baseline-provenance-v1','report.json','2e85cc46aeb4ade600c3c70680c200338f580dd20875a1a122d71f0988575e7c')
FUSION_RECOVERY=('ragtruth-metadata-free-fusion-recovery-v1','report.json','1c90a003abbda9044767bb9167b1ecc171f7a8a746b87093a143d833ba4ec0d6')
MC_AGREEMENT=('minicheck-fresh-legacy-agreement-v1','report.json','60ca5e5e1c2a3ce5a49c40ee7e2ee004b055ffd30876ee95dc9739a9d69a9489')
LIMITATIONS=[
 'Primary baselines are fresh post-thesis runtime variants, not exact reproductions of submitted thesis caches.',
 'MiniCheck fresh/legacy mean absolute score difference is 0.02947428918387699; 100/2700 fixed-threshold decisions differ. Cause remains unresolved.',
 'MiniCheck threshold is transferred from historical TRAIN scores; it was not selected or recalibrated on fresh TEST scores.',
 'The original MiniCheck script grouped 16 answers per score call; fresh inference uses one answer per call. Runtime, batching and tokenizer differences are possible contributors, not demonstrated causes.',
 'Fresh inference proves current input and checkpoint linkage; it does not independently prove historical training exclusions or checkpoint-to-legacy-cache linkage.',
 'S4 truncates pairs at 512 tokens; S2 filters/splits sentences and may truncate pairs. MiniCheck chunks/splits text; the judge consumes full answer/context. This is a practical-system comparison, not an architecture-only or common-evidence experiment.',
 'Baseline training/development exposure differs from the judge, including possible exposure to judge development examples.',
 'Baseline probabilities are uncalibrated here; judge raw and frozen calibrated probabilities are both reported.',
 'Observed inference times have different measurement scopes and GPU sharing histories; they are not a controlled speedup benchmark.',
 'Exact-overlap components are fixed; strict fuzzy document disjointness is not established.',
]


def read_report(spec):
    directory,filename,digest=spec[:3]
    bundle=json.loads((run_directory(directory)/filename).read_text(encoding='utf-8'))
    if bundle.get('report_sha256')!=digest or content_hash(bundle['report'])!=digest:
        raise RunConflict('changed or missing pinned report: '+directory)
    return bundle['report']


def align_sources(reports,targets):
    """Check current content linkage, not historical training provenance."""
    if not targets or len({r['sample_id'] for r in targets})!=len(targets):
        raise RunConflict('nonempty unique comparison targets required')
    projected={}
    for name,spec in SOURCES.items():
        report=reports[name]
        identity=report if name=='S2_S4_metadata_free' else report['identity']
        if (report['study_stage']!='post_thesis' or report['run_id']!=spec[0]
                or identity['TEST_manifest_sha256']!=TEST_MANIFEST_SHA256
                or report['valid_scores']!=len(targets)
                or report.get('pending',0)!=0 or report.get('terminal_failures',0)!=0):
            raise RunConflict('incomplete or differently aligned baseline: '+name)
        rows=report['predictions']
        if len(rows)!=len(targets): raise RunConflict('baseline coverage differs: '+name)
        projected[name]=[]
        for index,(row,target) in enumerate(zip(rows,targets)):
            if (target['test_index']!=index or row['status']!='ok' or row['test_index']!=index
                    or any(row[k]!=target[k] for k in ('sample_id','input_sha256'))):
                raise RunConflict('baseline input content/order differs: '+name)
            score=row[spec[3]]
            if (type(score) not in (int,float) or not math.isfinite(score)
                    or (name!='S2_dependency' and not 0<=score<=1)):
                raise RunConflict('invalid baseline score: '+name)
            projected[name].append({'sample_id':row['sample_id'],'test_index':index,
                                    'input_sha256':row['input_sha256'],'score':score})
    return projected


def build_manifest(revision, contract, targets, reports, legacy, recovery, drift):
    """Build a scoped comparison record only after artifact and alignment checks."""
    projected=align_sources(reports,targets)
    for name,system in [('S4','S4'),('MiniCheck_7B','MC_7B')]:
        rule=contract['operating_rules'][name]; historical=legacy['threshold_reproduction'][system]
        if (historical['status']!='matches_historical_TRAIN_audit'
                or historical['reproduced']['threshold']!=rule['threshold']
                or historical['comparator']!=rule['comparator']):
            raise RunConflict('baseline TRAIN threshold provenance differs')
    if (recovery['results']['threshold']['selected']['threshold']!=contract['operating_rules']['S2_S4_metadata_free']['threshold']
            or reports['S2_S4_metadata_free']['threshold']['value']!=.45
            or reports['S2_S4_metadata_free']['threshold']['comparator']!='>='
            or reports['S2_S4_metadata_free']['full_model']!=recovery['results']['full_model']
            or reports['S2_S4_metadata_free']['full_model']['metadata_features'] is not False
            or reports['S2_S4_metadata_free']['fusion_recovery_report_sha256']!=FUSION_RECOVERY[2]
            or reports['S2_S4_metadata_free']['fresh_source_report_sha256']!={'S2':SOURCES['S2_dependency'][2], 'S4':SOURCES['S4'][2]}
            or drift['fresh_report_sha256']!=SOURCES['MiniCheck_7B'][2]
            or drift['agreement']['fixed_threshold_decision_changes']!=100):
        raise RunConflict('fusion or MiniCheck drift provenance differs')
    baselines={}
    for name in ('S4','MiniCheck_7B','S2_S4_metadata_free'):
        baselines[name]={
            'variant':'fresh_post_thesis_runtime', 'report_sha256':SOURCES[name][2],
            'aligned_predictions_sha256':content_hash(projected[name]),'valid_scores':len(targets),
            'score_field':SOURCES[name][3],'operating_rule':contract['operating_rules'][name],
            'comparison_ready':True,'input_content_verified':True,'checkpoint_provenance_verified':True,
            'threshold_provenance_verified':True,'evidence_visibility_documented':True,
            'provenance_scope':'these exact fresh inference artifacts and fixed historical TRAIN-derived operating policies',
            'historical_training_provenance_independently_verified':False,
            'exact_thesis_cache_reproduction_claimed':False,
            'evidence_visibility':('512-token longest-first pair truncation' if name=='S4' else
                 'NLTK document chunks and answer sentences; zero truncated prompts in this run' if name=='MiniCheck_7B' else
                 'S2 sentence filtering/pair truncation and S4 512-token pair truncation'),
        }
    manifest={'study_stage':'post_thesis','run_id':RUN_ID,'code_revision':revision,
              'status':'fresh_baselines_frozen_judge_predictions_pending',
              'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'judge_freeze_sha256':FREEZE_SHA256,
              'evaluation_contract_sha256':CONTRACT_SHA256,'targets':targets,
              'groups':{'count':len({r['component_sha256'] for r in targets}),
                        'mapping_sha256':content_hash([(r['sample_id'],r['component_sha256']) for r in targets]),
                        'source':'unchanged canonical TEST exact-overlap components'},
              'baselines':baselines,'dependency_report_sha256':{name:spec[2] for name,spec in SOURCES.items()},
              'historical_threshold_provenance_sha256':LEGACY_PROVENANCE[2],
              'fusion_recovery_report_sha256':FUSION_RECOVERY[2],'minicheck_agreement_report_sha256':MC_AGREEMENT[2],
              'minicheck_drift':{k:drift['agreement'][k] for k in ('mean_absolute_difference','maximum_absolute_difference','fixed_threshold_decision_changes')},
              'primary_baselines_selection':'fresh inference chosen for verifiable current inputs/checkpoints; not selected using TEST label metrics',
              'legacy_cache_policy':'preserve as historical artifacts; do not mix scores or choose the more favorable runtime',
              'limitations':LIMITATIONS,'judge_comparison_ready':False,'judge_TEST_predictions_present':False,
              'paired_metrics_ready':False,'model_calls':0,'fitting_calls':0,'TEST_metrics_computed':False,
              'HaluBench_read':False,'source_files_rewritten':False,'generation_allowance':0}
    return manifest


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision=code_revision(); frozen=load_freeze(); contract=load_contract()
    bundle=json.loads((run_directory(TEST_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    validate_inputs(bundle,frozen)
    targets=[{k:r[k] for k in ('sample_id','test_index','input_sha256','component_sha256')}
             for r in bundle['manifest']['offline_rows']]
    if len(targets)!=2700 or len({r['component_sha256'] for r in targets})!=450:
        raise RunConflict('canonical TEST population or grouping differs')
    reports={name:read_report(spec) for name,spec in SOURCES.items()}
    manifest=build_manifest(revision,contract,targets,reports,read_report(LEGACY_PROVENANCE),
                            read_report(FUSION_RECOVERY),read_report(MC_AGREEMENT))
    output={'manifest_sha256':content_hash(manifest),'manifest':manifest}
    directory=run_directory(RUN_ID)
    with exclusive_run(directory): created=save_once(directory/'manifest.json',output)
    print('Post-thesis fresh baseline comparison manifest')
    print('New manifest:',created)
    print('Examples / groups:',len(targets),'/',manifest['groups']['count'])
    print('Baseline readiness:',json.dumps({k:v['comparison_ready'] for k,v in manifest['baselines'].items()}))
    print('MiniCheck drift:',json.dumps(manifest['minicheck_drift']))
    print('Thresholds:',json.dumps({k:v['operating_rule'] for k,v in manifest['baselines'].items()}))
    print('Manifest SHA256:',output['manifest_sha256'])
    print('Private manifest:',directory/'manifest.json')
    print('Fresh-runtime comparison only. Historical provenance limits retained. Judge predictions pending.')
    print('No model calls, fitting, TEST metrics or scoring allowance; bounded judge execution remains next.')
    return 0


if __name__=='__main__': raise SystemExit(main())
