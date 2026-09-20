"""CPU-only legacy-cache identity, OOF and historical threshold reproduction."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path

from .check_frozen_fit import FIT_RUN_ID, load_freeze, validate_fit
from .prepare_pilot import TRAIN_SHA256, _records, file_sha256, read_train
from .prepare_test_manifest import RUN_ID as TEST_RUN_ID, save_once
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, exclusive_run

RUN_ID='ragtruth-legacy-baseline-provenance-v1'
TEST_MANIFEST_SHA256='bc564d50ed4c0fcff17b5c8dda4feef558d896f2f0c8b4c7239a5085f9ccc3d3'
THRESHOLD_REPORT_SHA256='1a7af7f7dc7c98dca5672fb9467dc0c9402880043282ebf2a12f16503b8b1dc6'
REPO_ROOT=Path(__file__).resolve().parents[2]
FILES={
 'signal4_results_train_oof.json':('7be2eb53ac3767e17477ea737390119fa5a809764679ff3424212a51d9904b42','train','signal4_score',True),
 'signal4_results_test.json':('91d38d199d6e3c6e26ec4adfdb7cbd6767496de75e3a577daed159648b47c1b2','test','signal4_score',True),
 'minicheck_results_train_7b.json':('720651ed41938ebeb92f84b3925237e730db1be04ddfbf35554949b80cc373d9','train','minicheck_score',True),
 'minicheck_results_test_7b.json':('4e20383ab51a20c53934db5073d04e124e22079bc9a8bd99d5253eee7e320bcf','test','minicheck_score',True),
 'relevance_results_train_v2.json':('7377db83e163a318dac16a04e6574746c89e4bb78f93bd1698461c35b82f6647','train','raw_min_relevance',False),
 'relevance_results_test_v2.json':('93d93755b584146b63aece0eab053ecb919713408258fa8f6fb06c71709c3b36','test','raw_min_relevance',False),
}


def align(rows, expected, field, probability):
    if not isinstance(rows,list) or len(rows)!=len(expected):
        raise RunConflict('cache row count differs from pinned split')
    by_idx={};missing=0
    for row in rows:
        idx=row.get('idx')
        if type(idx) is not int or not 0<=idx<len(expected) or idx in by_idx:
            raise RunConflict('duplicate, invalid or out-of-split cache index')
        target=expected[idx];label=row.get('ground_truth_hallucination')
        if (type(label) not in (int,bool) or label not in (0,1) or int(label)!=target['label']
                or any(row.get(k)!=target['metadata'][k] for k in ('model','task_type'))):
            raise RunConflict('cache label or metadata disagrees with pinned row index')
        value=row[field]
        if value is None:missing+=1
        elif type(value) not in (int,float) or not math.isfinite(value) or (probability and not 0<=value<=1):
            raise RunConflict('invalid cached score')
        by_idx[idx]=row
    if set(by_idx)!=set(range(len(expected))):raise RunConflict('incomplete cache index coverage')
    return [by_idx[i] for i in range(len(expected))], {'rows':len(rows),'valid_scores':len(rows)-missing,
            'missing_scores':missing,'label_and_metadata_alignment':True,
            'answer_context_alignment_verified':False,
            'limitation':'Legacy caches contain no answer/context hashes; do not manufacture retrospective input provenance.'}


def oof_check(rows):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    labels=np.array([int(r['ground_truth_hallucination']) for r in rows])
    expected={}
    for fold,(_,heldout) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(labels)),labels),start=1):
        expected.update({int(i):fold for i in heldout})
    if any(type(r.get('fold')) is not int or r['fold']!=expected[i] or r.get('score_type')!='out_of_fold' for i,r in enumerate(rows)):
        raise RunConflict('stored OOF membership/type differs from historical five-fold plan')
    return {'fold_membership_matches':True,'fold_counts':dict(sorted(Counter(str(r['fold']) for r in rows).items())),
            'score_type':'out_of_fold','training_exclusion_independently_proven':False,
            'limitation':'Fold tags and code match; this does not independently prove which examples trained each checkpoint.'}


def reproduce_threshold(rows, field, *, support, reference):
    import numpy as np
    grid=np.arange(.10,.901,.05) if support else np.arange(.05,.951,.05)
    # Preserve the historical float grid and first-maximum tie rule exactly.
    valid=[r for r in rows if r[field] is not None]
    if not support and len(valid)!=len(rows):
        raise RunConflict('historical S4 threshold audit requires complete TRAIN scores')
    if not valid:raise RunConflict('no valid TRAIN scores for threshold reproduction')
    best=None;table=[]
    for t in grid:
        predictions=[bool(r[field]<t if support else r[field]>=t) for r in valid]
        y=[int(r['ground_truth_hallucination']) for r in valid]
        tp=sum(a and b for a,b in zip(predictions,y));fp=sum(a and not b for a,b in zip(predictions,y));fn=sum(not a and b for a,b in zip(predictions,y))
        denominator=2*tp+fp+fn;f1=2*tp/denominator if denominator else 0.0
        candidate={'threshold':float(t),'threshold_hex':float(t).hex(),'tp':tp,'fp':fp,'fn':fn,'f1':f1}
        table.append(candidate)
        if best is None or f1>best['f1']:best=candidate
    matches=best['threshold']==reference['threshold'] and abs(best['f1']-reference['train_f1'])<=1e-12
    return {'status':'matches_historical_TRAIN_audit' if matches else 'mismatch_do_not_replace_historical_policy',
            'score_space':'support_probability' if support else 'unsupported_probability',
            'comparator':'<' if support else '>=','selected_on_TRAIN_only':True,'valid_train_rows':len(valid),
            'missing_train_rows':len(rows)-len(valid),'reproduced':best,
            'historical_reference':{'threshold':reference['threshold'],'train_f1':reference['train_f1']},
            'grid_candidates':table,'policy_changed':False}


def checkpoint_inventory(directory):
    """Hash file bytes only. Never deserialize model weights or pickle/bin objects."""
    directory=Path(directory)
    if not directory.is_dir():return {'path':str(directory),'status':'missing','model_to_cache_link_verified':False}
    files={}
    for path in sorted(directory.iterdir()):
        if path.is_file() and (path.suffix in ('.json','.safetensors') or path.name in ('pytorch_model.bin','spm.model','vocab.txt')
                              or (path.name.startswith('pytorch_model-') and path.suffix=='.bin')):
            files[path.name]={'bytes':path.stat().st_size,'sha256':file_sha256(path)}
    weights=any(name.endswith('.safetensors') or (name.startswith('pytorch_model') and name.endswith('.bin')) for name in files)
    return {'path':str(directory.resolve()),'status':'present','files':files,'weight_files_present':weights,
            'model_to_cache_link_verified':False,'deserialized':False,
            'limitation':'Current snapshot fingerprints do not prove these weights generated the historical score cache.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--train-parquet',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path,default=Path('/workspace'))
    args=parser.parse_args();revision=code_revision()
    versions={p:version(p) for p in ('numpy','scipy','scikit-learn')}
    if versions!={'numpy':'1.26.4','scipy':'1.14.1','scikit-learn':'1.5.2'}:
        raise RunConflict('install requirements-baseline-audit.txt in the CPU environment')
    frozen=load_freeze();fit=json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8'));validate_fit(fit,frozen)
    bundle=json.loads((run_directory(TEST_RUN_ID)/'manifest.json').read_text(encoding='utf-8'))
    if bundle.get('manifest_sha256')!=TEST_MANIFEST_SHA256 or content_hash(bundle['manifest'])!=TEST_MANIFEST_SHA256:
        raise RunConflict('expected completed cluster TEST manifest')
    train,_=_records(read_train(args.train_parquet))
    expected={'train':train,'test':bundle['manifest']['offline_rows']}
    caches={};checks={}
    for name,(digest,split,field,probability) in FILES.items():
        path=args.baseline_dir/name
        if file_sha256(path)!=digest:raise RunConflict('cache differs from operator-reported original: '+name)
        rows=json.loads(path.read_text(encoding='utf-8'))
        caches[name],checks[name]=align(rows,expected[split],field,probability)
        checks[name].update(sha256=digest,split=split)
    oof=oof_check(caches['signal4_results_train_oof.json'])
    reference_path=REPO_ROOT/'results/evaluation/table41_threshold_audit_results.json'
    if file_sha256(reference_path)!=THRESHOLD_REPORT_SHA256:raise RunConflict('historical threshold reference changed')
    reference=json.loads(reference_path.read_text())['results']
    thresholds={name:reproduce_threshold(caches[filename],field,support=support,reference=reference[name])
                for name,filename,field,support in [('S4','signal4_results_train_oof.json','signal4_score',False),
                                                    ('MC_7B','minicheck_results_train_7b.json','minicheck_score',True)]}
    snapshots={}
    for name in ['signal4_model']+['signal4_oof_models/fold_'+str(i) for i in range(1,6)]:
        print('Fingerprinting saved checkpoint files:',name,flush=True)
        snapshots[name]=checkpoint_inventory(args.checkpoint_root/name)
    report={'study_stage':'post_thesis','version':'legacy-baseline-provenance-v1','code_revision':revision,'versions':versions,
            'TEST_manifest_sha256':TEST_MANIFEST_SHA256,'TRAIN_parquet_sha256':TRAIN_SHA256,
            'fit_report_sha256':frozen['source_fit_report_sha256'],'cache_checks':checks,'oof_check':oof,
            'historical_threshold_report_sha256':THRESHOLD_REPORT_SHA256,'threshold_reproduction':thresholds,
            'checkpoint_snapshots':snapshots,'generation_calls':0,'judge_fitting_calls':0,
            'TRAIN_threshold_reproduction':True,'TEST_metrics_computed':False,'source_files_rewritten':False,
            'fusion_model_reconstructed':False,'comparison_ready':False,
            'remaining_gaps':['answer/context identity of legacy inference','checkpoint-to-cache linkage',
                              'historical evidence visibility','metadata-free fusion model and predictions']}
    output={'report_sha256':content_hash(report),'report':report};directory=run_directory(RUN_ID)
    with exclusive_run(directory):created=save_once(directory/'report.json',output)
    print('Created provenance report.' if created else 'Identical provenance report verified; no rewrite.')
    print('Cache checks:',json.dumps(checks,indent=2));print('OOF:',json.dumps(oof))
    print('Threshold reproduction:',json.dumps({k:{f:v[f] for f in ('status','score_space','comparator','reproduced','historical_reference')} for k,v in thresholds.items()},indent=2))
    print('Checkpoint summaries:',json.dumps({k:{'status':v['status'],'weight_files_present':v.get('weight_files_present',False),'files':len(v.get('files',{}))} for k,v in snapshots.items()},indent=2))
    print('Report SHA256:',output['report_sha256']);print('Code revision:',revision);print('Private report:',directory/'report.json')
    print('Post-thesis legacy-baseline audit only. No TEST metrics, new model calls or judge refitting. Comparison provenance remains incomplete.')
    return 0 if all(v['status']=='matches_historical_TRAIN_audit' for v in thresholds.values()) else 1


if __name__=='__main__':raise SystemExit(main())
