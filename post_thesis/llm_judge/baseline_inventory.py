"""Read-only RAGTruth cache inventory; index agreement is not text provenance."""
import json
import math
from pathlib import Path

from .prepare_pilot import file_sha256
from .prompts import content_hash

SPECS = {
    'S4': ('signal4_results_test.json','signal4_score',True),
    'MiniCheck_7B': ('minicheck_results_test_7b.json','minicheck_score',True),
    'S2_dependency': ('relevance_results_test_v2.json','raw_min_relevance',False),
}


def inspect_cache(path, expected_rows, field, probability):
    report={'filename':path.name,'path':str(path.resolve()),'comparison_ready':False}
    if not path.is_file(): return {**report,'status':'missing'}
    report['sha256']=file_sha256(path)
    try:
        rows=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):
            raise ValueError('expected list of per-example records')
        expected={r['test_index']:r for r in expected_rows}
        seen=set();missing_scores=0;content_matches=0
        for row in rows:
            idx=row.get('idx')
            if type(idx) is not int or idx not in expected or idx in seen:
                raise ValueError('invalid, duplicate or non-TEST row index')
            seen.add(idx);target=expected[idx]
            label=row.get('ground_truth_hallucination')
            if type(label) not in (bool,int) or label not in (0,1) or int(label)!=target['label']:
                raise ValueError('label/index mismatch')
            for key in ('model','task_type'):
                if key in row and row[key]!=target['metadata'][key]:
                    raise ValueError('metadata/index mismatch')
            if 'sample_id' in row and row['sample_id']!=target['sample_id']:
                raise ValueError('sample ID/index mismatch')
            if field not in row: raise ValueError('missing expected score field '+field)
            value=row[field]
            if value is None: missing_scores+=1
            elif (type(value) not in (int,float) or not math.isfinite(value)
                  or (probability and not 0<=value<=1)):
                raise ValueError('invalid score value')
            if 'input_sha256' in row:
                if row['input_sha256']!=target['input_sha256']: raise ValueError('input hash mismatch')
                content_matches+=1
        missing_ids=sorted(set(expected)-seen)
        report.update(rows=len(rows),missing_indices=len(missing_ids),missing_index_examples=missing_ids[:20],
                      missing_scores=missing_scores,valid_scores=len(rows)-missing_scores,
                      rows_with_matching_input_hash=content_matches)
        if missing_ids or missing_scores: status='incomplete'
        elif content_matches!=len(expected): status='index_consistent_content_provenance_unproven'
        else: status='content_hashes_match_model_and_threshold_provenance_pending'
        report.update(status=status,model_checkpoint_provenance_verified=False,
                      threshold_provenance_verified=False,evidence_visibility_verified=False)
    except (ValueError,TypeError,KeyError,UnicodeError) as error:
        report.update(status='invalid_cache',error=str(error))
    return report


def inventory(directory, bundle):
    directory=Path(directory)
    baselines={name:inspect_cache(directory/filename,bundle['manifest']['offline_rows'],field,probability)
               for name,(filename,field,probability) in SPECS.items()}
    baselines['S2_S4_metadata_free']={
        'status':'per_example_fusion_predictions_and_fitted_model_provenance_pending',
        'comparison_ready':False,'metadata_features':False,'dependencies':['S4','S2_dependency'],
        'note':'Do not reconstruct predictions from aggregate metrics or substitute metadata-aware fusion.'}
    record={'study_stage':'post_thesis','version':'ragtruth-baseline-inventory-v1',
            'test_manifest_sha256':bundle['manifest_sha256'],'baseline_directory':str(directory.resolve()),
            'baselines':baselines,'metrics_computed':False,'model_calls':0,'comparison_ready':False,
            'historical_cache_scores_read_if_present':True,
            'limitation':'Index/label agreement does not prove text identity, checkpoint or threshold provenance.'}
    return {'inventory_sha256':content_hash(record),'inventory':record}
