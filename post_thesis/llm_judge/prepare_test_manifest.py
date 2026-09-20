"""Offline RAGTruth TEST text/native alignment; no judge calls or metrics."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from .audit_development_tokens import load_fit_protocol, validate_inputs as validate_development
from .check_frozen_fit import FIT_RUN_ID, FREEZE_SHA256, load_freeze, validate_fit
from .cross_split_audit import project_metadata
from .judge import JudgeInput
from .prepare_pilot import DATASET, DATASET_REVISION, file_sha256, group_hash
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .source_audit import INPUT_ID, NATIVE_FILES, NATIVE_REVISION, context_from_native, indicators, verified_lines
from .storage import RunConflict, atomic_json, exclusive_run

RUN_ID = 'ragtruth-test-input-manifest-v1'
TEST_FILE = 'data/test-00000-of-00001.parquet'
TEST_SHA256 = '2fc4fb703ea47ee0d4ab6110b86312f94fdf0bda157bc6ee67c7e61fb90d3bbd'
CROSS_SHA256 = '91f87c50a6c1e42d584b33d925ee62d41964093f83f212980d29587592ac1823'
TEST_ROWS = 2700
# Reviewed before judge TEST inference. Preserve processed input bytes exactly.
KNOWN_CONTEXT_DIFFERENCE = {
    'source_id':'14347', 'sample_ids':[str(i) for i in range(12180,12186)],
    'processed_context_sha256':'454a12c1b962446aa5194fb1549e96218279f78e7c6daff1d06293f480d6d69b',
    'native_formatted_context_sha256':'39f10d166699aa009df7922104b8ab5cfc2d23d2f01c3504f1906bcaf096ae5f',
    'native_extra_ASCII_space_offset':1522,
}


def context_relation(rid, sid, processed, native):
    if processed == native: return 'exact'
    known = KNOWN_CONTEXT_DIFFERENCE; offset = known['native_extra_ASCII_space_offset']
    if (sid == known['source_id'] and rid in known['sample_ids']
            and content_hash(processed) == known['processed_context_sha256']
            and content_hash(native) == known['native_formatted_context_sha256']
            and native[offset:offset+1] == ' ' and native[:offset]+native[offset+1:] == processed
            and group_hash(processed) == group_hash(native)):
        return 'reviewed_single_ASCII_space_difference'
    raise RunConflict('unreviewed native/processed context mismatch for '+rid)


def read_test(path):
    if file_sha256(path) != TEST_SHA256:
        raise RunConflict('expected exact pinned processed TEST parquet')
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    if len(rows) != TEST_ROWS:
        raise RunConflict('expected all 2700 TEST rows')
    return rows


def component_map(sources):
    """Same native/context/evidence/business closure as the prior split audit."""
    parent = {sid:sid for sid in sources}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]; sid = parent[sid]
        return sid
    groups = defaultdict(set)
    for sid,source in sources.items():
        groups['context:'+group_hash(context_from_native(source))].add(sid)
        for key in indicators(source): groups[key].add(sid)
    for group in groups.values():
        ordered = sorted(group)
        for sid in ordered[1:]:
            a,b = find(ordered[0]),find(sid)
            if a != b: parent[max(a,b)] = min(a,b)
    members = defaultdict(list)
    for sid in sorted(sources): members[find(sid)].append(sid)
    return {sid:content_hash(ids) for ids in members.values() for sid in ids}


def build_manifest(rows, response_rows, sources, cross, development, *, revision, fit_hash):
    """Pure alignment core; CLI verifies all source-file and prior-report hashes."""
    responses = {}; metadata = []; seen = set()
    for raw in response_rows:
        rid = raw['id']
        if rid in seen:
            raise RunConflict('duplicate native response ID')
        seen.add(rid)
        metadata.append({k:raw[k] for k in ('id','source_id','split')})
        if raw['split'] == 'test':
            responses[rid] = {k:raw[k] for k in ('id','source_id','response','model','quality')}
    ids = [r['id'] for r in rows]
    if (len(ids) != len(set(ids)) or set(ids) != set(responses)
            or any(type(rid) is not str or not rid for rid in ids)):
        raise RunConflict('processed/native TEST membership or IDs differ')
    if content_hash(sorted(metadata,key=lambda r:r['id'])) != cross['native_split_metadata_sha256']:
        raise RunConflict('native metadata differs from prior overlap audit')
    components = component_map(sources)
    train_ids = {r['id'] for r in metadata if r['split']=='train'}
    if train_ids != {r['sample_id'] for r in cross['train_mapping']} or train_ids & set(ids):
        raise RunConflict('prior TRAIN mapping or split membership differs')
    for old in cross['train_mapping']:
        if components[old['source_id']] != old['component_sha256']:
            raise RunConflict('group closure differs from prior cross-split audit')
    reservations = development['train_reservation']
    blocked = {r['component_sha256'] for r in reservations if r['partition'] in ('calibration','operating_threshold')}
    pilot = {r['component_sha256'] for r in reservations if r['linked_to_pilot']}
    model_inputs, offline = [], []
    for index,row in enumerate(rows):
        rid=row['id'];native=responses[rid];sid=native['source_id'];source=sources[sid]
        item=JudgeInput(answer=row['output'],context=row['context'])
        formatted=context_from_native(source)
        relation=context_relation(rid,sid,item.context,formatted)
        if (item.answer != native['response']
                or row['model'] != native['model'] or row['quality'] != native['quality']
                or row['task_type'] != source['task_type']):
            raise RunConflict('exact native answer/context or metadata mismatch for '+rid)
        counts=[row['hallucination_labels_processed'][k] for k in ('evident_conflict','baseless_info')]
        if any(type(v) is not int or v<0 for v in counts):
            raise RunConflict('invalid processed TEST label counts')
        if components[sid] in blocked | pilot:
            raise RunConflict('TEST component overlaps reserved fit or pilot components')
        model_inputs.append({'sample_id':rid,**asdict(item)})
        offline.append({'sample_id':rid,'test_index':index,'native_source_id':sid,
                        'component_sha256':components[sid],'input_sha256':content_hash(asdict(item)),
                        'answer_sha256':content_hash(item.answer),'context_sha256':content_hash(item.context),
                        'native_formatted_context_sha256':content_hash(formatted),'context_relation':relation,
                        'label':int(any(v>0 for v in counts)),
                        'metadata':{k:row[k] for k in ('task_type','model','quality')}})
    if not offline: raise RunConflict('empty TEST manifest')
    payload={'study_stage':'post_thesis','version':'ragtruth-test-input-manifest-v1','code_revision':revision,
             'freeze_sha256':FREEZE_SHA256,'fit_report_sha256':fit_hash,
             'dataset':{'repository':DATASET,'revision':DATASET_REVISION,'file':TEST_FILE,'sha256':TEST_SHA256,'split':'test'},
             'native_revision':NATIVE_REVISION,'native_files':NATIVE_FILES,'cross_split_report_sha256':CROSS_SHA256,
             'reviewed_context_difference':KNOWN_CONTEXT_DIFFERENCE,
             'model_inputs':model_inputs,'offline_rows':offline,
             'summary':{'test_rows':len(rows),'exact_native_answer_matches':len(rows),'exact_formatted_context_matches':sum(r['context_relation']=='exact' for r in offline),
                        'reviewed_whitespace_only_context_matches':sum(r['context_relation']!='exact' for r in offline),
                        'native_test_sources':len({r['native_source_id'] for r in offline}),
                        'test_components':len({r['component_sha256'] for r in offline}),
                        'labels_offline':dict(sorted(Counter(str(r['label']) for r in offline).items())),
                        'tasks_offline':dict(sorted(Counter(r['metadata']['task_type'] for r in offline).items())),
                        'generation_calls':0,'fitting_calls':0,'http_requests':0,'metrics_computed':False,
                        'processed_TEST_inputs_and_labels_read':True,'native_TEST_answers_used':True,
                        'native_annotation_labels_used':False,'strict_document_disjointness_established':False,
                        'pilot_or_fit_component_overlap':False,'HaluBench_read':False,
                        'token_audit_completed':False,'scoring_authorized':False}}
    return {'manifest_sha256':content_hash(payload),'manifest':payload}


def save_once(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RunConflict('existing alignment artifact differs; preserve it')
        return False
    atomic_json(path,value)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-parquet',type=Path,required=True)
    parser.add_argument('--native-dir',type=Path,default=run_directory(INPUT_ID))
    parser.add_argument('--baseline-dir',type=Path,help='Explicit directory holding original RAGTruth score caches')
    args=parser.parse_args();revision=code_revision();frozen=load_freeze()
    fit=json.loads((run_directory(FIT_RUN_ID)/'fit.json').read_text(encoding='utf-8'))
    validate_fit(fit,frozen)
    dev=json.loads((run_directory('ragtruth-train-development-reservation-v1')/'manifest.json').read_text(encoding='utf-8'))
    validate_development(dev,load_fit_protocol())
    prior=json.loads((run_directory('ragtruth-native-cross-split-audit-v1')/'report.json').read_text(encoding='utf-8'))
    if prior.get('report_sha256') != CROSS_SHA256 or content_hash(prior['report']) != CROSS_SHA256:
        raise RunConflict('prior cross-split audit checksum mismatch')
    responses=list(verified_lines(args.native_dir/'response.jsonl',NATIVE_FILES['response.jsonl']))
    metadata,sources=project_metadata(responses,verified_lines(args.native_dir/'source_info.jsonl',NATIVE_FILES['source_info.jsonl']))
    if Counter(r['split'] for r in metadata) != {'train':15090,'test':2700}:
        raise RunConflict('unexpected native split sizes')
    bundle=build_manifest(read_test(args.test_parquet),responses,sources,prior['report'],dev['manifest'],
                          revision=revision,fit_hash=frozen['source_fit_report_sha256'])
    from .baseline_inventory import inventory
    from research_paths import WORKSPACE
    baseline_dir=args.baseline_dir if args.baseline_dir is not None else WORKSPACE
    baseline=inventory(baseline_dir,bundle)
    directory=run_directory(RUN_ID)
    with exclusive_run(directory):
        created=save_once(directory/'manifest.json',bundle)
        invpath=directory/('baseline-inventory-'+baseline['inventory_sha256']+'.json')
        save_once(invpath,baseline)
    print('Created TEST manifest.' if created else 'Identical TEST manifest verified; no rewrite.')
    print(json.dumps(bundle['manifest']['summary'],indent=2))
    print('Manifest SHA256:',bundle['manifest_sha256'])
    print('Code revision:',revision)
    print('Private manifest:',directory/'manifest.json')
    print('Baseline inventory:',json.dumps(baseline['inventory']['baselines'],indent=2))
    print('Private inventory:',invpath)
    print('Post-thesis offline alignment only. No judge metrics, tokenization, refit or model calls.')
    print('Baseline provenance and token/execution manifests remain required before scoring.')
    return 0


if __name__=='__main__': raise SystemExit(main())
