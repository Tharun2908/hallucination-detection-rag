"""Offline native TRAIN/TEST source-overlap check; no TEST answers/labels used."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .prepare_pilot import group_hash
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .source_audit import (NATIVE_FILES, NATIVE_REVISION, INPUT_ID, POLICY, VERSION as TRAIN_VERSION,
                           context_from_native, indicators, verified_lines)
from .storage import RunConflict, atomic_json, exclusive_run

VERSION = "native-cross-split-exact-audit-v1"
RUN_ID = "ragtruth-native-cross-split-audit-v1"
TRAIN_REPORT_SHA256 = "34c1943a2d32a09dabc3a856a963ae898fc43e97175e86549c4f037b6aadae14"
TRAIN_REPORT_REVISION = "80d2fa1baab81a50da674625e21fdf9a825c1f19"


def validate_train_report(bundle):
    report = bundle["report"]
    if bundle.get("report_sha256") != TRAIN_REPORT_SHA256 or content_hash(report) != TRAIN_REPORT_SHA256:
        raise RunConflict("expected pinned completed TRAIN source audit")
    if (report["version"] != TRAIN_VERSION or report["code_revision"] != TRAIN_REPORT_REVISION
            or report["native_files"] != NATIVE_FILES or report["native_revision"] != NATIVE_REVISION
            or report["policy"] != POLICY):
        raise RunConflict("TRAIN source audit provenance mismatch")
    return report


def project_metadata(response_rows, source_rows):
    """Read only response ID, source ID and split; discard answer/label fields."""
    metadata, seen = [], set()
    for raw in response_rows:
        row = {k: raw[k] for k in ("id", "source_id", "split")}
        if (any(type(row[k]) is not str or not row[k] for k in row)
                or row["split"] not in ("train", "test") or row["id"] in seen):
            raise RunConflict("invalid or duplicate native split metadata")
        seen.add(row["id"]); metadata.append(row)
    needed = {r["source_id"] for r in metadata}
    sources = {}
    for raw in source_rows:
        sid = raw["source_id"]
        if sid in sources or sid not in needed:
            raise RunConflict("duplicate or unreferenced native source")
        sources[sid] = {k: raw[k] for k in ("source_id", "task_type", "source", "source_info")}
    if set(sources) != needed:
        raise RunConflict("missing native source")
    return metadata, sources


def audit(train_report, metadata, sources):
    """Match TRAIN provenance, then compute label-blind exact overlap closure."""
    prior_rows = train_report["train_mapping"]
    old = {r["sample_id"]: r for r in prior_rows}
    native_train = {r["id"]: r for r in metadata if r["split"] == "train"}
    if len(old) != len(prior_rows) or set(old) != set(native_train):
        raise RunConflict("native TRAIN membership differs from prior audit")
    for rid, row in native_train.items():
        source = sources[row["source_id"]]
        if (old[rid]["source_id"] != row["source_id"]
                or old[rid]["context_group_sha256"] != group_hash(context_from_native(source))):
            raise RunConflict("native TRAIN source/context differs from prior audit")
    splits = defaultdict(set)
    for row in metadata: splits[row["source_id"]].add(row["split"])
    parent = {sid: sid for sid in sources}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]; sid = parent[sid]
        return sid
    def union(group):
        ordered = sorted(group)
        for sid in ordered[1:]:
            a, b = find(ordered[0]), find(sid)
            if a != b: parent[max(a,b)] = min(a,b)
    contexts, units = defaultdict(set), defaultdict(set)
    missing_business = []
    for sid, source in sources.items():
        contexts[group_hash(context_from_native(source))].add(sid)
        keys = indicators(source)
        if source["task_type"] == "Data2txt" and not keys: missing_business.append(sid)
        for key in keys: units[key].add(sid)
    for group in list(contexts.values()) + list(units.values()): union(group)
    components = defaultdict(list)
    for sid in sorted(sources): components[find(sid)].append(sid)
    hashes = {sid: content_hash(members) for members in components.values() for sid in members}
    test_sources = {sid for sid in sources if 'test' in splits[sid]}
    test_components = {hashes[sid] for sid in test_sources}
    pilot_sources = {r['source_id'] for r in prior_rows if r['original_partition']=='pilot'}
    pilot_components = {hashes[sid] for sid in pilot_sources}
    prior_excluded = set(train_report['proposed_exclusion_ids'])
    mapping, additionally_test_linked, additionally_pilot_linked = [], [], []
    for row in prior_rows:
        component = hashes[row['source_id']]
        test_linked, pilot_linked = component in test_components, component in pilot_components
        prior = row['sample_id'] in prior_excluded
        if test_linked and not prior: additionally_test_linked.append(row['sample_id'])
        if pilot_linked and not prior: additionally_pilot_linked.append(row['sample_id'])
        mapping.append({"sample_id":row['sample_id'],"source_id":row['source_id'],
                        "original_partition":row['original_partition'],"component_sha256":component,
                        "linked_to_native_test":test_linked,"linked_to_pilot":pilot_linked,
                        "proposed_excluded":prior or test_linked or pilot_linked})
    def cross_edges(groups):
        return [{"indicator":key,"source_ids":sorted(group)} for key,group in sorted(groups.items())
                if set().union(*(splits[sid] for sid in group))=={'train','test'}]
    cross_contexts, cross_units = cross_edges(contexts), cross_edges(units)
    both_components = [members for members in components.values()
                       if set().union(*(splits[sid] for sid in members))=={'train','test'}]
    counts = Counter(r['split'] for r in metadata)
    excluded = [r['sample_id'] for r in mapping if r['proposed_excluded']]
    return {"summary": {
            "native_train_responses":counts['train'],"native_test_responses_metadata_only":counts['test'],
            "native_train_sources":sum('train' in splits[sid] for sid in sources),
            "native_test_sources":len(test_sources),"native_sources_shared_between_splits":sum(len(s)>1 for s in splits.values()),
            "cross_split_full_context_groups":len(cross_contexts),
            "cross_split_exact_evidence_indicators":sum(r['indicator'].startswith('evidence:') for r in cross_units),
            "cross_split_business_indicators":sum(r['indicator'].startswith('business:') for r in cross_units),
            "cross_split_connected_components":len(both_components),"connected_components":len(components),
            "pilot_train_rows_linked_to_native_test":sum(r['original_partition']=='pilot' and r['linked_to_native_test'] for r in mapping),
            "train_rows_linked_to_native_test":sum(r['linked_to_native_test'] for r in mapping),
            "additional_test_linked_candidate_rows":len(additionally_test_linked),
            "additional_pilot_linked_candidate_rows":len(additionally_pilot_linked),
            "proposed_total_excluded_rows":len(excluded),"remaining_candidate_rows":len(mapping)-len(excluded),
            "sources_missing_business_identity":len(missing_business),"generation_calls":0,
            "native_TEST_answers_used":False,"native_annotation_labels_used":False,
            "native_TEST_source_content_used":True,"processed_TEST_read":False,
            "threshold_subset_selected":False,"calibration_fitted":False,
            "strict_document_disjointness_established":False},
            "additional_test_linked_candidate_ids":sorted(additionally_test_linked),
            "additional_pilot_linked_candidate_ids":sorted(additionally_pilot_linked),
            "proposed_exclusion_ids":sorted(excluded),"train_mapping":mapping,
            "cross_split_context_groups":cross_contexts,"cross_split_shared_indicators":cross_units,
            "cross_split_source_components":both_components,"missing_business_source_ids":sorted(missing_business),
            "native_split_metadata_sha256":content_hash(sorted(metadata,key=lambda r:r['id']))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=run_directory(INPUT_ID))
    args = parser.parse_args()
    revision = code_revision()
    prior = validate_train_report(json.loads((run_directory('ragtruth-train-source-audit-v1')/'report.json').read_text(encoding='utf-8')))
    metadata,sources = project_metadata(
        verified_lines(args.input_dir/'response.jsonl',NATIVE_FILES['response.jsonl']),
        verified_lines(args.input_dir/'source_info.jsonl',NATIVE_FILES['source_info.jsonl']))
    if Counter(r['split'] for r in metadata) != {'train':15090,'test':2700}:
        raise RunConflict('unexpected native split sizes')
    result = audit(prior,metadata,sources)
    report = {'study_stage':'post_thesis','version':VERSION,'code_revision':revision,
              'train_report_sha256':TRAIN_REPORT_SHA256,'native_revision':NATIVE_REVISION,
              'native_files':NATIVE_FILES,'exact_overlap_rules':{k:v for k,v in POLICY.items() if k!='not_covered'},
              'scope':'native_source_overlap_only; TEST answers, labels and performance not used',
              'not_covered':['fuzzy/partial overlap','paraphrase duplicates','canonical document IDs/URLs',
                             'missing business identity resolution','processed TEST identity verification',
                             'HaluBench overlap','baseline training exposure'],
              'native_release_contains_both_splits':True,'original_reservations_modified':False,**result}
    output = {'report_sha256':content_hash(report),'report':report}
    with exclusive_run(run_directory(RUN_ID)):
        path = run_directory(RUN_ID)/'report.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != output:
                raise RunConflict('existing cross-split audit differs; preserve it')
        else: atomic_json(path,output)
    print(json.dumps(result['summary'],indent=2))
    print('Report SHA256:',output['report_sha256'])
    print('Audit code revision:',revision)
    print('Private cross-split audit:',path)
    print('Post-thesis source-content overlap only. No TEST answers, labels, scores or model calls used.')
    print('Original reservations preserved. No final split or threshold selected.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
