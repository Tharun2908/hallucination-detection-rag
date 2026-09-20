"""Offline TRAIN provenance and conservative exact-evidence overlap audit."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .audit_label_pilot import MANIFEST_SHA256
from .prepare_pilot import (TRAIN_SHA256, build_bundle,
                            group_hash, pilot_examples, read_train)
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run

VERSION = "native-train-source-audit-v1"
RUN_ID = "ragtruth-train-source-audit-v1"
INPUT_ID = "source-audit-inputs-v1"
NATIVE_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
NATIVE_FILES = {
    "response.jsonl": {"sha256": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
                       "git_blob_sha1": "f9ae5913fb9976684f3e13be3cd3465062db54cc"},
    "source_info.jsonl": {"sha256": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
                          "git_blob_sha1": "119277774009f110aa58596b93f9512f62594a89"},
}
PREPARATION_REVISION = "c44f4cf172811a086437e69c4f45b15fcda3ed16"
POLICY = {
    "native_id": "all responses to one native source_info record",
    "context": "original context_NFKC_whitespace_collapsed_case_preserved_v1",
    "evidence_units": "exact NFKC/whitespace/case-preserved QA passage or full Summary article; nonempty; no minimum length",
    "business_identity": "all four name/address/city/state present; NFKC/whitespace/casefold tuple",
    "closure": "connected components across all indicators; exclude pilot-linked components",
    "not_covered": ["fuzzy or partial text matches", "paraphrase duplicates", "canonical document URLs/IDs",
                    "TRAIN-TEST overlap", "HaluBench overlap", "baseline training exposure"],
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def verified_lines(path, expected):
    data = Path(path).read_bytes()
    if (hashlib.sha256(data).hexdigest() != expected["sha256"] or
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != expected["git_blob_sha1"]):
        raise RunConflict(f"native release checksum mismatch: {Path(path).name}")
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def project_native(response_rows, source_rows, train_ids):
    """Mixed release files are parsed; select TRAIN IDs before using payload fields.

    Native annotation labels are never accessed. Do not describe this as never
    reading TEST bytes: the native release files contain both splits.
    """
    responses, seen, ignored = {}, set(), 0
    for raw in response_rows:
        rid = raw["id"]
        if rid in seen:
            raise RunConflict("duplicate native response ID")
        seen.add(rid)
        if rid not in train_ids:
            ignored += 1
            continue
        if raw["split"] != "train":
            raise RunConflict("processed TRAIN ID maps to native non-TRAIN record")
        responses[rid] = {k: raw[k] for k in ("id", "source_id", "response", "model", "quality")}
    if set(responses) != set(train_ids):
        raise RunConflict("missing native TRAIN response IDs")
    needed = {r["source_id"] for r in responses.values()}
    sources, seen_sources, ignored_sources = {}, set(), 0
    for raw in source_rows:
        sid = raw["source_id"]
        if sid in seen_sources:
            raise RunConflict("duplicate native source ID")
        seen_sources.add(sid)
        if sid not in needed:
            ignored_sources += 1
            continue
        sources[sid] = {k: raw[k] for k in ("source_id", "task_type", "source", "source_info")}
    if set(sources) != needed:
        raise RunConflict("missing native source records")
    return responses, sources, {"native_response_records_ignored": ignored,
                                "native_source_records_ignored": ignored_sources}


def context_from_native(source):
    """Exact export formatting, including suffixes; no fuzzy match accepted."""
    task, info = source["task_type"], source["source_info"]
    if task == "QA":
        return re.sub(r"passage \d+:", "", info["passages"]).strip()
    if task == "Summary":
        return info + "\noutput:"
    if task == "Data2txt":
        return "\n" + str(info) + "\nOverview:"
    raise RunConflict("unknown native task")


def indicators(source):
    info, task = source["source_info"], source["task_type"]
    if task == "QA":
        parts = re.split(r"passage \d+:", info["passages"])
        if parts[0].strip() or len(parts) < 2:
            raise RunConflict("unexpected QA passage serialization")
        return {"evidence:" + content_hash(normalize(p)) for p in parts[1:] if normalize(p)}
    if task == "Summary":
        if not normalize(info):
            raise RunConflict("empty Summary source")
        return {"evidence:" + content_hash(normalize(info))}
    if task == "Data2txt":
        fields = [info.get(k) for k in ("name", "address", "city", "state")]
        if all(isinstance(v, str) and normalize(v) for v in fields):
            return {"business:" + content_hash([normalize(v).casefold() for v in fields])}
        return set()
    raise RunConflict("unknown native task")


def audit(rows, bundle, responses, sources):
    """Labels verify historical identity only; grouping never uses labels/scores."""
    if bundle.get("manifest_sha256") != MANIFEST_SHA256:
        raise RunConflict("expected original pilot manifest")
    pilot_examples(bundle)
    # Rebuild the deterministic selection to verify *all* reservations and inputs.
    if build_bundle(rows, revision=PREPARATION_REVISION) != bundle:
        raise RunConflict("TRAIN records differ from original pilot preparation")
    if set(responses) != {r["id"] for r in rows}:
        raise RunConflict("native/processed membership mismatch")
    manifest = bundle["manifest"]
    reservations = {r["sample_id"]: r for r in manifest["train_reservation"]}
    context_sources, source_contexts = defaultdict(set), defaultdict(set)
    mapping = []
    for row in rows:
        native = responses[row["id"]]
        source = sources[native["source_id"]]
        if (row["output"] != native["response"] or row["task_type"] != source["task_type"]
                or row["model"] != native["model"] or row["quality"] != native["quality"]
                or row["context"] != context_from_native(source)):
            raise RunConflict(f"native provenance mismatch for TRAIN ID {row['id']}")
        sid, context = native["source_id"], group_hash(row["context"])
        context_sources[context].add(sid); source_contexts[sid].add(context)
        mapping.append({"sample_id": row["id"], "source_id": sid, "task": row["task_type"],
                        "context_group_sha256": context, "original_partition": reservations[row["id"]]["partition"]})
    parent = {sid: sid for sid in sources}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]; sid = parent[sid]
        return sid
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for group in context_sources.values():
        first = min(group)
        for sid in group: union(first, sid)
    units, missing_business = defaultdict(set), []
    for sid, source in sources.items():
        keys = indicators(source)
        if source["task_type"] == "Data2txt" and not keys: missing_business.append(sid)
        for key in keys: units[key].add(sid)
    overlaps = []
    for key, group in sorted(units.items()):
        if len(group) > 1:
            overlaps.append({"indicator": key, "source_ids": sorted(group)})
            first = min(group)
            for sid in group: union(first, sid)
    components = defaultdict(list)
    for sid in sorted(sources): components[find(sid)].append(sid)
    component_id = {sid: content_hash(members) for members in components.values() for sid in members}
    pilot_sources = {r["source_id"] for r in mapping if r["original_partition"] == "pilot"}
    pilot_components = {component_id[sid] for sid in pilot_sources}
    additional, exclusions = [], []
    for row in mapping:
        row["component_sha256"] = component_id[row["source_id"]]
        linked = row["component_sha256"] in pilot_components
        # Preserve all original exclusions even if a future grouping differs.
        excluded = linked or row["original_partition"] != "threshold_candidate"
        row["proposed_excluded_from_future_development"] = excluded
        if excluded: exclusions.append(row["sample_id"])
        if linked and row["original_partition"] == "threshold_candidate": additional.append(row["sample_id"])
    return {"summary": {"processed_train_rows": len(rows), "native_response_matches": len(mapping),
            "exact_answer_matches": len(mapping), "exact_formatted_context_matches": len(mapping),
            "native_train_sources": len(sources), "original_context_groups": len(context_sources),
            "contexts_with_multiple_native_sources": sum(len(s)>1 for s in context_sources.values()),
            "native_sources_with_multiple_context_groups": sum(len(s)>1 for s in source_contexts.values()),
            "pilot_native_sources": len(pilot_sources), "pilot_context_groups": len({r['context_group_sha256'] for r in mapping if r['original_partition']=='pilot'}),
            "native_siblings_in_original_candidate_pool": sum(r['source_id'] in pilot_sources and r['original_partition']=='threshold_candidate' for r in mapping),
            "shared_exact_evidence_indicators": sum(o['indicator'].startswith('evidence:') for o in overlaps),
            "shared_business_indicators": sum(o['indicator'].startswith('business:') for o in overlaps),
            "sources_missing_business_identity": len(missing_business), "connected_components": len(components),
            "additional_pilot_linked_candidate_rows": len(additional),
            "proposed_total_excluded_rows": len(exclusions), "remaining_candidate_rows": len(rows)-len(exclusions),
            "generation_calls": 0, "threshold_subset_selected": False, "calibration_fitted": False,
            "strict_document_disjointness_established": False, "processed_TEST_read": False},
            "additional_exclusion_ids": sorted(additional), "proposed_exclusion_ids": sorted(exclusions),
            "missing_business_source_ids": sorted(missing_business), "shared_indicators": overlaps,
            "context_multi_source_groups": [{"context_group_sha256": key,"source_ids": sorted(sids)} for key,sids in sorted(context_sources.items()) if len(sids)>1],
            "train_mapping": mapping}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=run_directory(INPUT_ID))
    args = parser.parse_args()
    revision = code_revision()
    rows = read_train(args.input_dir / "train.parquet")
    bundle = json.loads((run_directory("ragtruth-train-pilot-50-v1") / "manifest.json").read_text(encoding="utf-8"))
    responses, sources, ignored = project_native(
        verified_lines(args.input_dir / "response.jsonl", NATIVE_FILES["response.jsonl"]),
        verified_lines(args.input_dir / "source_info.jsonl", NATIVE_FILES["source_info.jsonl"]),
        {r["id"] for r in rows})
    result = audit(rows, bundle, responses, sources)
    report = {"study_stage": "post_thesis", "version": VERSION, "code_revision": revision,
              "native_repository": "ParticleMedia/RAGTruth", "native_revision": NATIVE_REVISION,
              "native_files": NATIVE_FILES, "processed_train_sha256": TRAIN_SHA256,
              "pilot_manifest_sha256": MANIFEST_SHA256, "policy": POLICY,
              "native_release_contains_both_splits": True, "native_annotation_labels_used": False,
              "original_manifest_modified": False, "ignored_release_records": ignored, **result}
    digest = content_hash(report)
    directory = run_directory(RUN_ID)
    output = {"report_sha256": digest, "report": report}
    with exclusive_run(directory):
        path = directory / "report.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != output:
                raise RunConflict("existing source audit differs; preserve it for review")
        else:
            atomic_json(path, output)
    print(json.dumps(result["summary"], indent=2))
    print("Ignored native release records:", ignored)
    print("Report SHA256:", digest)
    print("Audit code revision:", revision)
    print("Private source audit:", path)
    print("Post-thesis offline TRAIN provenance only. Mixed native release parsed; no TEST labels used.")
    print("Original reservations preserved. No model calls or final split selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
