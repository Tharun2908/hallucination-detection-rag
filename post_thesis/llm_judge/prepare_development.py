"""Reserve disjoint TRAIN development components offline; no inference or fitting."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .cross_split_audit import validate_train_report
from .prepare_pilot import build_bundle, read_train, _records
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .source_audit import INPUT_ID, PREPARATION_REVISION
from .storage import RunConflict, atomic_json, exclusive_run

VERSION = "ragtruth-development-manifest-v1"
RUN_ID = "ragtruth-train-development-reservation-v1"
DESIGN_PATH = Path(__file__).parent / "configs" / "development_reservation_v1.json"
DESIGN_SHA256 = "4b8e9722370ef50b4b193a4109e3d890144e66c2cf002b5ff0834dbe42871574"
PARTITIONS = ("excluded", "calibration", "operating_threshold", "unallocated")


def load_design():
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if content_hash(design) != DESIGN_SHA256:
        raise RunConflict("reservation design changed; preserve v1 and register a revision")
    return design


def validate_reports(cross_bundle, train_bundle, design):
    train = validate_train_report(train_bundle)
    cross = cross_bundle["report"]
    expected = design["cross_split_report_sha256"]
    if (cross_bundle.get("report_sha256") != expected or content_hash(cross) != expected
            or cross["code_revision"] != design["cross_split_code_revision"]
            or cross["train_report_sha256"] != train_bundle["report_sha256"]):
        raise RunConflict("expected pinned completed cross-split audit")
    return cross, train


def allocate(mapping, design):
    """Label-blind core: never consumes processed data, labels or predictions."""
    groups, source_components, seen = defaultdict(list), {}, set()
    for row in mapping:
        rid, sid, component = row["sample_id"], row["source_id"], row["component_sha256"]
        if (type(rid) is not str or not re.fullmatch(r"0|[1-9][0-9]*", rid)
                or rid in seen or type(sid) is not str or not sid
                or type(component) is not str or not re.fullmatch(r"[0-9a-f]{64}", component)):
            raise RunConflict("invalid or duplicate mapping identity")
        seen.add(rid)
        if sid in source_components and source_components[sid] != component:
            raise RunConflict("one source assigned to multiple components")
        source_components[sid] = component
        for key in ("proposed_excluded", "linked_to_pilot", "linked_to_native_test"):
            if type(row[key]) is not bool:
                raise RunConflict("non-boolean component flag")
        if row["original_partition"] not in ("pilot", "pilot_group_excluded", "threshold_candidate"):
            raise RunConflict("unknown original partition")
        if (not row["proposed_excluded"] and
                (row["linked_to_pilot"] or row["linked_to_native_test"]
                 or row["original_partition"] != "threshold_candidate")):
            raise RunConflict("eligible row touches an excluded source or original reservation")
        groups[component].append(row)
    for members in groups.values():
        flags = {(r["proposed_excluded"], r["linked_to_pilot"], r["linked_to_native_test"])
                 for r in members}
        if len(flags) != 1:
            raise RunConflict("inconsistent flags within component")
    excluded = sum(r["proposed_excluded"] for r in mapping)
    if (len(mapping) != design["expected_train_rows"] or excluded != design["expected_excluded_rows"]
            or len(mapping) - excluded != design["expected_candidate_rows"]):
        raise RunConflict("reservation row counts differ from frozen design")
    ranking = design["ranking"]

    def rank(component):
        payload = f"{ranking['namespace']}\n{ranking['seed']}\n{component}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), component

    eligible = sorted((k for k, rows in groups.items() if not rows[0]["proposed_excluded"]), key=rank)
    n_cal = design["allocation"]["calibration_components"]
    n_threshold = design["allocation"]["operating_threshold_components"]
    if len(eligible) < n_cal + n_threshold:
        raise RunConflict("insufficient eligible components; no replacement or smaller allocation")
    assigned = {c: "excluded" for c in groups if groups[c][0]["proposed_excluded"]}
    ranked = []
    for i, component in enumerate(eligible):
        partition = ("calibration" if i < n_cal else
                     "operating_threshold" if i < n_cal + n_threshold else "unallocated")
        assigned[component] = partition
        ranked.append({"rank": i, "component_sha256": component,
                       "ranking_sha256": rank(component)[0], "partition": partition})
    reservations = []
    for row in sorted(mapping, key=lambda r: int(r["sample_id"])):
        reasons = []
        if row["original_partition"] != "threshold_candidate":
            reasons.append("original_" + row["original_partition"])
        if row["linked_to_pilot"]:
            reasons.append("pilot_component")
        if row["linked_to_native_test"]:
            reasons.append("native_test_component")
        if row["proposed_excluded"] and not reasons:
            raise RunConflict("excluded row has no recorded reason")
        reservations.append({**row, "partition": assigned[row["component_sha256"]],
                             "exclusion_reasons": reasons})
    return reservations, ranked


def build_manifest(rows, pilot, cross_bundle, train_bundle, *, revision):
    design = load_design()
    cross, train = validate_reports(cross_bundle, train_bundle, design)
    # Select before attaching labels or metadata. Historical identity checks below
    # read TRAIN labels only to verify the original immutable manifest.
    reservations, ranking = allocate(cross["train_mapping"], design)
    if (pilot.get("manifest_sha256") != design["pilot_manifest_sha256"]
            or content_hash(pilot["manifest"]) != design["pilot_manifest_sha256"]
            or build_bundle(rows, revision=PREPARATION_REVISION) != pilot):
        raise RunConflict("processed TRAIN or original pilot manifest differs")
    records, inputs = _records(rows)
    by_id = {r["sample_id"]: r for r in records}
    old = {r["sample_id"]: r for r in train["train_mapping"]}
    originals = {r["sample_id"]: r for r in pilot["manifest"]["train_reservation"]}
    ids = {r["sample_id"] for r in reservations}
    if ids != set(by_id) or ids != set(old) or ids != set(originals):
        raise RunConflict("incomplete TRAIN mapping coverage")
    excluded = {r["sample_id"] for r in reservations if r["partition"] == "excluded"}
    if (excluded != set(cross["proposed_exclusion_ids"])
            or not set(train["proposed_exclusion_ids"]).issubset(excluded)):
        raise RunConflict("historical exclusions not preserved")
    for row in reservations:
        rid = row["sample_id"]
        record = by_id[rid]
        if (row["source_id"] != old[rid]["source_id"]
                or row["original_partition"] != originals[rid]["partition"]
                or record["group_sha256"] != old[rid]["context_group_sha256"]
                or record["input_sha256"] != originals[rid]["input_sha256"]):
            raise RunConflict("TRAIN source/input/reservation identity mismatch")
        row["input_sha256"] = record["input_sha256"]
    missing = set(cross["missing_business_source_ids"])
    summaries, membership, model_inputs, offline = {}, {}, {}, {}
    for partition in PARTITIONS:
        members = [r for r in reservations if r["partition"] == partition]
        selected_ids = [r["sample_id"] for r in members]
        components = sorted({r["component_sha256"] for r in members})
        sources = {r["source_id"] for r in members}
        labels = Counter(str(by_id[rid]["label"]) for rid in selected_ids)
        summaries[partition] = {
            "rows": len(members), "components": len(components), "native_sources": len(sources),
            "labels_offline": dict(sorted(labels.items())),
            "tasks_offline": dict(sorted(Counter(by_id[rid]["metadata"]["task_type"] for rid in selected_ids).items())),
            "generators_offline": dict(sorted(Counter(by_id[rid]["metadata"]["model"] for rid in selected_ids).items())),
            "missing_business_identity_sources": len(sources & missing),
            "rows_with_missing_business_identity": sum(r["source_id"] in missing for r in members),
        }
        membership[partition] = {"sample_ids_sha256": content_hash(selected_ids),
                                 "component_ids_sha256": content_hash(components)}
        if partition in ("calibration", "operating_threshold"):
            model_inputs[partition] = [{"sample_id": rid, **inputs[rid]} for rid in selected_ids]
            offline[partition] = [{"sample_id": rid, "label": by_id[rid]["label"],
                                   "metadata": by_id[rid]["metadata"]} for rid in selected_ids]
    component_sets = {p: {r["component_sha256"] for r in reservations if r["partition"] == p}
                      for p in PARTITIONS}
    disjoint = all(not component_sets[a] & component_sets[b]
                   for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:])
    if not disjoint:
        raise RunConflict("components cross partitions")
    class_status = {p: set(summaries[p]["labels_offline"]) == {"0", "1"}
                    for p in ("calibration", "operating_threshold")}
    manifest = {
        "study_stage": "post_thesis", "version": VERSION, "code_revision": revision,
        "design_sha256": DESIGN_SHA256, "design": design,
        "input_hashes": {k: design[k] for k in ("cross_split_report_sha256", "train_report_sha256",
                                                 "pilot_manifest_sha256", "processed_train_sha256")},
        "native_revision": cross["native_revision"], "native_files": cross["native_files"],
        "partitions": summaries, "membership_hashes": membership,
        "component_ranking": ranking, "train_reservation": reservations,
        "model_inputs": model_inputs, "labels_offline": offline,
        "checks": {"every_TRAIN_row_once": True, "component_disjoint": disjoint,
                   "selected_components_touch_pilot_or_native_test": False,
                   "prior_exclusions_preserved": True, "both_classes_per_arm": class_status},
        "status": "reserved" if all(class_status.values()) else "reserved_missing_class_do_not_fit",
        "limitations": cross["not_covered"], "strict_document_disjointness_established": False,
        "generation_calls": 0, "http_requests": 0, "model_weights_loaded": False,
        "native_release_read_this_step": False, "processed_TEST_read": False,
        "original_reservations_modified": False, "calibration_fitted": False,
        "threshold_selected": False, "scoring_authorized": False,
    }
    return {"manifest_sha256": content_hash(manifest), "manifest": manifest}


def save_manifest(directory, bundle):
    if content_hash(bundle["manifest"]) != bundle["manifest_sha256"]:
        raise RunConflict("invalid development manifest checksum")
    with exclusive_run(directory):
        path = directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != bundle:
                raise RunConflict("existing development reservation differs; preserve it")
            return False
        atomic_json(path, bundle)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=run_directory(INPUT_ID))
    args = parser.parse_args()
    revision = code_revision()

    def read(run_id, filename):
        return json.loads((run_directory(run_id) / filename).read_text(encoding="utf-8"))

    bundle = build_manifest(
        read_train(args.input_dir / "train.parquet"),
        read("ragtruth-train-pilot-50-v1", "manifest.json"),
        read("ragtruth-native-cross-split-audit-v1", "report.json"),
        read("ragtruth-train-source-audit-v1", "report.json"), revision=revision)
    created = save_manifest(run_directory(RUN_ID), bundle)
    manifest = bundle["manifest"]
    print("Created development reservation." if created else "Verified identical development reservation; no rewrite.")
    print(json.dumps(manifest["partitions"], indent=2))
    print("Checks:", json.dumps(manifest["checks"], sort_keys=True))
    print("Status:", manifest["status"])
    print("Design SHA256:", DESIGN_SHA256)
    print("Manifest SHA256:", bundle["manifest_sha256"])
    print("Allocator code revision:", revision)
    print("Private manifest:", run_directory(RUN_ID) / "manifest.json")
    print("Post-thesis TRAIN reservation only. Generation calls: 0. HTTP requests: 0.")
    print("No calibration, threshold selection or scoring allowance. Exact-overlap components only; strict document disjointness unproven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
