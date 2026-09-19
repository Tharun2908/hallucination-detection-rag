"""Prepare a private, TRAIN-only development manifest. Never calls a judge."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unicodedata

from .judge import JudgeInput, PROTOCOL_ID
from .prompts import DEVELOPMENT_PROMPT, content_hash
from .runner import Example, run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run

DATASET = "wandb/RAGTruth-processed"
DATASET_REVISION = "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
TRAIN_SHA256 = "c14ae31ff459c829edc860bda034ee2dbc0a11107b7511195a32bb4ab1ee8000"
TRAIN_ROWS = 15090
PREPARATION_VERSION = "ragtruth_train_pilot_v1"
PILOT_SIZE = 50
SEED = 20260919
GROUP_RULE = "context_NFKC_whitespace_collapsed_case_preserved_v1"


def group_hash(context):
    # Normalize only for grouping; the judge receives the original text.
    return content_hash(" ".join(unicodedata.normalize("NFKC", context).split()))


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_train(path):
    """The checksum forbids accidentally supplying TEST or a changed export."""
    if file_sha256(path) != TRAIN_SHA256:
        raise ValueError("TRAIN parquet checksum mismatch; do not substitute another export")
    import pyarrow.parquet as pq  # Optional CPU dependency, no serving imports.
    rows = pq.read_table(path).to_pylist()
    if len(rows) != TRAIN_ROWS:
        raise ValueError(f"expected {TRAIN_ROWS} TRAIN rows, got {len(rows)}")
    return rows


def _records(rows):
    records, inputs, seen = [], {}, set()
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"missing/invalid/duplicate id at TRAIN row {index}")
        seen.add(sample_id)
        item = JudgeInput(answer=row["output"], context=row["context"])
        if not item.context.strip():
            raise ValueError(f"empty context at TRAIN row {index}; review grouping before selection")
        labels = row["hallucination_labels_processed"]
        counts = [labels[key] for key in ("evident_conflict", "baseless_info")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(f"invalid label counts at TRAIN row {index}")
        metadata = {key: row[key] for key in ("task_type", "model", "quality")}
        if any(not isinstance(value, str) or not value.strip() for value in metadata.values()):
            raise ValueError(f"invalid offline metadata at TRAIN row {index}")
        inputs[sample_id] = asdict(item)
        records.append({"sample_id": sample_id, "train_index": index,
                        "group_sha256": group_hash(item.context),
                        "input_sha256": content_hash(asdict(item)),
                        "label": int(any(value > 0 for value in counts)),
                        "metadata": metadata})
    if not records:
        raise ValueError("empty TRAIN data")
    return records, inputs


def build_bundle(rows, *, revision, pilot_size=PILOT_SIZE, seed=SEED):
    """Pure selection core. The CLI additionally validates the pinned parquet."""
    if type(pilot_size) is not int or pilot_size < 1 or type(seed) is not int:
        raise ValueError("invalid pilot size or seed")
    records, inputs = _records(rows)
    groups = defaultdict(list)
    for record in records:
        groups[record["group_sha256"]].append(record)
    if len(groups) <= pilot_size:
        raise ValueError("insufficient groups to select a pilot and leave a development pool")

    def rank(kind, identifier):
        return content_hash([PREPARATION_VERSION, seed, kind, identifier]), identifier

    chosen_groups = sorted(groups, key=lambda key: rank("group", key))[:pilot_size]
    selected = [min(groups[key], key=lambda row: rank("row", row["sample_id"]))
                for key in chosen_groups]
    selected_ids = {row["sample_id"] for row in selected}
    chosen_set = set(chosen_groups)
    # Every sibling is blocked, even if its answer was not selected for scoring.
    reservation = []
    for row in records:
        partition = ("pilot" if row["sample_id"] in selected_ids else
                     "pilot_group_excluded" if row["group_sha256"] in chosen_set else
                     "threshold_candidate")
        reservation.append({key: row[key] for key in
                            ("sample_id", "train_index", "group_sha256", "input_sha256")}
                           | {"partition": partition})
    pilot = [{"sample_id": row["sample_id"], "input": inputs[row["sample_id"]]}
             for row in selected]
    # Labels/metadata have their own offline section; never serialize a whole
    # bundle or a label row into a request. pilot_examples is the strict boundary.
    payload = {
        "study_stage": "post_thesis", "protocol_id": PROTOCOL_ID,
        "preparation_version": PREPARATION_VERSION, "code_revision": revision,
        "dataset": {"repository": DATASET, "revision": DATASET_REVISION,
                    "split": "train", "file": TRAIN_FILE, "sha256": TRAIN_SHA256,
                    "rows": len(records), "ordered_records_sha256": content_hash(records)},
        "selection": {"seed": seed, "pilot_size": pilot_size,
                      "method": "hash-ranked groups, then one hash-ranked row per group",
                      "uses_labels_or_task_or_model": False, "group_rule": GROUP_RULE,
                      "group_limitation": "shared-context proxy; not native document IDs or near-duplicate detection"},
        "initial_prompt": {"version": DEVELOPMENT_PROMPT.version,
                           "sha256": DEVELOPMENT_PROMPT.sha256},
        "pilot_inputs": pilot, "pilot_labels_offline": selected,
        "train_reservation": reservation,
        "audit": {"train_groups": len(groups), "pilot_groups": len(chosen_groups),
                  "partition_counts": dict(sorted(Counter(row["partition"] for row in reservation).items())),
                  "pilot_label_counts": dict(sorted(Counter(str(row["label"]) for row in selected).items())),
                  "pilot_task_counts": dict(sorted(Counter(row["metadata"]["task_type"] for row in selected).items())),
                  "pilot_model_counts": dict(sorted(Counter(row["metadata"]["model"] for row in selected).items())),
                  "pilot_context_characters": [len(row["input"]["context"]) for row in pilot],
                  "pilot_answer_characters": [len(row["input"]["answer"]) for row in pilot],
                  "test_read": False, "judge_calls": 0,
                  "formatted_token_lengths_audited": False,
                  "native_source_overlap_audited": False,
                  "threshold_subset_selected": False},
    }
    return {"manifest_sha256": content_hash(payload), "manifest": payload}


def pilot_examples(bundle):
    """Validate saved identity/alignment and expose ONLY answer/context to judge."""
    manifest = bundle["manifest"]
    if bundle["manifest_sha256"] != content_hash(manifest):
        raise RunConflict("pilot manifest checksum mismatch")
    if (manifest["study_stage"] != "post_thesis" or manifest["protocol_id"] != PROTOCOL_ID
            or manifest["preparation_version"] != PREPARATION_VERSION
            or manifest["dataset"]["split"] != "train"):
        raise RunConflict("unexpected pilot contract or split")
    reservation = manifest["train_reservation"]
    ids = [row["sample_id"] for row in reservation]
    if len(set(ids)) != len(ids):
        raise RunConflict("duplicate reservation IDs")
    index = {row["sample_id"]: row for row in reservation}
    pilot_groups = {row["group_sha256"] for row in reservation if row["partition"] == "pilot"}
    candidates = [row for row in reservation if row["partition"] == "threshold_candidate"]
    if any(row["group_sha256"] in pilot_groups for row in candidates):
        raise RunConflict("pilot/threshold group overlap")
    result = []
    for row in manifest["pilot_inputs"]:
        if set(row) != {"sample_id", "input"} or set(row["input"]) != {"answer", "context"}:
            raise RunConflict("unexpected judge input field")
        item = JudgeInput(**row["input"])
        stored = index[row["sample_id"]]
        if (stored["partition"] != "pilot"
                or stored["input_sha256"] != content_hash(asdict(item))
                or stored["group_sha256"] != group_hash(item.context)):
            raise RunConflict("pilot input or group alignment mismatch")
        result.append(Example(row["sample_id"], item))
    returned_ids = [row.sample_id for row in result]
    if (len(returned_ids) != manifest["selection"]["pilot_size"]
            or len(set(returned_ids)) != len(returned_ids)
            or set(returned_ids) != {row["sample_id"] for row in reservation if row["partition"] == "pilot"}):
        raise RunConflict("pilot membership mismatch")
    return tuple(result)


def save_bundle(directory, bundle):
    pilot_examples(bundle)
    with exclusive_run(directory):
        target = directory / "manifest.json"
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            pilot_examples(previous)
            if previous != bundle:
                raise RunConflict("existing pilot differs; preserve it and review the change")
            return False
        atomic_json(target, bundle)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True,
                        help="Local pinned TRAIN parquet; no network or model requests")
    args = parser.parse_args()
    revision = code_revision()
    rows = read_train(args.train_parquet)
    bundle = build_bundle(rows, revision=revision)
    directory = run_directory("ragtruth-train-pilot-50-v1")
    created = save_bundle(directory, bundle)
    audit = bundle["manifest"]["audit"]
    print("Created pilot manifest." if created else "Reused identical pilot manifest.")
    print("Pilot examples: 50; judge calls: 0; TEST read: False")
    print("Partitions:", audit["partition_counts"])
    print("Pilot labels (offline):", audit["pilot_label_counts"])
    print("Pilot tasks (offline):", audit["pilot_task_counts"])
    print("Manifest SHA256:", bundle["manifest_sha256"])
    print("Grouping: shared-context proxy; native-source overlap audit pending")
    print("Token-length audit and compute-budget freeze pending; do not score yet.")
    print("Private manifest:", directory / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
