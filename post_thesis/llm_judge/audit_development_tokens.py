"""CPU-only formatted-length audit of both frozen TRAIN development arms."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .audit_label_pilot import LABELS, summary
from .check_label_tokenizer import load_tokenizer, package_versions, inspect_tokenizer, verify_reference
from .judge import JudgeInput
from .label_score_contract import LABEL_PROMPT, REVISION, prepare_tokenized_input
from .prompts import content_hash
from .runner import Example, run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import load_profile

RUN_ID = "ragtruth-train-development-token-audit-label-score-v1"
ARMS = ("calibration", "operating_threshold")
FIT_PATH = Path(__file__).parent / "configs" / "label_score_fit_v1.json"
FIT_SHA256 = "8902d4010ce17b0d3ac8b9a18fdfe70703cd72c4e62216bb41c6c7767a5703bc"
MANIFEST_SHA256 = "56b77ada74b638720586f93835ed801d8f090d7a04b9d1f1272a60e2677d7362"


def load_fit_protocol():
    protocol = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    if (content_hash(protocol) != FIT_SHA256
            or protocol["development_manifest_sha256"] != MANIFEST_SHA256
            or protocol["prompt_version"] != LABEL_PROMPT.version
            or protocol["prompt_sha256"] != LABEL_PROMPT.sha256
            or protocol["model_revision"] != REVISION):
        raise RunConflict("changed fitting protocol or prompt/model identity")
    return protocol


def validate_inputs(bundle, protocol):
    """Verify whole-manifest integrity, then project answer/context only.

    Manifest hashing includes offline labels. No label/metadata field is read
    into an Example, a rendered prompt or tokenizer messages.
    """
    manifest = bundle["manifest"]
    if (bundle.get("manifest_sha256") != MANIFEST_SHA256
            or content_hash(manifest) != MANIFEST_SHA256
            or manifest["code_revision"] != protocol["allocator_code_revision"]
            or manifest["design_sha256"] != protocol["reservation_design_sha256"]
            or manifest["status"] != "reserved"):
        raise RunConflict("expected pinned completed development reservation")
    if set(manifest["model_inputs"]) != set(ARMS):
        raise RunConflict("unexpected model-input partition")
    reservations = manifest["train_reservation"]
    by_id = {row["sample_id"]: row for row in reservations}
    if len(by_id) != len(reservations):
        raise RunConflict("duplicate reserved ID")
    examples, seen, components_seen = {}, set(), set()
    for arm, config_key in zip(ARMS, ("calibration", "threshold")):
        rows = manifest["model_inputs"][arm]
        ids = [row["sample_id"] for row in rows]
        if len(ids) != protocol[config_key]["required_rows"] or len(set(ids)) != len(ids) or seen & set(ids):
            raise RunConflict("missing, duplicate or cross-arm input IDs")
        if ids != sorted((r["sample_id"] for r in reservations if r["partition"] == arm), key=int):
            raise RunConflict("input IDs/order differ from reserved arm")
        expected = protocol["partition_membership_hashes"][arm]
        components = {by_id[rid]["component_sha256"] for rid in ids}
        if (content_hash(ids) != expected["sample_ids_sha256"]
                or content_hash(sorted(components)) != expected["component_ids_sha256"]
                or len(components) != protocol[config_key]["required_components"]
                or components_seen & components):
            raise RunConflict("component membership differs or crosses arms")
        arm_examples = []
        for row in rows:
            if set(row) != {"sample_id", "answer", "context"}:
                raise RunConflict("unexpected model-input fields; no metadata permitted")
            item = JudgeInput(answer=row["answer"], context=row["context"])
            reservation = by_id[row["sample_id"]]
            if (content_hash(asdict(item)) != reservation["input_sha256"]
                    or reservation["proposed_excluded"] or reservation["linked_to_pilot"]
                    or reservation["linked_to_native_test"]):
                raise RunConflict("input changed or touches an excluded component")
            arm_examples.append(Example(row["sample_id"], item))
        examples[arm] = arm_examples
        seen.update(ids)
        components_seen.update(components)
    return examples


def summarize(rows, examples, model_limit):
    result = summary(rows, sum(len(examples[a]) for a in ARMS), model_limit)
    result["partitions"] = {
        arm: summary([r for r in rows if r["partition"] == arm], len(examples[arm]), model_limit)
        for arm in ARMS}
    return result


def validate_row(row, expected):
    prepared = row["prepared"]
    if (row["partition"] != expected["partition"] or row["sample_id"] != expected["sample_id"]
            or prepared["input_sha256"] != expected["input_sha256"]
            or prepared["labels"] != LABELS
            or type(prepared["input_tokens"]) is not int or prepared["input_tokens"] < 1
            or type(prepared["score_position"]) is not int
            or prepared["score_position"] != prepared["input_tokens"]
            or "rendered_prompt" in prepared):
        raise RunConflict("cached/prepared token row alignment, mapping or length differs")


def audit(examples, tokenizer, *, directory, revision, files, versions, reference_hash):
    """Resumable CPU tokenization; never constructs a model or network client."""
    if set(examples) != set(ARMS) or any(not examples[a] for a in ARMS):
        raise RunConflict("both nonempty reserved arms are required")
    ordered = [(arm, example) for arm in ARMS for example in examples[arm]]
    if len({ex.sample_id for _, ex in ordered}) != len(ordered):
        raise RunConflict("duplicate audit input ID")
    profile = load_profile("label-score-v1")
    identity = {
        "study_stage": "post_thesis", "version": "development-label-token-audit-v1",
        "code_revision": revision, "development_manifest_sha256": MANIFEST_SHA256,
        "fit_protocol_sha256": FIT_SHA256, "prompt_version": LABEL_PROMPT.version,
        "prompt_sha256": LABEL_PROMPT.sha256, "profile": profile, "class_mapping": LABELS,
        "tokenizer_files": files, "versions": versions, "tokenizer_reference_sha256": reference_hash,
        "inputs": [{"partition": arm, "sample_id": ex.sample_id,
                    "input_sha256": content_hash(asdict(ex.item))} for arm, ex in ordered],
        "output_allowance": 1, "truncation": "none", "scope": "reserved_TRAIN_development_only",
        "prepared_storage": "hashes_and_lengths; rendered_text_omitted",
    }
    directory = Path(directory)
    path = directory / "audit.json"

    def save(state):
        atomic_json(path, {"audit_sha256": content_hash(state), "audit": state})

    with exclusive_run(directory):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = saved["audit"]
            if saved["audit_sha256"] != content_hash(state) or state["identity"] != identity:
                raise RunConflict("changed/corrupt development token audit; preserve existing records")
            if len(state["rows"]) > len(ordered):
                raise RunConflict("too many cached token rows")
            for i, row in enumerate(state["rows"]):
                validate_row(row, identity["inputs"][i])
            expected_summary = summarize(state["rows"], examples, profile["max_model_len"])
            if state["summary"] != expected_summary:
                raise RunConflict("cached summary disagrees with token rows")
            if len(state["rows"]) == len(ordered):
                expected_status = "completed" if expected_summary["all_inputs_fit"] else "overlength"
                if state["status"] == expected_status:
                    return saved, 0  # No tokenizer requests or report rewrite.
        else:
            state = {"identity": identity, "rows": []}
        state["status"] = "in_progress"
        state["summary"] = summarize(state["rows"], examples, profile["max_model_len"])
        save(state)
        new = 0
        for arm, example in ordered[len(state["rows"]):]:
            prepared = prepare_tokenized_input(example.item, tokenizer)
            prepared.pop("rendered_prompt")  # Exact hashes retained; original inputs remain in manifest.
            row = {"partition": arm, "sample_id": example.sample_id, "prepared": prepared}
            validate_row(row, identity["inputs"][len(state["rows"])])
            state["rows"].append(row)
            new += 1
            state["summary"] = summarize(state["rows"], examples, profile["max_model_len"])
            save(state)
        state["status"] = "completed" if state["summary"]["all_inputs_fit"] else "overlength"
        save(state)
        return {"audit_sha256": content_hash(state), "audit": state}, new


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    protocol = load_fit_protocol()
    bundle = json.loads((run_directory("ragtruth-train-development-reservation-v1") /
                         "manifest.json").read_text(encoding="utf-8"))
    examples = validate_inputs(bundle, protocol)
    versions = package_versions()
    tokenizer, files = load_tokenizer(run_directory("label-tokenizer-cache-v1"), download=False)
    reference_hash = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
    directory = run_directory(RUN_ID)
    result, new = audit(examples, tokenizer, directory=directory, revision=revision,
                        files=files, versions=versions, reference_hash=reference_hash)
    print(json.dumps(result["audit"]["summary"], indent=2))
    print("New CPU tokenizations:", new)
    print("Audit status:", result["audit"]["status"])
    print("Development manifest SHA256:", MANIFEST_SHA256)
    print("Fit protocol SHA256:", FIT_SHA256)
    print("Prompt:", LABEL_PROMPT.version, LABEL_PROMPT.sha256)
    print("Audit SHA256:", result["audit_sha256"])
    print("Audit code revision:", revision)
    print("Private audit:", directory / "audit.json")
    print("Post-thesis reserved TRAIN only. No generation, fitting or threshold selection.")
    print("No scoring allowance; record a separate bounded plan after this audit.")
    return 0 if result["audit"]["summary"]["all_inputs_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
