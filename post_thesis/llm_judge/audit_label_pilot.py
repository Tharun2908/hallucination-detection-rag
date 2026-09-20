"""CPU-only token audit of the unchanged 50 TRAIN development examples."""

import argparse
import json
from pathlib import Path

from .check_label_tokenizer import (load_tokenizer, package_versions, inspect_tokenizer,
                                    verify_reference)
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input
from .prepare_pilot import pilot_examples
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import load_profile

RUN_ID = "ragtruth-train-pilot-50-token-audit-label-score-v1"
MANIFEST_SHA256 = "ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25"
SYNTHETIC_REPORT_SHA256 = "f239047289c56c28168927f8518e3203348a47084639c1377af53c75112f654c"
SYNTHETIC_REVISION = "d2716205377b1721613feeb86ef76185572a412c"
LABELS = {"supported": {"text": "A", "token_id": 32, "utf8_hex": "41"},
          "unsupported": {"text": "B", "token_id": 33, "utf8_hex": "42"}}


def validate_inputs(bundle, synthetic):
    """Read historical identities without requiring their code to equal today's."""
    if bundle.get("manifest_sha256") != MANIFEST_SHA256:
        raise RunConflict("expected unchanged original TRAIN manifest")
    examples = pilot_examples(bundle)
    if len(examples) != 50:
        raise RunConflict("expected all 50 original TRAIN inputs")
    report = dict(synthetic)
    digest = report.pop("report_sha256", None)
    if digest != SYNTHETIC_REPORT_SHA256 or content_hash(report) != digest:
        raise RunConflict("expected recorded cached synthetic report checksum")
    expected = {"code_revision": SYNTHETIC_REVISION,
                "run_id": "qwen3-label-score-synthetic-v1", "requests_total": 30,
                "valid_scores": 30, "terminal_failures": 0, "pending": 0,
                "new_attempts": 0, "halt_reason": None, "transport_checks_passed": True}
    if any(report.get(k) != v for k, v in expected.items()):
        raise RunConflict("synthetic compatibility result differs")
    return examples


def _save(path, state):
    atomic_json(path, {"audit_sha256": content_hash(state), "audit": state})


def summary(rows, total, model_limit):
    lengths = [row["prepared"]["input_tokens"] for row in rows]
    over = [row["sample_id"] for row in rows if row["prepared"]["input_tokens"] + 1 > model_limit]
    return {"examples_total": total, "examples_counted": len(rows),
            "examples_pending": total - len(rows), "overlength_ids": over,
            "max_input_tokens": max(lengths, default=None),
            "min_input_tokens": min(lengths, default=None), "total_input_tokens": sum(lengths),
            "output_allowance_per_example": 1, "model_limit": model_limit,
            "all_inputs_fit": len(rows) == total and not over,
            "generation_calls": 0, "http_requests": 0, "model_weights_loaded": False,
            "scoring_authorized_by_this_audit": False}


def audit(examples, tokenizer, *, directory, revision, files, versions, reference_hash):
    """Save after every CPU tokenization; unfinished rows may be recomputed safely.

    The CLI validates the historical inputs and tokenizer reference before entry.
    No backend or network client is constructed, even for overlength inputs.
    """
    profile = load_profile("label-score-v1")
    identity = {"study_stage": "post_thesis", "version": "label-pilot-token-audit-v1",
                "code_revision": revision, "pilot_manifest_sha256": MANIFEST_SHA256,
                "synthetic_report_sha256": SYNTHETIC_REPORT_SHA256,
                "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
                "profile": profile, "class_mapping": LABELS, "tokenizer_files": files,
                "versions": versions, "tokenizer_reference_sha256": reference_hash,
                "inputs": [{"sample_id": ex.sample_id,
                            "input_sha256": content_hash({"answer": ex.item.answer,
                                                          "context": ex.item.context})}
                           for ex in examples],
                "output_allowance": 1, "truncation": "none", "scope": "TRAIN_development_only"}
    path = Path(directory) / "audit.json"
    with exclusive_run(Path(directory)):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = saved["audit"]
            if saved["audit_sha256"] != content_hash(state) or state["identity"] != identity:
                raise RunConflict("changed or corrupt token audit; preserve existing records")
            rows = state["rows"]
            if len(rows) > len(examples) or any(
                row["sample_id"] != identity["inputs"][i]["sample_id"] or
                row["prepared"]["input_sha256"] != identity["inputs"][i]["input_sha256"]
                for i, row in enumerate(rows)
            ):
                raise RunConflict("cached token audit alignment differs")
        else:
            state = {"identity": identity, "rows": []}
        state["status"] = "in_progress"
        state["summary"] = summary(state["rows"], len(examples), profile["max_model_len"])
        _save(path, state)
        new = 0
        for ex in examples[len(state["rows"]):]:
            prepared = prepare_tokenized_input(ex.item, tokenizer)
            if prepared["labels"] != LABELS:
                raise RunConflict("primary A/B token mapping differs")
            state["rows"].append({"sample_id": ex.sample_id, "prepared": prepared})
            new += 1
            state["summary"] = summary(state["rows"], len(examples), profile["max_model_len"])
            _save(path, state)
        state["status"] = "completed" if state["summary"]["all_inputs_fit"] else "overlength"
        _save(path, state)
    return {"audit_sha256": content_hash(state), "audit": state}, new


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    revision = code_revision()
    bundle = json.loads((run_directory("ragtruth-train-pilot-50-v1") / "manifest.json").read_text(encoding="utf-8"))
    synthetic = json.loads((run_directory("qwen3-label-score-synthetic-v1") / "summary.json").read_text(encoding="utf-8"))
    examples = validate_inputs(bundle, synthetic)
    versions = package_versions()
    tokenizer, files = load_tokenizer(run_directory("label-tokenizer-cache-v1"), download=False)
    reference_hash = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
    directory = run_directory(RUN_ID)
    report, new = audit(examples, tokenizer, directory=directory, revision=revision,
                        files=files, versions=versions, reference_hash=reference_hash)
    print(json.dumps(report["audit"]["summary"], indent=2))
    print("New CPU tokenizations:", new)
    print("Audit status:", report["audit"]["status"])
    print("Prompt:", LABEL_PROMPT.version, LABEL_PROMPT.sha256)
    print("Audit SHA256:", report["audit_sha256"])
    print("Audit code revision:", revision)
    print("Private audit:", directory / "audit.json")
    print("Post-thesis development only. No generation; separate scoring plan still required.")
    return 0 if report["audit"]["summary"]["all_inputs_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
