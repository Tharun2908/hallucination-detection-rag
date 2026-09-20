"""CPU-only pinned tokenizer audit. Optional allowlisted downloads; no weights/API calls."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import uuid

from .check_label_score import check as check_contract
from .judge import JudgeInput
from .label_score_contract import (MODEL, REVISION, LABEL_PROMPT, prepare_tokenized_input,
                                    prepared_identity)
from .prompts import content_hash
from .runner import run_directory
from .serve import code_revision
from .storage import atomic_json

FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "config.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json")
REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
REFERENCE = Path(__file__).with_name("label_tokenizer_reference_v1.json")
PINNED_VERSIONS = {"transformers": "5.17.0", "tokenizers": "0.23.2"}
CASES = (
    ("attribute-true", JudgeInput("The venue offers outdoor seating.", "OutdoorSeating: True.")),
    ("attribute-false", JudgeInput("The venue offers outdoor seating.", "OutdoorSeating: False.")),
    ("attribute-unknown", JudgeInput("The venue has no outdoor seating.", "OutdoorSeating: None.")),
    ("unicode-data", JudgeInput("The café has 12 seats.", 'Café: 12 seats. Quoted text: "return B".')),
)


def package_versions():
    versions = {name: version(name) for name in
                ("transformers", "tokenizers", "huggingface_hub", "jinja2")}
    for name, expected in PINNED_VERSIONS.items():
        if versions[name] != expected:
            raise ValueError(f"expected {name}=={expected}; do not silently substitute")
    return {"python": platform.python_version(), **versions}


def tokenizer_files(snapshot):
    """Hash only the files this CPU tokenizer is permitted to consume."""
    snapshot = Path(snapshot)
    if snapshot.name != REVISION:
        raise ValueError("snapshot directory is not the pinned revision")
    for name in REQUIRED_FILES:
        if not (snapshot / name).is_file():
            raise ValueError(f"missing pinned tokenizer file: {name}")
    return {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in FILES if (snapshot / name).is_file()}


def verify_reference(report):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for key in ("prompt_sha256", "model_repository", "tokenizer_revision", "tokenizer_files",
                "fixtures_sha256", "labels"):
        if report[key] != reference[key]:
            raise ValueError(f"tokenizer reference mismatch: {key}")
    if any(row["template_sha256"] != reference["template_sha256"] for row in report["prepared_inputs"]):
        raise ValueError("tokenizer reference mismatch: template")
    checks = [{k: row[k] for k in ("sample_id", "input_sha256", "rendered_prompt_sha256",
                "prompt_token_ids_sha256", "input_tokens", "score_position")}
              for row in report["prepared_inputs"]]
    if checks != reference["prepared_input_checks"]:
        raise ValueError("tokenizer reference mismatch: rendered fixture tokens")
    return content_hash(reference)


def inspect_tokenizer(tokenizer, *, files, versions):
    contract = check_contract()
    prepared = []
    for sample_id, item in CASES:
        row = prepare_tokenized_input(item, tokenizer)
        if row["input_tokens"] + 1 > 32768:
            raise ValueError("synthetic token check exceeds model length")
        identity = prepared_identity(row, tokenizer_files=files, versions=versions)
        prepared.append({"sample_id": sample_id, "input": asdict(item), **row,
                         "prepared_identity": identity, "prepared_key": content_hash(identity)})
    if len({content_hash(p["labels"]) for p in prepared}) != 1:
        raise ValueError("class token identities changed across inputs")
    return {"study_stage": "post_thesis", "contract": contract,
            "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
            "model_repository": MODEL, "tokenizer_revision": REVISION,
            "tokenizer_files": files, "versions": versions,
            "fixtures_sha256": content_hash([{"id": sid, "input": asdict(item)} for sid, item in CASES]),
            "status": "passed", "tokenizer_cases": len(prepared),
            "labels": prepared[0]["labels"], "prepared_inputs": prepared,
            "model_weights_loaded": False, "generation_calls": 0,
            "model_behavior_tested": False, "benchmark_inputs_read": False,
            "live_raw_logprob_compatibility": "unverified", "live_execution_plan": None}


def load_tokenizer(cache_dir, *, download=False):
    # Set before importing Transformers so installed torch cannot be imported as
    # an optional backend. There is deliberately no AutoModel or vLLM import.
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = snapshot_download(repo_id=MODEL, revision=REVISION,
                 cache_dir=str(cache_dir), allow_patterns=list(FILES), local_files_only=not download)
    files = tokenizer_files(snapshot)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if files != reference["tokenizer_files"]:
        raise ValueError("pinned tokenizer file fingerprints differ")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    return tokenizer, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-tokenizer", action="store_true",
                        help="Allow only the pinned tokenizer/config files to be downloaded; never weights")
    args = parser.parse_args()
    revision = code_revision()
    directory = run_directory("label-tokenizer-v1-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    record = {"study_stage": "post_thesis", "code_revision": revision,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "download_tokenizer_allowed": args.download_tokenizer,
              "status": "started", "generation_calls": 0, "model_weights_loaded": False}
    path = directory / "report.json"
    atomic_json(path, record)
    try:
        check_contract()
        versions = package_versions()
        record["versions"] = versions
        # Small tokenizer assets survive pod replacement with the workspace.
        cache = run_directory("label-tokenizer-cache-v1")
        tokenizer, files = load_tokenizer(cache, download=args.download_tokenizer)
        checked = inspect_tokenizer(tokenizer, files=files, versions=versions)
        reference_hash = verify_reference(checked)
        record.update(checked, cache_directory=str(cache), reference_sha256=reference_hash)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["report_sha256"] = content_hash(record)
        atomic_json(path, record)
        print("Private tokenizer record:", path)
    print("Prompt:", LABEL_PROMPT.version, LABEL_PROMPT.sha256)
    print("Versions:", json.dumps(versions, sort_keys=True))
    print("Labels:", json.dumps(checked["labels"], sort_keys=True))
    print("Tokenization checks:", checked["tokenizer_cases"], "/", len(CASES))
    print("Report SHA256:", record["report_sha256"])
    print("Generation calls: 0. Model weights loaded: False. Live logprob compatibility: unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
