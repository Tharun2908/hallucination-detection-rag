"""Explicit synthetic-only GPU smoke run; never loads benchmark data."""

import argparse
import asyncio
from datetime import datetime, timezone
import os
import time
import uuid

from .judge import JudgeInput
from .runner import Example, RetryPolicy, run_directory, run_judge
from .serve import code_revision, resource_totals
from .storage import atomic_json
from .vllm_backend import VLLMBackend

# Manually specified support transitions; expectations never enter requests.
CASES = (
    Example("numeric-supported", JudgeInput("The trial enrolled 120 people.", "The trial enrolled 120 people.")),
    Example("numeric-contradiction", JudgeInput("The trial enrolled 120 people.", "The trial enrolled 80 people.")),
    Example("addition-supported", JudgeInput("The device is blue and weighs 2 kg.", "The blue device weighs 2 kg.")),
    Example("addition-unsupported", JudgeInput("The device is blue and weighs 2 kg.", "The device is blue. Its weight is not reported.")),
    Example("absence-supported", JudgeInput("The passage does not report a release date.", "The passage describes the device's blue casing.")),
    Example("absence-unsupported", JudgeInput("The passage does not report a release date.", "The device was released on 4 May 2020.")),
)
PAIRS = tuple((CASES[i].sample_id, CASES[i + 1].sample_id) for i in range(0, len(CASES), 2))


def diagnostics(report):
    scores = {p["sample_id"]: p["unsupported_probability"] for p in report["predictions"]}
    return [{"supported_id": supported, "unsupported_id": unsupported,
             "expected_direction_observed": None if scores[supported] is None or scores[unsupported] is None
             else scores[unsupported] > scores[supported]}
            for supported, unsupported in PAIRS]


async def smoke(args):
    revision = code_revision()
    directory = run_directory(args.run_id)
    # Each invocation has its own resource report; resuming cannot overwrite
    # the history of previous measured windows.
    resource_path = directory / ("client-window-" + uuid.uuid4().hex + ".json")
    record = {"study_stage": "post_thesis", "kind": "synthetic_smoke",
              "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "client_window_excludes_server_startup_and_other_idle_time",
              "gpu_count_basis": "declared_profile_assumes_exclusive_H200_server",
              "resources": None, "code_revision": revision}
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(resource_path, record)
    start = time.monotonic()
    try:
        async with VLLMBackend(args.base_url, api_key=os.environ.get("JUDGE_API_KEY")) as backend:
            record["server_preflight"] = await backend.preflight()
            atomic_json(resource_path, record)
            report = await asyncio.wait_for(run_judge(
                CASES, run_id=args.run_id, config=backend.config, backend=backend,
                code_revision=revision, dataset_revision="synthetic-support-pairs-v1",
                policy=RetryPolicy(max_attempts=1), max_new_attempts=6,
            ), timeout=300)
            record["diagnostics"] = diagnostics(report)
            record["status"] = "completed"
            print(f"Scored {report['examples_scored']}/6 synthetic examples; "
                  f"new attempts: {report['new_attempts_this_invocation']}")
            for pair in record["diagnostics"]:
                print(f"{pair['supported_id']} -> {pair['unsupported_id']}: "
                      f"expected direction observed = {pair['expected_direction_observed']}")
            return 0 if report["examples_scored"] == 6 else 1
    finally:
        if record["status"] == "started":
            record["status"] = "failed_or_interrupted"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["resources"] = resource_totals(time.monotonic() - start, 1)
        atomic_json(resource_path, record)
        print(f"Private run records: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", default="qwen3-synthetic-smoke-v1")
    args = parser.parse_args()
    return asyncio.run(smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
