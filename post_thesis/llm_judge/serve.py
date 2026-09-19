"""Launch the pinned Linux GPU server and record its supervised lifetime."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid

from .runner import run_directory
from .storage import atomic_json
from .vllm_backend import load_profile

REPO_ROOT = Path(__file__).resolve().parents[2]


def code_revision():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=REPO_ROOT, text=True)
    if dirty:
        raise ValueError("commit tracked changes before recording a model run")
    return revision


def observe_gpu(device):
    completed = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=name,uuid,memory.total,driver_version",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True,
        timeout=10)
    rows = completed.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("expected one physical GPU")
    parts = [p.strip() for p in rows[0].split(",")]
    if len(parts) != 4 or "H200" not in parts[0]:
        raise ValueError("this profile requires one H200; create a new profile for other GPUs")
    return dict(zip(("name", "uuid", "memory_mib", "driver_version"), parts))


def server_command(port=8000, *, profile_name="default"):
    p = load_profile(profile_name)
    return ["vllm", "serve", p["model_repository"],
            "--revision", p["model_revision"], "--tokenizer-revision", p["tokenizer_revision"],
            "--served-model-name", p["served_model_name"],
            "--dtype", p["dtype"], "--tensor-parallel-size", str(p["tensor_parallel_size"]),
            "--max-model-len", str(p["max_model_len"]), "--max-num-seqs", str(p["concurrency"]),
            "--gpu-memory-utilization", str(p["gpu_memory_utilization"]),
            "--generation-config", p["generation_config"], "--seed", str(p["seed"]),
            "--no-enable-prefix-caching", "--host", "127.0.0.1", "--port", str(port)]


def resource_totals(seconds, gpu_count, hourly_rate=None, currency=None):
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            or type(gpu_count) is not int or gpu_count < 1):
        raise ValueError("invalid measured duration or GPU count")
    if ((hourly_rate is None) != (currency is None)
            or (hourly_rate is not None and (type(hourly_rate) not in (int, float)
                or not math.isfinite(hourly_rate) or hourly_rate < 0
                or not isinstance(currency, str) or not currency.strip()))):
        raise ValueError("supply both a nonnegative per-GPU hourly rate and currency, or neither")
    hours = seconds * gpu_count / 3600
    return {"wall_seconds": seconds, "allocated_gpu_hours": hours,
            "estimated_cost": None if hourly_rate is None else hours * hourly_rate,
            "currency": currency, "rate_per_gpu_hour": hourly_rate,
            "cost_basis": "declared_rate_times_measured_window" if hourly_rate is not None else "unknown"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", action="store_true", help="No GPU access or launch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", choices=("default", "evidence-v1"), default="default")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--currency")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.device < 0:
        parser.error("invalid port or GPU index")
    resource_totals(0, 1, args.hourly_rate, args.currency)
    command = server_command(args.port, profile_name=args.profile)
    if args.print_command:
        print(shlex.join(command))
        return 0
    if os.name != "posix":
        parser.error("run the GPU server on Linux; the HTTP client also supports Windows")
    p = load_profile(args.profile)
    if importlib.metadata.version("vllm") != p["vllm_version"]:
        raise ValueError("installed vLLM does not match the pinned profile")
    revision = code_revision()
    gpu = observe_gpu(args.device)
    directory = run_directory("server-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    record = {"study_stage": "post_thesis", "kind": "serving_session", "status": "started",
              "code_revision": revision, "profile": p, "gpu": gpu, "command": command,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "server_process_lifetime_including_startup_and_idle",
              "resources": None}
    atomic_json(directory / "resources.json", record)
    print(f"Serving session records: {directory}", flush=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    start = time.monotonic()
    process = None
    exit_code = 1
    old_handler = signal.getsignal(signal.SIGTERM)

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with (directory / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            exit_code = process.wait()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if process is not None:
            # Also terminate any workers left behind by an exited parent.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record.update(status="stopped", exit_code=exit_code,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      resources=resource_totals(time.monotonic() - start, p["gpu_count"],
                                                args.hourly_rate, args.currency))
        atomic_json(directory / "resources.json", record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
