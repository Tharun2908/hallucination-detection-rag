"""Sequential, resumable post-thesis judging with bounded, explicit retries.

No API client or model is selected here. Raw request/response data stays in the
private artifact tree. A new run_id explicitly starts an independent run.
"""

import asyncio
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from research_paths import WORKSPACE
from .judge import (
    BackendResponse, JudgeBackend, JudgeConfig, JudgeInput, PROTOCOL_ID,
    TokenUsage, build_request, judge_once,
)
from .parse import JudgeParseError, parse_score
from .prompts import DEVELOPMENT_PROMPT, PromptSpec, canonical_json, content_hash
from .storage import Journal, RunConflict, atomic_json, exclusive_run


@dataclass(frozen=True)
class Example:
    sample_id: str
    item: JudgeInput

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if not isinstance(self.item, JudgeInput):
            raise TypeError("item must be JudgeInput; invalid rows must be fixed before running")


@dataclass(frozen=True)
class RetryPolicy:
    # Includes the initial call and interrupted attempts, across all resumes.
    max_attempts: int = 1
    retry_statuses: tuple[str, ...] = ("backend_error", "interrupted")
    delay_seconds: float = 0.0

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        allowed = {"backend_error", "interrupted", "invalid_output", "incomplete"}
        if (type(self.retry_statuses) is not tuple
                or any(type(s) is not str or s not in allowed for s in self.retry_statuses)
                or len(set(self.retry_statuses)) != len(self.retry_statuses)):
            raise ValueError("retry_statuses must be a unique tuple of retryable statuses")
        if (type(self.delay_seconds) not in (int, float)
                or not 0 <= self.delay_seconds <= 60):
            raise ValueError("delay_seconds must be finite and in [0,60]")


def run_directory(run_id: str, artifact_root=None) -> Path:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must be a safe 1-80 character name")
    root = WORKSPACE if artifact_root is None else Path(artifact_root).expanduser().resolve()
    base = (root / "post_thesis" / "llm_judge").resolve()
    directory = base / run_id
    # Refuse existing symlinks that could redirect writes outside the namespace.
    if directory.resolve().parent != base:
        raise ValueError("run directory escapes the post-thesis namespace")
    return directory


def _status(row):
    return row["result"]["status"] if row["state"] == "finished" else "interrupted"


def _validate_records(records, requests, policy):
    """Validate all cached data before any new call, not only the next example."""
    histories = {key: [] for key in requests}
    for row in records:
        key = row["request_key"]
        if key not in histories:
            raise RunConflict("journal contains an unknown request")
        history = histories[key]
        if row["ordinal"] != len(history) + 1 or row["ordinal"] > policy.max_attempts:
            raise RunConflict("attempt sequence violates the run policy")
        if history and _status(history[-1]) not in policy.retry_statuses:
            raise RunConflict("attempt follows a terminal result")
        if row["state"] == "finished":
            result = row["result"]
            try:
                if canonical_json(result["request"]) != canonical_json(asdict(requests[key])):
                    raise ValueError
                if result["study_stage"] != "post_thesis":
                    raise ValueError
                latency = result["latency_seconds"]
                if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
                    raise ValueError
                usage = TokenUsage(**result["usage"])
                status, score, error = result["status"], result["unsupported_probability"], result["error_code"]
                if status == "backend_error":
                    if result["response"] is not None or score is not None or not isinstance(error, str):
                        raise ValueError
                else:
                    data = dict(result["response"])
                    data["usage"] = TokenUsage(**data["usage"])
                    response = BackendResponse(**data)
                    if response.usage != usage:
                        raise ValueError
                    if response.outcome != "completed":
                        expected_status, expected_score, expected_error = response.outcome, None, response.outcome
                    else:
                        try:
                            expected_score = parse_score(response.text)
                            expected_status, expected_error = "ok", None
                        except JudgeParseError as exc:
                            expected_status, expected_score, expected_error = "invalid_output", None, exc.code
                    if (status, score, error) != (expected_status, expected_score, expected_error):
                        raise ValueError
                    if status == "ok" and type(score) not in (int, float):
                        raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                raise RunConflict("cached result violates the judge contract") from None
        history.append(row)
    return histories


def _report(examples, requests_by_id, histories, manifest_hash, policy,
            fresh_keys, new_attempts):
    predictions = []
    seen_successes = set()
    for example in examples:
        key = requests_by_id[example.sample_id].key
        history = histories[key]
        last = history[-1] if history else None
        status = _status(last) if last else "pending"
        successful = status == "ok"
        terminal = bool(history) and (successful or status not in policy.retry_statuses
                                     or len(history) >= policy.max_attempts)
        predictions.append({
            "sample_id": example.sample_id, "request_key": key,
            "status": status, "terminal": terminal,
            "unsupported_probability": last["result"]["unsupported_probability"] if successful else None,
            "attempts": len(history),
            "reused_success": successful and (key not in fresh_keys or key in seen_successes),
        })
        if successful:
            seen_successes.add(key)
    records = [row for history in histories.values() for row in history]
    known = {"input_tokens": 0, "output_tokens": 0}
    unknown = {"input_tokens": 0, "output_tokens": 0}
    latencies = []
    status_counts = {}
    for row in records:
        status = _status(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        result = row["result"] if row["state"] == "finished" else None
        if result:
            latencies.append(result["latency_seconds"])
        for field in known:
            value = result["usage"][field] if result else None
            if value is None:
                unknown[field] += 1
            else:
                known[field] += value
    scored = sum(p["status"] == "ok" for p in predictions)
    return {
        "study_stage": "post_thesis", "protocol_id": PROTOCOL_ID,
        "manifest_sha256": manifest_hash,
        "examples_total": len(examples), "examples_scored": scored,
        "examples_attempted": sum(bool(p["attempts"]) for p in predictions),
        "coverage": scored / len(examples),
        "examples_terminal_failure": sum(p["terminal"] and p["status"] != "ok" for p in predictions),
        "examples_pending": sum(not p["terminal"] for p in predictions),
        "unique_requests": len(histories), "attempts_total": len(records),
        "retry_attempts_total": sum(max(0, len(h) - 1) for h in histories.values()),
        "new_attempts_this_invocation": new_attempts,
        "attempt_status_counts": status_counts,
        "known_token_totals": known, "attempts_with_unknown_tokens": unknown,
        "completed_attempt_latency_seconds": latencies,
        "attempts_with_unknown_latency": len(records) - len(latencies),
        "cost": {"amount": None, "unit": None, "status": "not_configured"},
        "predictions": predictions,
    }


async def run_judge(examples, *, run_id: str, config: JudgeConfig,
                    backend: JudgeBackend, code_revision: str, dataset_revision: str,
                    policy: RetryPolicy = RetryPolicy(),
                    prompt: PromptSpec = DEVELOPMENT_PROMPT,
                    artifact_root=None, max_new_attempts: int | None = None):
    """Run/resume an exact ordered manifest. Returns an offline summary dict.

    max_new_attempts is an invocation cap, not a financial budget. Model calls are
    sequential. Changing frozen settings requires a new run_id. All valid cache
    entries in this run are reused; a new run is independent (e.g. stability tests).
    """
    examples = tuple(examples)
    if not examples or any(not isinstance(e, Example) for e in examples):
        raise ValueError("supply a nonempty sequence of Example objects")
    if len({e.sample_id for e in examples}) != len(examples):
        raise ValueError("duplicate sample_id; refusing ambiguous alignment")
    if not isinstance(policy, RetryPolicy):
        raise TypeError("policy must be RetryPolicy")
    for revision in (code_revision, dataset_revision):
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("explicit code and dataset revisions are required")
    if max_new_attempts is not None and (type(max_new_attempts) is not int or max_new_attempts < 0):
        raise ValueError("max_new_attempts must be a nonnegative integer or None")
    requests_by_id = {e.sample_id: build_request(e.item, config, prompt) for e in examples}
    requests = {r.key: r for r in requests_by_id.values()}
    manifest = {
        "study_stage": "post_thesis", "protocol_id": PROTOCOL_ID,
        "storage_version": 1, "run_id": run_id,
        "code_revision": code_revision, "dataset_revision": dataset_revision,
        "retry_policy": asdict(policy), "concurrency": 1,
        "examples": [{"sample_id": e.sample_id, "request_key": requests_by_id[e.sample_id].key}
                     for e in examples],
        "requests": {key: asdict(request) for key, request in requests.items()},
    }
    directory = run_directory(run_id, artifact_root)
    fresh_keys = set()
    new_attempts = 0
    with exclusive_run(directory):
        journal = Journal(directory / "journal.sqlite3", manifest)
        try:
            histories = _validate_records(journal.records(), requests, policy)
            journal.recover()
            atomic_json(directory / "manifest.json", manifest)
            for example in examples:
                key = requests_by_id[example.sample_id].key
                history = histories[key]
                while len(history) < policy.max_attempts:
                    if history and _status(history[-1]) not in policy.retry_statuses:
                        break
                    if max_new_attempts is not None and new_attempts >= max_new_attempts:
                        break
                    if history and policy.delay_seconds:
                        await asyncio.sleep(policy.delay_seconds)
                    attempt_id = journal.start(key, len(history) + 1)
                    new_attempts += 1
                    # Cancellation/programming errors leave a durable started
                    # record and propagate. The next invocation recovers it.
                    result = await judge_once(example.item, config=config, backend=backend, prompt=prompt)
                    journal.finish(attempt_id, asdict(result))
                    history.append({"state": "finished", "result": asdict(result)})
                    if result.status == "ok":
                        fresh_keys.add(key)
                # Includes pending and terminal failures; never silently drops rows.
            report = _report(examples, requests_by_id, histories, content_hash(manifest),
                             policy, fresh_keys, new_attempts)
            atomic_json(directory / "summary.json", report)
            return report
        finally:
            journal.close()
