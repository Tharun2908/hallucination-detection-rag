"""Offline durability, cache, alignment, and retry tests; never loads a model."""

import asyncio
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

from post_thesis.llm_judge.judge import BackendError, BackendResponse, JudgeConfig, JudgeInput, TokenUsage
from post_thesis.llm_judge.prompts import DEVELOPMENT_PROMPT, content_hash
from post_thesis.llm_judge.runner import Example, RetryPolicy, run_directory, run_judge
from post_thesis.llm_judge.storage import RunConflict, RunLocked, exclusive_run

CONFIG = JudgeConfig("offline-fake", "fake-v1", "fake-adapter-v1")
EXAMPLES = (Example("a", JudgeInput("answer A", "context A")),
            Example("b", JudgeInput("answer B", "context B")))
GOOD = BackendResponse('{"unsupported_probability":0.25}', "returned-v1", TokenUsage(10, 3))


class FakeBackend:
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("unexpected model call")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    async def run_batch(self, backend, **changes):
        kwargs = dict(examples=EXAMPLES, run_id="offline-test", config=CONFIG,
                      backend=backend, code_revision="test-code-revision",
                      dataset_revision="synthetic-v1", artifact_root=self.root)
        kwargs.update(changes)
        return await run_judge(**kwargs)

    def database(self):
        return run_directory("offline-test", self.root) / "journal.sqlite3"

    def mutate(self, fn):
        with closing(sqlite3.connect(self.database())) as db:
            with db:
                fn(db)

    async def test_successful_resume_makes_no_new_calls(self):
        first = await self.run_batch(FakeBackend([GOOD, GOOD]))
        second = await self.run_batch(FakeBackend())
        self.assertEqual(first["attempts_total"], 2)
        self.assertEqual(second["new_attempts_this_invocation"], 0)
        self.assertEqual(second["coverage"], 1)
        self.assertTrue(all(p["reused_success"] for p in second["predictions"]))
        self.assertEqual(second["known_token_totals"], {"input_tokens": 20, "output_tokens": 6})
        self.assertEqual(len(second["completed_attempt_latency_seconds"]), 2)
        self.assertEqual(first["completed_attempt_latency_seconds"], second["completed_attempt_latency_seconds"])

    async def test_invocation_limit_saves_then_resumes(self):
        first = await self.run_batch(FakeBackend([GOOD]), max_new_attempts=1)
        self.assertEqual((first["examples_scored"], first["examples_pending"]), (1, 1))
        self.assertEqual(first["predictions"][1]["status"], "pending")
        self.assertIsNone(first["predictions"][1]["unsupported_probability"])
        second = await self.run_batch(FakeBackend([GOOD]))
        self.assertEqual(second["new_attempts_this_invocation"], 1)
        self.assertEqual(second["examples_scored"], 2)

    async def test_zero_limit_does_not_invoke_backend(self):
        result = await self.run_batch(FakeBackend(), max_new_attempts=0)
        self.assertEqual(result["attempts_total"], 0)
        self.assertEqual(result["examples_pending"], 2)

    async def test_duplicate_content_reuses_score_with_distinct_ids(self):
        result = await self.run_batch(FakeBackend([GOOD]), examples=(EXAMPLES[0], Example("other", EXAMPLES[0].item)))
        self.assertEqual(result["unique_requests"], 1)
        self.assertEqual(result["examples_scored"], 2)
        self.assertEqual(result["attempts_total"], 1)
        self.assertEqual([p["sample_id"] for p in result["predictions"]], ["a", "other"])
        self.assertEqual(result["known_token_totals"]["input_tokens"], 10)
        self.assertEqual([p["reused_success"] for p in result["predictions"]], [False, True])

    async def test_changed_manifest_rejected_before_backend_call(self):
        await self.run_batch(FakeBackend([GOOD]), max_new_attempts=1)
        variations = [
            {"examples": tuple(reversed(EXAMPLES))},
            {"examples": (replace(EXAMPLES[0], sample_id="changed"), EXAMPLES[1])},
            {"examples": (replace(EXAMPLES[0], item=JudgeInput("changed", "context A")), EXAMPLES[1])},
            {"config": replace(CONFIG, model="new-version")},
            {"config": replace(CONFIG, api_config_id="new-endpoint")},
            {"config": replace(CONFIG, temperature=0.1)},
            {"prompt": replace(DEVELOPMENT_PROMPT, version="next")},
            {"policy": RetryPolicy(2)},
            {"code_revision": "changed-code"},
            {"dataset_revision": "changed-data"},
        ]
        for changes in variations:
            with self.subTest(changes=changes), self.assertRaises(RunConflict):
                await self.run_batch(FakeBackend(), **changes)

    async def test_new_run_id_performs_independent_stability_run(self):
        await self.run_batch(FakeBackend([GOOD, GOOD]))
        other = await self.run_batch(FakeBackend([GOOD, GOOD]), run_id="independent-repeat")
        self.assertEqual(other["new_attempts_this_invocation"], 2)

    async def test_preflight_rejects_duplicate_ids_and_bad_items(self):
        for changes in ({"examples": (EXAMPLES[0], EXAMPLES[0])}, {"examples": ()},
                        {"examples": ({"label": 1},)}, {"max_new_attempts": True},
                        {"code_revision": ""}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                await self.run_batch(FakeBackend(), **changes)
        self.assertFalse(self.database().exists())
        with self.assertRaises(TypeError):
            Example("bad", None)

    async def test_retry_success_counts_every_attempt_and_unknown_usage(self):
        result = await self.run_batch(FakeBackend([BackendError("timeout"), GOOD, GOOD]), policy=RetryPolicy(2))
        self.assertEqual(result["attempts_total"], 3)
        self.assertEqual(result["attempt_status_counts"], {"backend_error": 1, "ok": 2})
        self.assertEqual(result["attempts_with_unknown_tokens"], {"input_tokens": 1, "output_tokens": 1})
        self.assertEqual(result["known_token_totals"]["input_tokens"], 20)
        self.assertIsNone(result["cost"]["amount"])

    async def test_retry_budget_persists_across_resumes(self):
        policy = RetryPolicy(2)
        first = await self.run_batch(FakeBackend([BackendError("timeout")]),
                                     policy=policy, max_new_attempts=1)
        self.assertFalse(first["predictions"][0]["terminal"])
        second = await self.run_batch(FakeBackend([BackendError("timeout"), GOOD]), policy=policy)
        self.assertEqual(second["examples_terminal_failure"], 1)
        self.assertEqual(second["predictions"][0]["attempts"], 2)
        third = await self.run_batch(FakeBackend(), policy=policy)
        self.assertEqual(third["new_attempts_this_invocation"], 0)
        self.assertIsNone(third["predictions"][0]["unsupported_probability"])

    async def test_refusal_never_retried_and_invalid_output_opt_in(self):
        refused = BackendResponse('{"unsupported_probability":0}', outcome="refused")
        first = await self.run_batch(FakeBackend([refused, BackendResponse("bad")]), policy=RetryPolicy(3))
        self.assertEqual(first["attempts_total"], 2)
        self.assertEqual(first["examples_terminal_failure"], 2)
        second = await self.run_batch(FakeBackend([BackendResponse("bad"), GOOD]),
                                     examples=EXAMPLES[:1], run_id="parse-retry",
                                     policy=RetryPolicy(2, ("invalid_output",)))
        self.assertEqual(second["attempts_total"], 2)
        self.assertEqual(second["coverage"], 1)

    async def test_started_attempt_is_durable_before_backend_runs(self):
        test = self

        class InspectBackend:
            async def complete(self, request):
                with closing(sqlite3.connect(test.database())) as db:
                    state = db.execute("SELECT state FROM attempts ORDER BY id DESC").fetchone()[0]
                test.assertEqual(state, "started")
                return GOOD

        await self.run_batch(InspectBackend())

    async def test_cancelled_attempt_is_recovered_and_counted(self):
        policy = RetryPolicy(2)
        with self.assertRaises(asyncio.CancelledError):
            await self.run_batch(FakeBackend([GOOD, asyncio.CancelledError()]), policy=policy)
        result = await self.run_batch(FakeBackend([GOOD]), policy=policy)
        self.assertEqual(result["attempts_total"], 3)
        self.assertEqual(result["new_attempts_this_invocation"], 1)
        self.assertEqual(result["attempt_status_counts"]["interrupted"], 1)
        self.assertEqual(result["attempts_with_unknown_latency"], 1)
        self.assertEqual(result["attempts_with_unknown_tokens"]["input_tokens"], 1)

    async def test_interrupted_last_attempt_stays_terminal(self):
        with self.assertRaises(RuntimeError):
            await self.run_batch(FakeBackend([RuntimeError("adapter bug")]))
        result = await self.run_batch(FakeBackend([GOOD]))
        self.assertEqual(result["predictions"][0]["status"], "interrupted")
        self.assertTrue(result["predictions"][0]["terminal"])
        self.assertEqual(result["new_attempts_this_invocation"], 1)

    async def test_corrupt_cache_fails_before_unscored_example(self):
        await self.run_batch(FakeBackend([GOOD]), max_new_attempts=1)
        self.mutate(lambda db: db.execute("UPDATE attempts SET result_hash='bad'"))
        with self.assertRaises(RunConflict):
            await self.run_batch(FakeBackend())

    async def test_cache_is_reparsed_even_with_matching_checksum(self):
        await self.run_batch(FakeBackend([GOOD]), max_new_attempts=1)

        def damage(db):
            result = json.loads(db.execute("SELECT result FROM attempts").fetchone()[0])
            result["unsupported_probability"] = 0.99
            db.execute("UPDATE attempts SET result=?,result_hash=?", (json.dumps(result), content_hash(result)))

        self.mutate(damage)
        with self.assertRaises(RunConflict):
            await self.run_batch(FakeBackend())

    async def test_unknown_request_key_is_not_silently_ignored(self):
        await self.run_batch(FakeBackend([GOOD]), max_new_attempts=1)
        self.mutate(lambda db: db.execute("UPDATE attempts SET request_key='wrong'"))
        with self.assertRaises(RunConflict):
            await self.run_batch(FakeBackend())

    async def test_lock_blocks_second_runner_before_call(self):
        with exclusive_run(run_directory("offline-test", self.root)):
            with self.assertRaises(RunLocked):
                await self.run_batch(FakeBackend())
        await self.run_batch(FakeBackend([GOOD, GOOD]))

    async def test_summary_is_regenerated_from_journal(self):
        first = await self.run_batch(FakeBackend([GOOD, GOOD]))
        path = run_directory("offline-test", self.root) / "summary.json"
        path.write_text("interrupted report write", encoding="utf-8")
        result = await self.run_batch(FakeBackend())
        self.assertEqual(json.loads(path.read_text())["attempts_total"], first["attempts_total"])
        self.assertEqual(result["new_attempts_this_invocation"], 0)

    async def test_ids_and_revision_metadata_do_not_reach_backend(self):
        backend = FakeBackend([GOOD])
        await self.run_batch(backend, examples=(Example("dataset-secret-id", EXAMPLES[0].item),))
        message = json.loads(backend.requests[0].messages[1].content)
        self.assertEqual(message, {"answer": "answer A", "context": "context A"})
        self.assertNotIn("dataset-secret-id", str(backend.requests))

    def test_paths_always_include_post_thesis_and_reject_traversal(self):
        path = run_directory("safe", self.root)
        # Windows temp roots can use an 8.3 alias (RUNNER~1). Compare canonical
        # paths because run_directory resolves the artifact root before use.
        expected = (self.root / "post_thesis" / "llm_judge" / "safe").resolve()
        self.assertEqual(path, expected)
        for run_id in ("..", "../escape", "/tmp/run", "a/b", "a\\b", "", "a" * 81):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                run_directory(run_id, self.root)

    def test_invalid_retry_policies(self):
        for kwargs in ({"max_attempts": True}, {"max_attempts": 0},
                       {"retry_statuses": ("refused",)}, {"retry_statuses": ["backend_error"]},
                       {"delay_seconds": float("nan")}, {"delay_seconds": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetryPolicy(**kwargs)

    async def test_real_process_death_releases_lock_and_recovers_journal(self):
        script = '''
import asyncio
from pathlib import Path
import sys
from post_thesis.llm_judge.runner import Example, RetryPolicy, run_judge
from post_thesis.llm_judge.judge import JudgeConfig, JudgeInput
class Block:
    async def complete(self, request):
        Path(sys.argv[1], "entered").write_text("ready")
        await asyncio.Event().wait()
asyncio.run(run_judge(
    (Example("a", JudgeInput("answer A", "context A")),),
    run_id="offline-test", config=JudgeConfig("offline-fake", "fake-v1", "fake-adapter-v1"),
    backend=Block(), code_revision="test-code-revision", dataset_revision="synthetic-v1",
    policy=RetryPolicy(2), artifact_root=sys.argv[1]))
'''
        process = subprocess.Popen([sys.executable, "-S", "-c", script, str(self.root)],
                                   cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not (self.root / "entered").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(process.communicate()[1].decode())
                await asyncio.sleep(0.02)
            self.assertTrue((self.root / "entered").exists(), "child never reached backend")
            process.kill()
            process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
        result = await self.run_batch(FakeBackend([GOOD]), examples=EXAMPLES[:1], policy=RetryPolicy(2))
        self.assertEqual(result["attempts_total"], 2)
        self.assertEqual(result["attempt_status_counts"], {"interrupted": 1, "ok": 1})
        self.assertEqual(result["coverage"], 1)


if __name__ == "__main__":
    unittest.main()
