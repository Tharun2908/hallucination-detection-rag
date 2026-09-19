"""Serving configuration, synthetic diagnostics, and resource arithmetic."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge import serve
from post_thesis.llm_judge.serve import REPO_ROOT, resource_totals, server_command, observe_gpu
from post_thesis.llm_judge.smoke import CASES, diagnostics
from post_thesis.llm_judge.vllm_backend import load_profile


class ServingTests(unittest.TestCase):
    def test_launcher_pins_weights_tokenizer_dtype_and_server_defaults(self):
        command = server_command()
        profile = load_profile()
        for flag, expected in (("--revision", profile["model_revision"]),
                               ("--tokenizer-revision", profile["tokenizer_revision"]),
                               ("--dtype", "bfloat16"), ("--generation-config", "vllm"),
                               ("--max-num-seqs", "1"), ("--host", "127.0.0.1")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertEqual(len(profile["model_revision"]), 40)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertNotIn("--quantization", command)

    def test_print_command_needs_no_gpu_or_httpx(self):
        result = subprocess.run([sys.executable, "-S", "-m", "post_thesis.llm_judge.serve", "--print-command"],
                                cwd=REPO_ROOT, text=True, capture_output=True, check=True)
        self.assertIn("Qwen/Qwen3-32B", result.stdout)
        self.assertIn("--revision", result.stdout)

    def test_resource_scope_arithmetic_and_unknown_cost(self):
        measured = resource_totals(1800, 2)
        self.assertEqual(measured["allocated_gpu_hours"], 1)
        self.assertIsNone(measured["estimated_cost"])
        priced = resource_totals(1800, 2, 3.5, "EUR")
        self.assertEqual(priced["estimated_cost"], 3.5)
        self.assertEqual(priced["cost_basis"], "declared_rate_times_measured_window")

    @unittest.skipUnless(os.name == "posix", "GPU supervisor targets Linux")
    def test_server_interrupt_stops_child_and_records_window(self):
        child = Mock(pid=12345)
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "session"
            with patch.object(sys, "argv", ["serve"]), \
                    patch.object(serve.importlib.metadata, "version", return_value=load_profile()["vllm_version"]), \
                    patch.object(serve, "code_revision", return_value="test-revision"), \
                    patch.object(serve, "observe_gpu", return_value={"uuid": "GPU-test", "name": "H200"}), \
                    patch.object(serve, "run_directory", return_value=directory), \
                    patch.object(serve.subprocess, "Popen", return_value=child), \
                    patch.object(serve.os, "killpg") as terminate:
                self.assertEqual(serve.main(), 130)
                terminate.assert_called_once_with(child.pid, serve.signal.SIGTERM)
            record = json.loads((directory / "resources.json").read_text())
        self.assertEqual(record["status"], "stopped")
        self.assertEqual(record["exit_code"], 130)
        self.assertGreaterEqual(record["resources"]["allocated_gpu_hours"], 0)
        self.assertIsNone(record["resources"]["estimated_cost"])

    def test_invalid_resource_values(self):
        for args in ((-1, 1), (float("nan"), 1), (1, True), (1, 1, 2, None),
                     (1, 1, -1, "EUR"), (1, 1, float("inf"), "EUR")):
            with self.assertRaises(ValueError):
                resource_totals(*args)

    def test_gpu_is_observed_without_loading_model(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0,
                   "NVIDIA H200, GPU-test, 143771, test-driver\n")) as run:
            info = observe_gpu(0)
        self.assertEqual(info["uuid"], "GPU-test")
        self.assertIn("--id=0", run.call_args.args[0])
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0,
                   "NVIDIA A100, GPU-test, 40960, test-driver\n")):
            with self.assertRaises(ValueError):
                observe_gpu(0)

    def test_synthetic_pairs_and_diagnostics_include_missing_scores(self):
        self.assertEqual(len(CASES), 6)
        predictions = []
        for index, case in enumerate(CASES):
            predictions.append({"sample_id": case.sample_id,
                                "unsupported_probability": 0.1 if index % 2 == 0 else 0.9})
        report = {"predictions": predictions}
        self.assertTrue(all(p["expected_direction_observed"] for p in diagnostics(report)))
        predictions[1]["unsupported_probability"] = None
        self.assertIsNone(diagnostics(report)[0]["expected_direction_observed"])
        for i in (0, 2, 4):
            self.assertEqual(CASES[i].item.answer, CASES[i + 1].item.answer)
            self.assertNotEqual(CASES[i].item.context, CASES[i + 1].item.context)


if __name__ == "__main__":
    unittest.main()
