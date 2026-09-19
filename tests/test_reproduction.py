"""Run the cached audit from outside the checkout; protect canonical artifacts."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def run_script(self, args, workspace, cwd):
        return subprocess.run(
            [sys.executable, *args], cwd=cwd,
            env={**os.environ, "RAG_WORKSPACE": str(workspace)},
            text=True, capture_output=True, timeout=60,
        )

    def test_committed_inputs_and_missing_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile, code in [("halubench", 0), ("fusion", 1)]:
                result = self.run_script(
                    [str(ROOT / "research_paths.py"), "--check", profile], tmp, tmp,
                )
                self.assertEqual(result.returncode, code, result.stderr)
                if code:
                    self.assertIn("MISSING", result.stdout)
                    self.assertIn("signal4_results_train_oof.json", result.stdout)

    def test_relative_override_is_anchored_to_repository(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = self.run_script([str(ROOT / "research_paths.py")], "my-artifacts", cwd)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(ROOT / "my-artifacts"), result.stdout)
            self.assertFalse((Path(cwd) / "my-artifacts").exists())

    def test_regenerated_cache_takes_precedence_without_changing_bundled_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "halubench_group_split.json"
            cache.write_text("{}")
            code = (
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "from research_paths import cached_input, workspace_path; "
                "print(cached_input('halubench_group_split.json')); "
                "print(workspace_path('halubench_per_example_scores.json'))"
            )
            result = self.run_script(["-c", code], tmp, tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                str(cache), str(Path(tmp) / "halubench_per_example_scores.json"),
            ])

    def test_halubench_audit_matches_canonical_results(self):
        canonical = ROOT / "results/cross_domain/halubench_groupfix/halubench_groupfix_thresholdfix_results.json"
        digest = hashlib.sha256(canonical.read_bytes()).digest()
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script(
                [str(ROOT / "cross_domain/halubench_groupfix_thresholdfix.py")],
                Path(tmp) / "outputs with spaces", tmp,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = Path(tmp) / "outputs with spaces/halubench_groupfix_thresholdfix_results.json"
            actual = json.loads(output.read_text())
            expected = json.loads(canonical.read_text())

            def compare(a, b):
                if isinstance(b, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in b:
                        compare(a[key], b[key])
                elif isinstance(b, list):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        compare(x, y)
                elif isinstance(b, float):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                else:
                    self.assertEqual(a, b)
            compare(actual, expected)
        self.assertEqual(hashlib.sha256(canonical.read_bytes()).digest(), digest)

    def test_active_code_has_no_cluster_absolute_paths(self):
        for folder in ["signals", "fusion", "evaluation", "robustness", "cross_domain", "efficiency"]:
            for path in (ROOT / folder).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        value = node.value
                        if "\n" not in value and " " not in value:
                            self.assertFalse(value == "/workspace" or value.startswith("/workspace/"), str(path))


if __name__ == "__main__":
    unittest.main()
