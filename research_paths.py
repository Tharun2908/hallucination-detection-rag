"""Portable paths for active thesis experiments (no ML dependencies)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# Relative overrides are anchored to the checkout, never the caller's directory.
_configured = Path(os.environ.get("RAG_WORKSPACE", ".artifacts")).expanduser()
WORKSPACE = (_configured if _configured.is_absolute() else REPO_ROOT / _configured).resolve()

# These are the only per-example inputs bundled for the canonical HaluBench audit.
BUNDLED_INPUTS = {
    "halubench_group_split.json": "results/cross_domain/halubench_groupfix/halubench_group_split.json",
    "halubench_final_s2s4_scores.json": "results/cross_domain/halubench_final_s2s4_scores.json",
    "halubench_per_example_scores.json": "results/cross_domain/halubench_per_example_scores.json",
}

PROFILES = {
    "halubench": list(BUNDLED_INPUTS),
    "fusion": [
        "relevance_results_train_v2.json", "relevance_results_test_v2.json",
        "signal4_results_train_oof.json", "signal4_results_test.json",
    ],
    "cascade": [
        "relevance_results_train_v2.json", "relevance_results_test_v2.json",
        "signal4_results_train_oof.json", "signal4_results_test.json",
        "minicheck_results_test_7b.json",
    ],
}


def workspace_path(relative: str = "") -> str:
    """Generated files/checkpoints always stay in the chosen artifact directory."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return str(WORKSPACE / relative)


def cached_input(relative: str) -> str:
    """Prefer a regenerated cache; otherwise use the explicitly mapped committed input."""
    candidate = WORKSPACE / relative
    if candidate.exists():
        return str(candidate)
    if relative in BUNDLED_INPUTS:
        return str(REPO_ROOT / BUNDLED_INPUTS[relative])
    return str(candidate)


def repo_path(relative: str) -> str:
    return str(REPO_ROOT / relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=PROFILES, help="Check inputs without loading models")
    args = parser.parse_args()
    print(f"Repository: {REPO_ROOT}")
    print(f"Artifacts:  {WORKSPACE}")
    missing = []
    for name in PROFILES.get(args.check, []):
        path = Path(cached_input(name))
        present = path.is_file()
        print(f"{'OK     ' if present else 'MISSING'} {path}")
        if not present:
            missing.append(name)
    if missing:
        print("Regenerate or supply these score files; see docs/REPRODUCING.md.")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
