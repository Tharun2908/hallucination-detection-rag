"""Plot published frozen aggregates on CPU; no raw predictions or metric fitting.

Run: python -m post_thesis.llm_judge.plot_frozen_results
Requires matplotlib 3.10.8. Source JSON hashes ignore whitespace/line endings.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("RAGTruth", "ragtruth_frozen_evaluation_20260928.json",
     "c4527d7f82778e7eee81ac975f268c224d09c5ede5789b824d1fa1899f22e850"),
    ("HaluBench", "halubench_frozen_evaluation_20260929.json",
     "057fff2355c9974877dca8e6827ae6203880ab1c55db2b8ce80d5ddab2d12eaa"),
)
SYSTEMS = ("judge", "S4", "MiniCheck_7B", "S2_S4_metadata_free")
LABELS = ("Qwen3-32B judge", "S4", "MiniCheck-7B", "S2+S4 (no metadata)")
COLORS = ("#2463A6", "#B64D16")


def load_sources():
    records = []
    for benchmark, filename, expected_hash in SOURCES:
        path = ROOT / "results/post_thesis/llm_judge" / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != expected_hash:
            raise ValueError(f"Changed published result: {filename}")
        if record["study_stage"] != "post_thesis":
            raise ValueError("Expected post-thesis results")
        records.append((benchmark, record))
    return records


def save_svg(fig, path, metadata):
    fig.savefig(path, metadata=metadata)
    # Matplotlib adds trailing spaces in path data; normalize without changing geometry.
    text = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
    path.write_bytes(text.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/post_thesis/llm_judge")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_sources()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "post-thesis-frozen-results-v1",
                         "axes.spines.top": False, "axes.spines.right": False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None, "Creator": "post_thesis.llm_judge.plot_frozen_results"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "f1"), ("AUROC", "F1 at frozen operating rules")):
        for index, (benchmark, record) in enumerate(records):
            ys = [i + (-0.13 if index == 0 else 0.13) for i in range(4)]
            values = [record["point_metrics"][s][metric] for s in SYSTEMS]
            ax.scatter(values, ys, color=COLORS[index], marker=("o", "s")[index],
                       s=48, label=benchmark, zorder=3)
            for x, y in zip(values, ys):
                ax.annotate(f"{x:.3f}", (x, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=COLORS[index])
        ax.set(xlim=(0, 1), yticks=range(4), yticklabels=LABELS, title=title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Score (higher is better)")
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle("Frozen post-thesis results: ranking depends on the benchmark", fontsize=14)
    fig.text(0.5, 0.03, "RAGTruth: 2,700 responses  |  HaluBench: 8,000 responses\n"
             "Point estimates only; separate populations, unchanged decision rules.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save_svg(fig, args.output_dir / "frozen_results_overview.svg", metadata)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    ys_labels = [f"{b} / {label}" for b, _ in records for label in LABELS[1:]]
    for ax, metric, title in zip(axes, ("auroc", "f1"), ("AUROC difference", "F1 difference")):
        for index, (_, record) in enumerate(records):
            for row, system in enumerate(SYSTEMS[1:]):
                interval = record["paired_judge_minus_baseline"][system][metric]
                x, lo, hi = (interval[k] for k in ("point_difference", "lower", "upper"))
                if not lo <= x <= hi or interval["valid_draws"] != 2000:
                    raise ValueError("Unexpected frozen interval")
                y = index * 3 + row
                ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt=("o", "s")[index],
                            color=COLORS[index], capsize=4, markersize=6)
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.axhline(2.5, color="#CCCCCC", linewidth=0.8)
        ax.set(xlim=(-0.17, 0.45), yticks=range(6), yticklabels=ys_labels, title=title)
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Judge minus baseline; positive favors judge")
    axes[0].invert_yaxis()
    fig.suptitle("Paired uncertainty within each benchmark", fontsize=14)
    fig.text(0.5, 0.025, "2,000 paired group-bootstrap draws; nominal 95% percentile intervals, unadjusted.\n"
             "450 RAGTruth groups; 7,198 HaluBench components. Fixed models; no retraining uncertainty.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save_svg(fig, args.output_dir / "paired_frozen_differences.svg", metadata)
    plt.close(fig)
    print("Wrote two figures from published frozen aggregates; no model calls or fitting.")


if __name__ == "__main__":
    main()
