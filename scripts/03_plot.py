"""Generate the headline figure: bar chart of 4 ICL conditions on TruthfulQA.

Matches the format of the TruthfulQA subfigure in Figure 1 of the paper:
4 bars (zero-shot, random labels, ICM labels, gold labels), accuracy on y-axis.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=str, default="results/eval_results.json")
    p.add_argument("--out", type=str, default="results/figure_truthfulqa.png")
    return p.parse_args()


CONDITION_ORDER = ["zero_shot", "random_labels", "icm_labels", "gold_labels"]
CONDITION_LABELS = {
    "zero_shot": "Zero-shot",
    "random_labels": "Random labels",
    "icm_labels": "ICM (ours)",
    "gold_labels": "Gold labels",
}
CONDITION_COLORS = {
    "zero_shot": "#cccccc",
    "random_labels": "#888888",
    "icm_labels": "#1f77b4",   # blue, highlighted
    "gold_labels": "#2ca02c",  # green
}


def main():
    args = parse_args()
    with open(args.eval) as f:
        results = json.load(f)

    accs = [results[c]["accuracy"] for c in CONDITION_ORDER]
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)

    # Add accuracy labels above bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("TruthfulQA — in-context learning, Llama-3.1-405B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

    # Also save as PDF for the report
    pdf_path = args.out.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
