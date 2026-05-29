"""Run the ICM algorithm on TruthfulQA training data.

Reads data/truthfulqa_train.json (256 examples), runs simulated annealing
to find a coherent labeling, saves the result to results/icm_labels.json.

Usage:
    python scripts/01_run_icm.py --alpha 50 --K 500 --num_seed 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from icm.algorithm import ICMConfig, run_icm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/truthfulqa_train.json")
    p.add_argument("--out", type=str, default="results/icm_labels.json")
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--num_seed", type=int, default=8)
    p.add_argument("--K", type=int, default=500)
    p.add_argument("--initial_T", type=float, default=10.0)
    p.add_argument("--final_T", type=float, default=0.01)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--scheduler", type=str, default="log", choices=["exp", "log"])
    p.add_argument("--max_demos", type=int, default=32,
                   help="Cap fewshot context size to control cost.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_path", type=str, default="results/icm_log.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    with open(args.data) as f:
        examples = json.load(f)
    print(f"loaded {len(examples)} training examples from {args.data}")

    cfg = ICMConfig(
        alpha=args.alpha,
        num_seed=args.num_seed,
        K=args.K,
        initial_T=args.initial_T,
        final_T=args.final_T,
        decay=args.decay,
        scheduler=args.scheduler,
        max_demos_in_context=args.max_demos,
        seed=args.seed,
    )
    print(f"ICM config: {cfg}")

    result = run_icm(examples, cfg, log_path=args.log_path)

    # Serialize and save
    serializable_demos = {}
    for uid, item in result["demonstrations"].items():
        serializable_demos[str(uid)] = {
            "uid": item["uid"],
            "question": item.get("question"),
            "choice": item.get("choice"),
            "consistency_id": item.get("consistency_id"),
            "label": item.get("label"),
            "vanilla_label": item.get("vanilla_label"),
            "type": item.get("type"),
        }

    out = {
        "config": cfg.__dict__,
        "n_flips": result["n_flips"],
        "final_metric": result["final_metric"],
        "demonstrations": serializable_demos,
        "log": result["log"],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}")

    # Quick summary
    labeled = [v for v in serializable_demos.values() if v["label"] is not None]
    gold_match = sum(
        1 for v in labeled
        if v["vanilla_label"] is not None and v["label"] == v["vanilla_label"]
    )
    print(f"\nlabeled {len(labeled)} examples; gold agreement = {gold_match/len(labeled):.3f}")
    print(f"final train_prob = {result['final_metric']['train_prob']:.3f}")
    print(f"total accept-flips = {result['n_flips']}")


if __name__ == "__main__":
    main()
