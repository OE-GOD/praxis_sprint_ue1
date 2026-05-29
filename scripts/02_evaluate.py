"""Evaluate ICL accuracy on TruthfulQA test set across 4 conditions.

Conditions:
  - zero_shot:     no fewshot examples
  - random_labels: 8 random examples with randomized binary labels
  - icm_labels:    8 examples chosen from ICM's labeled set
  - gold_labels:   8 random training examples with ground-truth labels

Saves results to results/eval_results.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from icm.evaluate import (
    evaluate_condition,
    build_demos_zero_shot,
    build_demos_random,
    build_demos_icm,
    build_demos_gold,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="data/truthfulqa_train.json")
    p.add_argument("--test", type=str, default="data/truthfulqa_test.json")
    p.add_argument("--icm_labels", type=str, default="results/icm_labels.json")
    p.add_argument("--n_demos", type=int, default=8)
    p.add_argument("--max_test", type=int, default=None,
                   help="cap test set for faster/cheaper eval")
    p.add_argument("--out", type=str, default="results/eval_results.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.train) as f:
        train_examples = json.load(f)
    with open(args.test) as f:
        test_examples = json.load(f)
    with open(args.icm_labels) as f:
        icm_data = json.load(f)
    print(f"train: {len(train_examples)} test: {len(test_examples)}")

    # Build demos for each condition
    demos_zs = build_demos_zero_shot()
    demos_rand = build_demos_random(train_examples, args.n_demos, seed=args.seed)
    demos_icm = build_demos_icm(icm_data["demonstrations"], n=args.n_demos)
    demos_gold = build_demos_gold(train_examples, n=args.n_demos, seed=args.seed)

    print(f"\ndemos per condition:")
    print(f"  zero_shot: 0")
    print(f"  random:    {len(demos_rand)}")
    print(f"  icm:       {len(demos_icm)}")
    print(f"  gold:      {len(demos_gold)}")

    # Run each condition
    results = {}
    for name, demos in [
        ("zero_shot", demos_zs),
        ("random_labels", demos_rand),
        ("icm_labels", demos_icm),
        ("gold_labels", demos_gold),
    ]:
        print(f"\n--- evaluating {name} ---")
        r = evaluate_condition(test_examples, demos, name, max_test=args.max_test)
        print(f"  accuracy: {r['accuracy']:.3f} ({r['n_evaluated']}/{r['n_test']})  "
              f"parse_failures: {r['n_parse_failures']}")
        results[name] = r

    # Save
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'condition':<18} {'accuracy':>10} {'n':>6}")
    for name, r in results.items():
        print(f"  {name:<18} {r['accuracy']:>10.3f} {r['n_evaluated']:>6}")


if __name__ == "__main__":
    main()
