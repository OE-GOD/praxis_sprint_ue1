"""Internal Coherence Maximization (ICM) — Algorithm 1 from Wen et al. 2025.

Implementation skips the consistency-fix step (per sprint instructions).
Search uses simulated annealing on a coherence energy:

    energy(D) = alpha * mean_predictability(D)

where mean_predictability is the average log-probability the base model
assigns to the labels in D, conditioning on the OTHER labels in D as
in-context demonstrations.

Algorithm:
  1. Initialize: pick num_seed examples, give them random labels.
  2. For K iterations:
     a. Pick a random unlabeled example (or labeled — for label flip proposals).
     b. Score both possible labels using the base model.
     c. Propose the higher-scoring label.
     d. If it differs from current, accept/reject via simulated annealing.
"""
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from icm.api import get_client, base_logprobs
from icm.prompts import format_fewshot_prompt, LABEL_TOKENS


@dataclass
class ICMConfig:
    alpha: float = 50.0
    num_seed: int = 8
    K: int = 500
    initial_T: float = 10.0
    final_T: float = 0.01
    decay: float = 0.99
    scheduler: str = "log"  # 'exp' or 'log'
    max_demos_in_context: int = 32  # cap fewshot context size for cost
    seed: int = 42


def get_temperature(iteration: int, cfg: ICMConfig) -> float:
    if cfg.scheduler == "exp":
        return max(cfg.final_T, cfg.initial_T * (cfg.decay ** iteration))
    elif cfg.scheduler == "log":
        return max(cfg.final_T, cfg.initial_T / (1 + 2 * np.log(1 + iteration)))
    raise ValueError(f"unknown scheduler: {cfg.scheduler}")


def score_label_for_example(
    client,
    example: dict,
    demos: list[dict],
    cfg: ICMConfig,
) -> tuple[float, float]:
    """Return (logprob_True, logprob_False) for the example given demos."""
    # Cap demos for cost / context-length reasons
    if len(demos) > cfg.max_demos_in_context:
        # Use a random subset, biased toward labeled examples
        demos = random.sample(demos, cfg.max_demos_in_context)

    prompt = format_fewshot_prompt(example, demos)
    target_tokens = [LABEL_TOKENS[1], LABEL_TOKENS[0]]
    logprobs = base_logprobs(client, prompt, target_tokens)
    return logprobs[LABEL_TOKENS[1]], logprobs[LABEL_TOKENS[0]]


def compute_energy(
    client,
    demonstrations: dict,
    cfg: ICMConfig,
    sample_size: Optional[int] = None,
) -> dict:
    """Compute the coherence energy of the current labeling.

    For each labeled example, score how well its label is predicted by the
    OTHER labeled examples. Returns:
      - 'train_prob': mean signed log-probability of the correct (assigned) label
      - 'train_accuracy': fraction of items where the predicted label matches the assigned

    For cost reasons, optionally sample only `sample_size` items rather than all.
    """
    labeled = {k: v for k, v in demonstrations.items() if v.get("label") is not None}
    items = list(labeled.values())
    if sample_size and sample_size < len(items):
        items = random.sample(items, sample_size)

    probs = []
    correct = 0
    for item in items:
        other_demos = [v for k, v in labeled.items() if k != item["uid"]]
        try:
            lp_true, lp_false = score_label_for_example(client, item, other_demos, cfg)
        except Exception as e:
            print(f"  score error for uid {item['uid']}: {e}")
            continue
        # signed score: + if assigned label is the more probable one
        if item["label"] == 1:
            probs.append(lp_true - lp_false)
            if lp_true > lp_false:
                correct += 1
        else:
            probs.append(lp_false - lp_true)
            if lp_false > lp_true:
                correct += 1

    return {
        "train_prob": float(np.mean(probs)) if probs else -1e6,
        "train_accuracy": correct / len(items) if items else 0.0,
        "n_evaluated": len(probs),
    }


def get_energy(metric: dict, cfg: ICMConfig) -> float:
    return cfg.alpha * metric["train_prob"]


def initialize_labels(examples: list[dict], cfg: ICMConfig) -> dict:
    """Pick num_seed examples, assign random binary labels balanced 50/50."""
    random.seed(cfg.seed)
    demonstrations = {}
    seed_labels = [1] * (cfg.num_seed // 2) + [0] * (cfg.num_seed // 2)
    random.shuffle(seed_labels)

    for idx, ex in enumerate(examples):
        item = deepcopy(ex)
        item["uid"] = idx
        item["vanilla_label"] = ex.get("label")  # preserve gold label for analysis
        if idx < cfg.num_seed:
            item["label"] = seed_labels[idx]
            item["type"] = "seed"
        else:
            item["label"] = None
            item["type"] = "unlabeled"
        demonstrations[idx] = item

    return demonstrations


def predict_label(
    client,
    example: dict,
    demonstrations: dict,
    cfg: ICMConfig,
) -> int:
    """Use the base model to predict a label for `example` given currently labeled demos."""
    labeled_demos = [
        v for k, v in demonstrations.items()
        if v.get("label") is not None and v["uid"] != example["uid"]
    ]
    lp_true, lp_false = score_label_for_example(client, example, labeled_demos, cfg)
    return 1 if lp_true > lp_false else 0


def run_icm(
    examples: list[dict],
    cfg: ICMConfig,
    log_path: Optional[str] = None,
) -> dict:
    """Main ICM loop. Returns the final demonstrations dict and metadata."""
    client = get_client()
    random.seed(cfg.seed)

    demonstrations = initialize_labels(examples, cfg)
    seed_gold_acc = np.mean(
        [demonstrations[i]["label"] == demonstrations[i]["vanilla_label"]
         for i in range(cfg.num_seed) if demonstrations[i]["vanilla_label"] is not None]
    )
    print(f"initialized {cfg.num_seed} seed labels (gold-agreement = {seed_gold_acc:.2f})")

    # Initial energy
    cur_metric = compute_energy(client, demonstrations, cfg, sample_size=cfg.num_seed)
    print(f"initial energy: train_prob={cur_metric['train_prob']:.3f} acc={cur_metric['train_accuracy']:.2f}")

    log_lines = []
    flip_cnt = 0

    for iter_ in tqdm(range(cfg.K), desc="ICM search"):
        # Pick an example to consider — bias toward unlabeled
        labeled_ids = [i for i, v in demonstrations.items() if v.get("label") is not None]
        unlabeled_ids = [i for i, v in demonstrations.items() if v.get("label") is None]

        # Weighted sample: 80% chance unlabeled, 20% relabel proposal
        if unlabeled_ids and random.random() < 0.8:
            example_id = random.choice(unlabeled_ids)
        elif labeled_ids:
            example_id = random.choice(labeled_ids)
        else:
            break

        example = demonstrations[example_id]
        try:
            proposed_label = predict_label(client, example, demonstrations, cfg)
        except Exception as e:
            print(f"  predict error: {e}")
            continue

        current_label = example.get("label")
        if current_label == proposed_label:
            continue  # nothing to change

        # Build the candidate state with the proposed label
        tmp = deepcopy(demonstrations)
        tmp[example_id]["label"] = proposed_label

        # Score it
        new_metric = compute_energy(client, tmp, cfg, sample_size=min(16, len(labeled_ids) + 1))

        # Simulated annealing acceptance
        cur_e = get_energy(cur_metric, cfg)
        new_e = get_energy(new_metric, cfg)
        T = get_temperature(flip_cnt, cfg)
        delta = new_e - cur_e

        if delta >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta / max(T, 1e-9))

        if accept:
            demonstrations = tmp
            cur_metric = new_metric
            flip_cnt += 1

        if iter_ % 50 == 0 or iter_ == cfg.K - 1:
            labeled_count = sum(1 for v in demonstrations.values() if v.get("label") is not None)
            gold_acc = np.mean([
                v["label"] == v["vanilla_label"]
                for v in demonstrations.values()
                if v.get("label") is not None and v["vanilla_label"] is not None
            ])
            print(f"  iter {iter_}: labeled={labeled_count} flips={flip_cnt} "
                  f"train_prob={cur_metric['train_prob']:.3f} "
                  f"acc_vs_gold={gold_acc:.3f} T={T:.4f}")
            if log_path:
                log_lines.append({
                    "iter": iter_,
                    "labeled_count": labeled_count,
                    "flips": flip_cnt,
                    "train_prob": cur_metric["train_prob"],
                    "gold_agreement": float(gold_acc),
                    "T": T,
                })

    return {
        "demonstrations": demonstrations,
        "final_metric": cur_metric,
        "n_flips": flip_cnt,
        "log": log_lines,
    }
