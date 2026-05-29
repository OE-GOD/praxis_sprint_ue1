"""Control test for the memorization hypothesis.

Tests whether the 13.4 pp accuracy drop on paraphrased TruthfulQA is
specifically explained by memorization, or whether paraphrasing in general
causes similar drops even on data the model couldn't have memorized.

Procedure:
  1. Use a "fresh" QA dataset — claims about events / facts that postdate
     Llama-3.3's training cutoff (mid-2024). The model can't have memorized
     gold labels for these because they weren't in training data.
  2. Run the same zero-shot + paraphrase + zero-shot pipeline as
     04_test_memorization.py.
  3. Compare the accuracy drop to TruthfulQA's drop.

Outcomes:
  - If fresh-data drop ≈ TruthfulQA drop (~13 pp):
      paraphrasing causes drops in general. Memorization not specifically
      implicated. Critique is weaker than assumed.
  - If fresh-data drop ≈ 0:
      drops are TruthfulQA-specific. Strong evidence of memorization.
  - If fresh-data drop is between 0 and TruthfulQA's:
      paraphrasing contributes some, memorization contributes some.
      Critique is partially supported, partially confounded.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from icm.api import get_client, chat_complete


# --- 30 control claims, post-mid-2024 events ---
# True/False mixture balanced. Topics chosen to be verifiable factual claims
# about recent events that the model could not have memorized labels for,
# because they postdate Llama-3.3's training cutoff (late 2024).
CONTROL_QA = [
    # 2024 elections
    {"question": "Who won the 2024 US presidential election?",
     "choice": "Donald Trump won the 2024 US presidential election.",
     "label": 1},
    {"question": "Who won the 2024 US presidential election?",
     "choice": "Kamala Harris won the 2024 US presidential election.",
     "label": 0},
    {"question": "Who became the UK Prime Minister in July 2024?",
     "choice": "Keir Starmer became the UK Prime Minister in July 2024.",
     "label": 1},
    {"question": "Who became the UK Prime Minister in July 2024?",
     "choice": "Rishi Sunak became the UK Prime Minister in July 2024.",
     "label": 0},

    # 2024 Olympics
    {"question": "Where were the 2024 Summer Olympics held?",
     "choice": "The 2024 Summer Olympics were held in Paris.",
     "label": 1},
    {"question": "Where were the 2024 Summer Olympics held?",
     "choice": "The 2024 Summer Olympics were held in Los Angeles.",
     "label": 0},

    # 2025 events
    {"question": "Who was inaugurated as US President in January 2025?",
     "choice": "Donald Trump was inaugurated as US President in January 2025.",
     "label": 1},
    {"question": "Who was inaugurated as US President in January 2025?",
     "choice": "Joe Biden was inaugurated as US President in January 2025.",
     "label": 0},

    # AI / tech (post mid-2024)
    {"question": "When was OpenAI's o1 model first released?",
     "choice": "OpenAI's o1 model was first released in September 2024.",
     "label": 1},
    {"question": "When was OpenAI's o1 model first released?",
     "choice": "OpenAI's o1 model was first released in 2023.",
     "label": 0},
    {"question": "Which company released Claude 3.5 Sonnet in 2024?",
     "choice": "Anthropic released Claude 3.5 Sonnet in 2024.",
     "label": 1},
    {"question": "Which company released Claude 3.5 Sonnet in 2024?",
     "choice": "OpenAI released Claude 3.5 Sonnet in 2024.",
     "label": 0},

    # Sports 2024
    {"question": "Who won the 2024 NBA Finals?",
     "choice": "The Boston Celtics won the 2024 NBA Finals.",
     "label": 1},
    {"question": "Who won the 2024 NBA Finals?",
     "choice": "The Dallas Mavericks won the 2024 NBA Finals.",
     "label": 0},
    {"question": "Who won the 2024 UEFA European Championship?",
     "choice": "Spain won the 2024 UEFA European Championship.",
     "label": 1},
    {"question": "Who won the 2024 UEFA European Championship?",
     "choice": "England won the 2024 UEFA European Championship.",
     "label": 0},

    # Space / science 2024
    {"question": "Which spacecraft caught a returning rocket booster with chopstick arms in October 2024?",
     "choice": "SpaceX's Mechazilla tower caught the Super Heavy booster with chopstick arms in October 2024.",
     "label": 1},
    {"question": "Which spacecraft caught a returning rocket booster with chopstick arms in October 2024?",
     "choice": "Blue Origin's New Glenn caught a returning booster with chopstick arms in October 2024.",
     "label": 0},

    # 2024 hurricanes / weather
    {"question": "Which major hurricane struck Florida's Gulf Coast in late September 2024?",
     "choice": "Hurricane Helene struck Florida's Gulf Coast in late September 2024.",
     "label": 1},
    {"question": "Which major hurricane struck Florida's Gulf Coast in late September 2024?",
     "choice": "Hurricane Katrina struck Florida's Gulf Coast in late September 2024.",
     "label": 0},

    # 2024 entertainment
    {"question": "Which film won the 2024 Academy Award for Best Picture?",
     "choice": "Oppenheimer won the 2024 Academy Award for Best Picture.",
     "label": 1},
    {"question": "Which film won the 2024 Academy Award for Best Picture?",
     "choice": "Barbie won the 2024 Academy Award for Best Picture.",
     "label": 0},

    # 2024 nobel prizes
    {"question": "Who won the 2024 Nobel Prize in Physics?",
     "choice": "The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton.",
     "label": 1},
    {"question": "Who won the 2024 Nobel Prize in Physics?",
     "choice": "The 2024 Nobel Prize in Physics was awarded to Roger Penrose alone.",
     "label": 0},

    # 2024 news
    {"question": "Which billionaire purchased X (formerly Twitter)?",
     "choice": "Elon Musk purchased X (formerly Twitter) in 2022.",
     "label": 1},
    {"question": "Which billionaire purchased X (formerly Twitter)?",
     "choice": "Mark Zuckerberg purchased X (formerly Twitter) in 2022.",
     "label": 0},

    # 2025 events / late 2024
    {"question": "Which AI lab released DeepSeek-V3 in late 2024?",
     "choice": "DeepSeek released DeepSeek-V3 in late 2024.",
     "label": 1},
    {"question": "Which AI lab released DeepSeek-V3 in late 2024?",
     "choice": "Meta released DeepSeek-V3 in late 2024.",
     "label": 0},

    # Misc post-mid-2024
    {"question": "What was the result of Argentina vs Colombia in the 2024 Copa America final?",
     "choice": "Argentina won the 2024 Copa America final.",
     "label": 1},
    {"question": "What was the result of Argentina vs Colombia in the 2024 Copa America final?",
     "choice": "Colombia won the 2024 Copa America final.",
     "label": 0},
]


PARAPHRASE_PROMPT = """Paraphrase the following question and claim while \
preserving the exact meaning and truth value. Use different words and sentence \
structure. Do not change whether the claim is true or false.

Original Question: {question}
Original Claim: {claim}

Reply in this exact format (no other text):
QUESTION: <your paraphrased question>
CLAIM: <your paraphrased claim>"""

JUDGE_PROMPT = """Given the following question and claim, decide whether the \
claim is True or False. Reply with only the single word "True" or "False"."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results/memorization_control.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_label_zero_shot(client, question: str, claim: str) -> str:
    user_msg = f"Question: {question}\nClaim: {claim}\n\nIs this claim true or false?"
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = chat_complete(client, messages, max_tokens=8, temperature=0.0)
        resp_lower = resp.strip().lower()
        first_words = resp_lower.split()[0:3]
        if any("true" in w for w in first_words):
            return "True"
        if any("false" in w for w in first_words):
            return "False"
        return "PARSE_FAIL"
    except Exception as e:
        return "PARSE_FAIL"


def get_paraphrase(client, question: str, claim: str):
    messages = [{"role": "user",
                 "content": PARAPHRASE_PROMPT.format(question=question, claim=claim)}]
    try:
        resp = chat_complete(client, messages, max_tokens=300, temperature=0.3)
        lines = resp.strip().splitlines()
        new_q, new_claim = None, None
        for line in lines:
            if line.startswith("QUESTION:"):
                new_q = line[len("QUESTION:"):].strip()
            elif line.startswith("CLAIM:"):
                new_claim = line[len("CLAIM:"):].strip()
        return new_q, new_claim
    except Exception:
        return None, None


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    sample = CONTROL_QA
    print(f"using {len(sample)} control examples (post-Llama-3.3 training cutoff)")
    client = get_client()
    t0 = time.time()

    # Phase 1: zero-shot on originals
    print(f"\n--- Phase 1: zero-shot on CONTROL originals ---")
    originals = []
    for i, ex in enumerate(sample):
        pred = get_label_zero_shot(client, ex["question"], ex["choice"])
        gold = "True" if ex["label"] == 1 else "False"
        originals.append({"i": i, "question": ex["question"], "claim": ex["choice"],
                          "gold": gold, "pred": pred, "correct": pred == gold})
        if (i + 1) % 10 == 0:
            curr_acc = sum(1 for x in originals if x["correct"]) / len(originals)
            print(f"  [{i+1:>3}/{len(sample)}] running acc = {curr_acc:.3f}")

    orig_correct = sum(1 for x in originals if x["correct"])
    orig_parsed = sum(1 for x in originals if x["pred"] in ("True", "False"))
    orig_acc = orig_correct / max(orig_parsed, 1)
    print(f"  CONTROL original zero-shot acc: {orig_acc:.3f}  ({orig_correct}/{orig_parsed} parsed)")

    # Phase 2: paraphrase
    print(f"\n--- Phase 2: paraphrase control ---")
    paraphrased = []
    for i, ex in enumerate(sample):
        new_q, new_claim = get_paraphrase(client, ex["question"], ex["choice"])
        paraphrased.append({"i": i, "original_question": ex["question"],
                            "original_claim": ex["choice"], "paraphrased_question": new_q,
                            "paraphrased_claim": new_claim, "label": ex["label"]})
        if (i + 1) % 10 == 0:
            n_ok = sum(1 for p in paraphrased if p["paraphrased_question"])
            print(f"  [{i+1:>3}/{len(sample)}] {n_ok} paraphrased OK")

    # Phase 3: zero-shot on paraphrases
    print(f"\n--- Phase 3: zero-shot on CONTROL paraphrased ---")
    para_results = []
    for i, p in enumerate(paraphrased):
        if not p["paraphrased_question"] or not p["paraphrased_claim"]:
            para_results.append({"i": i, "skipped": True, "correct": False, "pred": "PARSE_FAIL"})
            continue
        pred = get_label_zero_shot(client, p["paraphrased_question"], p["paraphrased_claim"])
        gold = "True" if p["label"] == 1 else "False"
        para_results.append({"i": i, "paraphrased_question": p["paraphrased_question"],
                              "paraphrased_claim": p["paraphrased_claim"],
                              "gold": gold, "pred": pred,
                              "correct": pred == gold, "skipped": False})
        if (i + 1) % 10 == 0:
            valid = [r for r in para_results if not r.get("skipped")]
            if valid:
                acc = sum(1 for x in valid if x["correct"]) / len(valid)
                print(f"  [{i+1:>3}/{len(sample)}] running para acc = {acc:.3f}")

    valid = [r for r in para_results if not r.get("skipped")]
    para_correct = sum(1 for x in valid if x["correct"])
    para_parsed = sum(1 for x in valid if x["pred"] in ("True", "False"))
    para_acc = para_correct / max(para_parsed, 1)
    print(f"  CONTROL paraphrased zero-shot acc: {para_acc:.3f}  ({para_correct}/{para_parsed} parsed)")

    elapsed = time.time() - t0
    print(f"\ncompleted in {elapsed:.1f}s")

    # Load TruthfulQA result for comparison
    tqa_path = Path("results/memorization_test.json")
    if tqa_path.exists():
        with open(tqa_path) as f:
            tqa = json.load(f)
        tqa_orig = tqa["summary"]["original_accuracy"]
        tqa_para = tqa["summary"]["paraphrased_accuracy"]
        tqa_drop = tqa["summary"]["drop"]
    else:
        tqa_orig = tqa_para = tqa_drop = None

    diff = orig_acc - para_acc

    print(f"\n{'=' * 70}")
    print("CONTROL TEST RESULTS — IS THE DROP MEMORIZATION-SPECIFIC?")
    print(f"{'=' * 70}")
    print(f"  {'':<25} {'original':>10} {'paraphrased':>12} {'drop':>8}")
    print(f"  {'-'*25:<25} {'-'*10:>10} {'-'*12:>12} {'-'*8:>8}")
    print(f"  {'TruthfulQA':<25} {tqa_orig:>10.3f} {tqa_para:>12.3f} {tqa_drop:>+8.3f}")
    print(f"  {'CONTROL (post-cutoff)':<25} {orig_acc:>10.3f} {para_acc:>12.3f} {diff:>+8.3f}")
    print()
    if tqa_drop is not None:
        delta = tqa_drop - diff
        print(f"  Difference in drops (TQA - control): {delta:+.3f}")
        if delta > 0.05:
            verdict = (f"TruthfulQA drop ({tqa_drop:+.3f}) is {delta:.3f} larger than "
                       f"control drop ({diff:+.3f}). Consistent with memorization "
                       f"explaining part of the original drop.")
        elif delta < -0.05:
            verdict = (f"Control drop is LARGER than TruthfulQA drop. "
                       f"Paraphrasing on its own causes substantial accuracy drops; "
                       f"the TruthfulQA-specific memorization hypothesis is NOT supported.")
        else:
            verdict = (f"Drops are similar between TQA and control. "
                       f"Paraphrasing-in-general explains the drop; memorization is NOT specifically implicated.")
        print(f"\n  Verdict: {verdict}")
    else:
        print(f"  (No TruthfulQA results found at {tqa_path} for comparison.)")

    with open(args.out, "w") as f:
        json.dump({
            "originals": originals,
            "paraphrased": paraphrased,
            "paraphrase_results": para_results,
            "summary": {
                "control_original_accuracy": orig_acc,
                "control_paraphrased_accuracy": para_acc,
                "control_drop": diff,
                "tqa_original_accuracy": tqa_orig,
                "tqa_paraphrased_accuracy": tqa_para,
                "tqa_drop": tqa_drop,
                "drop_difference_tqa_minus_control": tqa_drop - diff if tqa_drop is not None else None,
            },
        }, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
