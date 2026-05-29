"""Prompt templates for ICM on TruthfulQA.

The template mirrors the reference repo's truthfulQA setup:

  Question: {question}
  Claim: {answer}
  I think this claim is

The model is then queried for the next token, which should be "True" or "False".
"""

TRUTHFULQA_TEMPLATE = """Question: {question}
Claim: {choice}
I think this claim is"""


# Tokens we score. Llama tokenizes " True" and " False" with leading space.
LABEL_TOKENS = {
    1: " True",
    0: " False",
}


def format_example(question: str, choice: str) -> str:
    return TRUTHFULQA_TEMPLATE.format(question=question, choice=choice)


def format_fewshot_prompt(test_example: dict, demos: list[dict]) -> str:
    """Concatenate demos as fewshot examples, then the test example as a query.

    Each demo is a dict with 'question', 'choice', 'label' (0 or 1).
    The test_example is a dict with 'question', 'choice' (label not used here).
    """
    parts = []
    for d in demos:
        label_word = "True" if d["label"] == 1 else "False"
        parts.append(format_example(d["question"], d["choice"]) + " " + label_word)
    parts.append(format_example(test_example["question"], test_example["choice"]))
    return "\n\n".join(parts)
