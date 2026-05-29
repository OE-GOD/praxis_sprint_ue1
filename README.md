# Praxis Sprint UE1 — ICM Replication

Reimplementation of Internal Coherence Maximization (ICM) from Wen et al.,
*"Unsupervised Elicitation of Language Models"* (arXiv:2506.10139).

Reproduces the TruthfulQA result using Llama-3.1-405B via Hyperbolic API.
In-context learning evaluation only (no SFT, per sprint spec).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your Hyperbolic key
export HYPERBOLIC_API_KEY=...

# Place the dataset (from the sprint link) at:
#   data/truthfulqa_train.json   (256 examples)
#   data/truthfulqa_test.json    (100 examples)
```

## Run

```bash
# Step 1: verify API works and returns logprobs
python scripts/00_verify_api.py

# Step 2: run ICM to search for labels (most expensive step)
python scripts/01_run_icm.py --alpha 50 --K 500 --num_seed 8

# Step 3: evaluate all 4 conditions via in-context learning
python scripts/02_evaluate.py

# Step 4: generate the figure
python scripts/03_plot.py
```

Outputs go to `results/`.

## Files

```
praxis_sprint/
├── README.md                this file
├── requirements.txt
├── data/                    truthfulqa data (you provide)
├── results/                 outputs (generated)
├── icm/
│   ├── api.py               hyperbolic API wrapper
│   ├── algorithm.py         ICM main algorithm
│   ├── prompts.py           prompt templates
│   └── evaluate.py          ICL evaluation
├── scripts/
│   ├── 00_verify_api.py
│   ├── 01_run_icm.py
│   ├── 02_evaluate.py
│   └── 03_plot.py
├── critique.md              YOUR critique goes here (Part 2 of sprint)
└── AI_DISCLOSURE.md         honest disclosure of AI tool use
```

## AI Tool Disclosure

See `AI_DISCLOSURE.md`. Per the sprint instructions, this submission used
Claude as an AI tool. The code (Part 1) was written with substantial Claude
assistance; the critique (Part 2) was written by the author independently.

Chat logs are linked in the submission form.

## Expected behavior

- API verification (`00_verify_api.py`): ~5 seconds
- ICM run (`01_run_icm.py`) with K=500: ~30-60 minutes, ~$5-10 in API costs
- Evaluation (`02_evaluate.py`): ~10 minutes, ~$2-3
- Plot generation: instant

Total cost: well under the $20 budget if you stick to these parameters.
