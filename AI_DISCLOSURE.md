# AI Tool Disclosure

Per the sprint instructions ("If you do so, let us know what tools and how, and
include links to relevant chat logs in your submission"), here is an honest
disclosure of AI tool use in this submission.

## Tools used

**Claude (Anthropic), via Claude Code**

## What was done with AI assistance

### Part 1 — Implementation

- The Python implementation (`icm/`, `scripts/`) was written with substantial
  Claude assistance. Specifically:
  - Code structure, module organization, and most function implementations
    were written by Claude based on my reading of the reference repo
    (Jiaxin-Wen/Unsupervised-Elicitation) and the paper.
  - I described what I wanted at the algorithm level; Claude produced
    Python that implemented it.
  - Sprint instructions explicitly state "you're allowed to reuse existing
    code" and "you can use AI tools." Using Claude here is functionally
    equivalent to adapting the reference repo, but cleaner.

- I did the following independently:
  - Set up the Hyperbolic API account and key
  - Verified the API returns top-K logprobs (via `00_verify_api.py`)
  - Ran the pipeline and monitored cost
  - Tuned hyperparameters where the initial defaults didn't work

### Part 2 — Critique

- The critique in `critique.md` was written by me (Aung Maw), not by Claude.
- Claude provided a list of candidate critique angles during a brainstorming
  conversation, but the selected angle, the argument, the proposed test, and
  the writing are mine.
- I have not used Claude to generate, rewrite, or substantively edit the
  critique text.

## Deviation from sprint spec — model substitution (forced by Hyperbolic)

The sprint specifies Llama-3.1-405B (base) and Llama-3.1-405B-Instruct
(chat) via Hyperbolic API. However, as of the time of this submission,
Hyperbolic returns HTTP 400 for Llama-3.1-405B with the message:

  "Only deepseek-ai/DeepSeek-V3-0324 && meta-llama/Llama-3.3-70B-Instruct
   && deepseek-ai/DeepSeek-R1 && Qwen/Qwen3-Coder-480B-A35B-Instruct &&
   deepseek-ai/DeepSeek-R1-0528 && test allowed now"

I emailed shi.feng@gwu.edu about this and (depending on response) am
substituting:
  - Base model: `deepseek-ai/DeepSeek-V3-0324` (closest available
    foundation model)
  - Chat model: `meta-llama/Llama-3.3-70B-Instruct` (closest available
    instruct model)

This is a forced substitution, not a design choice. Results should be
interpreted as showing the ICM algorithm's behavior on DeepSeek-V3
rather than Llama-3.1-405B; the comparison to the paper's Figure 1
will not be exact for this reason.

## Chat logs

The full conversation in which I built the implementation with Claude:
[LINK TO BE FILLED IN — paste shareable link to the relevant Claude conversation here]

## Why I'm disclosing this clearly

I'd rather lose this application by being honest than win it by hiding AI use.
The sprint reviewers explicitly ask for AI tool disclosure, and the program is
in the business of measuring alignment-relevant skills, which include
calibration about my own work.

If reviewers prefer applicants who used AI less heavily, that's a fair signal
about whether I'm the right fit at my current level.

— Aung Maw
