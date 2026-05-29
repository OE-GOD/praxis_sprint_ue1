"""Hyperbolic API wrapper.

Two key things this wraps:
  1. Querying a base model (Llama-3.1-405B) and getting top-K logprobs.
     Used by the ICM coherence scorer.
  2. Querying a chat model (Llama-3.1-405B-instruct) for ICL evaluation.

Hyperbolic exposes an OpenAI-compatible completions endpoint at
https://api.hyperbolic.xyz/v1.
"""
import os
import time
from typing import Optional

from openai import OpenAI


HYPERBOLIC_BASE_URL = "https://api.hyperbolic.xyz/v1"

# Sprint spec requires Llama-3.1-405B base + Llama-3.1-405B-Instruct chat,
# but Hyperbolic does not currently allow Llama-3.1-405B for new accounts
# (returns 400 with allowlist of: DeepSeek-V3-0324, Llama-3.3-70B-Instruct,
# DeepSeek-R1, Qwen3-Coder-480B-A35B-Instruct, DeepSeek-R1-0528).
#
# Substituted with closest available models. Substitution documented in
# AI_DISCLOSURE.md and noted in the submission.
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def get_client() -> OpenAI:
    key = os.environ.get("HYPERBOLIC_API_KEY")
    if not key:
        raise SystemExit("Set HYPERBOLIC_API_KEY environment variable.")
    return OpenAI(api_key=key, base_url=HYPERBOLIC_BASE_URL)


def base_logprobs(
    client: OpenAI,
    prompt: str,
    target_tokens: list[str],
    max_retries: int = 6,
) -> dict[str, float]:
    """Query the base model and return the logprob of each target_token as the
    next token after `prompt`.

    Uses completions API with logprobs=20 (top-K). For each target token, we
    look it up in the returned top-K and report its logprob, or -inf if not
    in the top-K.

    Retries with exponential backoff. Hyperbolic backends can be cold/503 on
    first call to a model; later calls warm up. Max wait per attempt ~30s.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.completions.create(
                model=BASE_MODEL,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=20,
            )
            top = resp.choices[0].logprobs.top_logprobs[0]  # dict: token -> logprob
            out = {}
            for t in target_tokens:
                # try exact match first, then with leading space
                if t in top:
                    out[t] = top[t]
                elif (" " + t) in top:
                    out[t] = top[" " + t]
                else:
                    out[t] = float("-inf")
            return out
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Longer backoff for server-side errors (503, 500, overload)
            if any(k in err_str for k in ["503", "500", "overload", "not ready"]):
                wait = min(30, 5 * (2 ** attempt))
                print(f"  retry {attempt+1}/{max_retries} after {wait}s (server cold/overload)")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"base_logprobs failed after {max_retries} attempts: {last_err}")


def chat_complete(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int = 64,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> str:
    """Query the chat model. Returns the text of the assistant's response."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(k in err_str for k in ["503", "500", "overload", "not ready"]):
                wait = min(30, 5 * (2 ** attempt))
                print(f"  retry {attempt+1}/{max_retries} after {wait}s (server cold/overload)")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"chat_complete failed after {max_retries} attempts: {last_err}")
