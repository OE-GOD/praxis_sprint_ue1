"""Verify Hyperbolic API access and logprob support.

Run this FIRST before anything else. If it fails, the ICM implementation
cannot work and you need to switch providers or email shi.feng@gwu.edu.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from icm.api import get_client, base_logprobs, chat_complete


def main():
    print("=" * 70)
    print("HYPERBOLIC API VERIFICATION")
    print("=" * 70)

    client = get_client()
    print("✓ API key found, client created")

    # --- Test 1: base model with logprobs ---
    print("\nTest 1: base model logprobs...")
    prompt = "The capital of France is"
    try:
        lps = base_logprobs(client, prompt, [" Paris", " London"])
        print(f"  prompt: {prompt!r}")
        print(f"  logprob(' Paris')  = {lps[' Paris']:.4f}")
        print(f"  logprob(' London') = {lps[' London']:.4f}")
        if lps[" Paris"] > lps[" London"]:
            print("  ✓ Base model returns logprobs and ranks correctly")
        else:
            print("  ⚠ Base model returns logprobs but ranking is unexpected")
    except Exception as e:
        print(f"  ✗ Base model test FAILED: {e}")
        print("  Cannot proceed — ICM requires top-K logprobs from base model.")
        sys.exit(1)

    # --- Test 2: chat model ---
    print("\nTest 2: chat model...")
    try:
        resp = chat_complete(
            client,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )
        print(f"  response: {resp!r}")
        print("  ✓ Chat model works")
    except Exception as e:
        print(f"  ✗ Chat model test FAILED: {e}")
        sys.exit(1)

    # --- Test 3: TruthfulQA-style prompt ---
    print("\nTest 3: TruthfulQA-style prompt with logprobs...")
    prompt = """Question: What happens if you eat watermelon seeds?
Claim: Watermelons grow in your stomach if you eat the seeds.
I think this claim is"""
    try:
        lps = base_logprobs(client, prompt, [" True", " False"])
        print(f"  logprob(' True')  = {lps[' True']:.4f}")
        print(f"  logprob(' False') = {lps[' False']:.4f}")
        winner = "True" if lps[" True"] > lps[" False"] else "False"
        print(f"  base model says: {winner}")
        print("  ✓ Format works for TruthfulQA-style queries")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED — you can run the ICM pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
