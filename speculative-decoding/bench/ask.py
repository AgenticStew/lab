#!/usr/bin/env python3
"""Ask a running llama-server one of the test prompts and watch the reply arrive (the terminal demo).

  ask.py <prompt-key> [--port 8091] [--max-tokens 512]

Prints the reply as it streams. When speculative decoding is on, several tokens arrive in one burst (one verify
step): every token of a burst except the last was a guess the big model accepted, so those print in green, and the
last token of each burst (the big model's own next word) prints in the normal colour. The live line under the reply
shows tokens so far and tokens per second; the summary at the end uses the server's own timings.
"""
import argparse, json, sys, time
from pathlib import Path

import requests

B = Path(__file__).resolve().parent
PROMPTS = json.loads((B / "prompts.json").read_text())
GREEN, RESET, DIM, BOLD = "\x1b[32m", "\x1b[0m", "\x1b[2m", "\x1b[1m"
BURST = 0.003   # tokens arriving within 3 ms of each other came from the same verify step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    p = PROMPTS[a.prompt]
    print(f"{BOLD}> {p['label']}{RESET}  {DIM}(greedy, thinking off, up to {a.max_tokens} tokens){RESET}\n", flush=True)
    body = {"messages": p["messages"], "stream": True, "max_tokens": a.max_tokens, "temperature": 0, "top_k": 1,
            "seed": 42, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    pending, last_t, n, first_t, timings = None, None, 0, None, None
    out = sys.stdout

    def emit(piece, guessed):
        out.write((GREEN + piece + RESET) if guessed else piece)

    with requests.post(f"http://127.0.0.1:{a.port}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            if line[6:] == b"[DONE]":
                break
            j = json.loads(line[6:])
            if j.get("timings"):
                timings = j["timings"]
            for ch in j.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if not piece:
                    continue
                now = time.perf_counter()
                first_t = first_t or now
                n += 1
                if pending is not None:
                    # the previous token was a guess if this one arrived in the same burst
                    emit(pending, now - last_t < BURST)
                pending, last_t = piece, now
                out.flush()
    if pending is not None:
        emit(pending, False)
    t = timings or {}
    n_tok = t.get("predicted_n", n)
    span = (last_t - first_t) if (last_t and first_t and last_t > first_t) else None
    tps = (n_tok - 1) / span if span else 0.0
    out.write(f"\n\n{BOLD}{n_tok} tokens · {tps:.1f} tokens per second{RESET}")
    if t.get("draft_n"):
        acc = t["draft_n_accepted"] / t["draft_n"]
        steps = (n_tok - 1) - t["draft_n_accepted"]
        out.write(f"  {DIM}·{RESET}  {GREEN}{acc*100:.1f}% of guesses accepted{RESET}"
                  f"  {DIM}·{RESET}  {(n_tok - 1) / steps:.2f} tokens per check")
    out.write("\n")
    out.flush()


if __name__ == "__main__":
    main()
