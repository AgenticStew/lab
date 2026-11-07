#!/usr/bin/env python3
"""32-prompt quality run on the llama.cpp native API, streaming.

  run_quality.py --tag qwen|arete [--base http://127.0.0.1:8080]

For every prompt:
- render with the server's template (thinking off) via /apply-template;
- tokenize via /tokenize, so the exact input token IDs are recorded and can be compared across models;
- generate greedily from those IDs via /completion with stream=true, recording a wall-clock timestamp for every streamed
  piece (used for on-screen replays).

One excluded warm-up first; prompt order shuffled once with seed 103 (the same order for both models).
Writes results/<tag>-quality.jsonl; refuses to overwrite.
"""
import argparse, hashlib, json, random, time
from pathlib import Path

import requests

B = Path(__file__).resolve().parent
SETTINGS = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, typical_p=1.0, repeat_penalty=1.0,
                presence_penalty=0.0, frequency_penalty=0.0, seed=42, cache_prompt=False, n_predict=1536)
KW = {"chat_template_kwargs": {"enable_thinking": False}}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def render(base, text):
    r = requests.post(f"{base}/apply-template", json={"messages": [{"role": "user", "content": text}], **KW}, timeout=60)
    r.raise_for_status()
    return r.json()["prompt"]


def tokenize(base, text):
    r = requests.post(f"{base}/tokenize", json={"content": text, "add_special": True, "parse_special": True}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def generate(base, ids, n_predict):
    body = dict(SETTINGS, prompt=ids, n_predict=n_predict, stream=True)
    t0 = time.perf_counter()
    pieces, final = [], {}
    with requests.post(f"{base}/completion", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            d = json.loads(raw[5:].strip())
            now = time.perf_counter() - t0
            if d.get("content"):
                pieces.append([round(now, 5), d["content"]])
            if d.get("stop"):
                final = d
    wall = time.perf_counter() - t0
    return pieces, final, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    a = ap.parse_args()
    out = B / "results" / f"{a.tag}-quality.jsonl"
    fixture = [json.loads(x) for x in (B / "prompts.jsonl").read_text().splitlines() if x.strip()]
    assert len(fixture) == 32 and len({x["id"] for x in fixture}) == 32
    props = requests.get(f"{a.base}/props", timeout=10).json()
    warm = tokenize(a.base, render(a.base, "Reply with the word ready."))
    generate(a.base, warm, 16)                                  # excluded warm-up
    order = list(fixture)
    random.Random(103).shuffle(order)
    with out.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"meta": True, "tag": a.tag, "settings": SETTINGS, "kwargs": KW,
                            "model_path": props.get("model_path"), "build": props.get("build_info"),
                            "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
        for item in order:
            rendered = render(a.base, item["prompt"])
            ids = tokenize(a.base, rendered)
            assert len(ids) + SETTINGS["n_predict"] <= 16384
            pieces, final, wall = generate(a.base, ids, SETTINGS["n_predict"])
            text = "".join(p[1] for p in pieces)
            tm = final.get("timings", {})
            rec = dict(tag=a.tag, id=item["id"], category=item["category"], rendered_prompt=rendered,
                       prompt_token_ids=ids, prompt_hash=digest(rendered), token_hash=digest(ids),
                       settings=SETTINGS, text=text, pieces=pieces, wall_s=round(wall, 4),
                       timings=tm, stop_type=final.get("stop_type"), truncated=final.get("truncated"),
                       tokens_predicted=final.get("tokens_predicted"), tokens_evaluated=final.get("tokens_evaluated"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{item['id']} {tm.get('predicted_n')} tok {tm.get('predicted_per_second', 0):.1f} tok/s "
                  f"stop={final.get('stop_type')} wall={wall:.1f}s", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
