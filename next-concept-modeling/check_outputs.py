#!/usr/bin/env python3
"""Repetition-loop and empty-output check on the quality pickles (TESTPLAN step 6: "every output checked for
repetition loops before its number is used"; C038 empty strings).

    python -I check_outputs.py /results/results.json /results/ncm-pen.pkl /results/ncm-nopen.pkl \
        /results/duo-ancestral.pkl /results/duo-greedy.pkl

The pickles are this run's own outputs (eval_shard.py's and the harness's), opened inside the container.
For each file and seed it counts empty decoded strings, samples whose raw ids hold a run of 8 or more of one
token, samples where one 4-gram fills 25% or more of the 4-grams, and distinct-1/2 over the seed. It flags; it
does not drop anything. Written to results.json under output_checks.<file stem> and printed.
"""
import collections
import json
import os
import pickle
import sys
from datetime import datetime, timezone

RUN = 8
NGRAM = 4
NGRAM_SHARE = 0.25


def longest_run(ids):
    best = cur = 0
    prev = object()
    for t in ids:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


def top_ngram_share(ids, n=NGRAM):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return collections.Counter(grams).most_common(1)[0][1] / len(grams)


def distinct(seqs, n):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def check(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    hyps, raw = d["hyps"], d.get("raw_ids") or []
    seeds = d.get("seeds") or [None] * len(hyps)
    if len(raw) != len(hyps):
        sys.exit(f"STOP: {path}: {len(raw)} raw id rows vs {len(hyps)} decoded strings")
    per = {}
    for s in dict.fromkeys(seeds):
        sel = [k for k, x in enumerate(seeds) if x == s]
        rs = [raw[k] for k in sel]
        loops_run = [k for k in sel if longest_run(raw[k]) >= RUN]
        loops_gram = [k for k in sel if top_ngram_share(raw[k]) >= NGRAM_SHARE]
        per[str(s)] = {
            "samples": len(sel),
            "empty_strings": sum(1 for k in sel if not hyps[k].strip()),
            "lengths": sorted(set(len(r) for r in rs)),
            f"run_ge_{RUN}": len(loops_run),
            f"top_{NGRAM}gram_share_ge_{int(NGRAM_SHARE * 100)}pct": len(loops_gram),
            "distinct_1": distinct(rs, 1),
            "distinct_2": distinct(rs, 2),
            "examples_flagged": [hyps[k][:160] for k in sorted(set(loops_run + loops_gram))[:3]],
        }
    tot = lambda key: sum(v[key] for v in per.values())  # noqa: E731
    return {"file": path, "samples": len(hyps), "empty_strings": tot("empty_strings"),
            f"run_ge_{RUN}": tot(f"run_ge_{RUN}"),
            f"top_{NGRAM}gram_share_ge_{int(NGRAM_SHARE * 100)}pct": tot(f"top_{NGRAM}gram_share_ge_{int(NGRAM_SHARE * 100)}pct"),
            "per_seed": per}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    results, paths = sys.argv[1], sys.argv[2:]
    out = {}
    for p in paths:
        r = check(p)
        r["time_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out[os.path.splitext(os.path.basename(p))[0]] = r
        print(f"{p}: {r['samples']} samples, {r['empty_strings']} empty, {r[f'run_ge_{RUN}']} with a {RUN}+ token run, "
              f"{r[f'top_{NGRAM}gram_share_ge_{int(NGRAM_SHARE * 100)}pct']} with a dominant {NGRAM}-gram")
    doc = {}
    if os.path.exists(results):
        with open(results, encoding="utf-8") as f:
            doc = json.load(f)
    doc.setdefault("output_checks", {}).update(out)
    tmp = results + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, results)


if __name__ == "__main__":
    main()
