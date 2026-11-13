#!/usr/bin/env python3
"""Every 5090 number the two videos quote, from the morning session (results_s1_morning/) and the quiet re-run
(results/, queue6.sh), side by side. Medians of the runs, as report.py computes them.
  python3 numbers.py [--json numbers.json]
"""
import json
import sys
from pathlib import Path

import report

B = Path(__file__).resolve().parent
T5 = ["code", "edit", "explain", "story", "json"]


def tables(res_dir: Path) -> dict:
    report.RES = res_dir
    rows = report.load()
    return {k: report.summarize(v) for k, v in rows.items()}


def tps(S, row, task):
    try:
        return S[row][task]["tps"]
    except KeyError:
        return None


def verify(res_dir: Path):
    p = res_dir / "verify_cost_27b.json"
    if not p.exists():
        return {}
    out = {}
    for d in json.loads(p.read_text()):
        if d.get("n_prompt") and not d.get("n_gen"):
            out[d["n_prompt"]] = round(1000 * d["n_prompt"] / d["avg_ts"], 2)     # ms per pass
    return out


def collect(res_dir: Path) -> dict:
    S = tables(res_dir)
    N = {}
    for row in ("27b-none", "27b-mtp-n3", "27b-df2-n7", "27b-ngram-mod", "27b-d08-n8"):
        for t in T5:
            N[f"{row}/{t}"] = tps(S, row, t)
    for t in T5:
        base = N[f"27b-none/{t}"]
        for row in ("27b-mtp-n3", "27b-df2-n7", "27b-ngram-mod", "27b-d08-n8"):
            v = N[f"{row}/{t}"]
            N[f"x {row}/{t}"] = round(v / base, 3) if v and base else None
        N[f"acc 27b-d08-n8/{t}"] = (S.get("27b-d08-n8", {}).get(t) or {}).get("accept")
        N[f"acc 27b-df2-n7/{t}"] = (S.get("27b-df2-n7", {}).get(t) or {}).get("accept")
        N[f"same 27b-df2-n7/{t}"] = (S.get("27b-df2-n7", {}).get(t) or {}).get("text") == (S.get("27b-none", {}).get(t) or {}).get("text")
    for n in (1, 2, 3, 4, 6):
        for t in ("code", "story"):
            N[f"mtp-n{n}/{t}"] = tps(S, f"27b-mtp-n{n}", t)
    for sfx, tasks in (("-t07", ("code", "explain", "story")), ("-think", ("code", "explain"))):
        for row in ("27b-none", "27b-mtp-n3", "27b-df2-n7"):
            for t in tasks:
                N[f"{row}{sfx}/{t}"] = tps(S, row + sfx, t)
    for row in ("9b-none", "9b-mtp-n2"):
        for t in ("code", "story"):
            N[f"{row}/{t}"] = tps(S, row, t)
    for t in T5:
        a, b = tps(S, "l8-d1-n12", t), tps(S, "l8-none", t)
        N[f"x l8-d1-n12/{t}"] = round(a / b, 3) if a and b else None
    for row in ("27b-16gb-none", "27b-16gb-mtp-n3"):
        for t in ("code", "explain", "story"):
            N[f"{row}/{t}"] = tps(S, row, t)
    for row in ("l70-none-ngl52", "l70-d1-n8-ngl50"):
        for t in ("code", "explain"):
            N[f"{row}/{t}"] = tps(S, row, t)
    for k, v in verify(res_dir).items():
        N[f"verify ms/pass pp{k}"] = v
    return N


def main():
    old, new = collect(B / "results_s1_morning"), collect(B / "results")
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v)))
    for k in old:
        o, n = old[k], new.get(k)
        ch = ""
        if isinstance(o, float) and isinstance(n, float) and o:
            ch = f"{100 * (n - o) / o:+.1f}%"
        print(f"{k:34s} {fmt(o):>10s} {fmt(n):>10s} {ch:>8s}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"morning": old, "quiet": new}, indent=1))


if __name__ == "__main__":
    main()
