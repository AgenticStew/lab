#!/usr/bin/env python3
"""Flag rows whose repeat runs disagree: (max - min) / median above a threshold (default 5%) on any task. A disturbed run
(another app, a Windows task) shows up as one or two slow runs among steady ones; re-run those row/task pairs.
  python3 spread_check.py [--since 2026-11-13T13:50] [--pct 5]
"""
import json
import statistics
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else ""
pct = float(sys.argv[sys.argv.index("--pct") + 1]) if "--pct" in sys.argv else 5.0
bad = []
for f in sorted((B / "results").glob("*/row.json")):
    r = json.loads(f.read_text())
    if since and r.get("started", "") < since:
        continue
    for task, runs in (r.get("runs") or {}).items():
        t = [x["tps"] for x in runs if x.get("tps")]
        if len(t) < 2:
            continue
        spread = 100 * (max(t) - min(t)) / statistics.median(t)
        flag = spread > pct
        if flag:
            bad.append((f.parent.name, task))
        print(f"{'!!' if flag else '  '} {f.parent.name:24s} {task:8s} {[round(x, 1) for x in t]}  spread {spread:4.1f}%")
print("RE-RUN:", " ".join(f"{a}:{b}" for a, b in bad) if bad else "none")
