#!/usr/bin/env python3
"""Race replay data: every token of the measured code answer with its real arrival time (seconds from the first token).

  race_data.py            # writes replay/race_code.json (27b-none, 27b-mtp-n3, 27b-df2-n7), replay/race_ysk.json (+ ysk-speed, YASAKANI) and replay/race_l70.json
                          # (l70-none-ngl52, l70-d1-n8-ngl50) from bench/results/<row>/row.json

The median-speed run of each row (greedy, so all runs are the same text), so the replay matches the 3-run medians
shown on screen. The arrival clock is the streaming client's perf_counter. (race_l70.json was first built inline on
13 Nov morning; it is made here since the afternoon re-run.)
"""
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / "bench" / "results"


def race(labels, name):
    out = {}
    for lab in labels:
        runs = json.loads((B / lab / "row.json").read_text())["runs"]["code"]
        run = sorted(runs, key=lambda r: r["tps"])[len(runs) // 2]          # the median run
        t0 = run["chunks"][0][0]
        out[lab] = {"tps": run["tps"], "n": run["n"], "text": run["text"],
                    "tokens": [[round(t - t0, 4), piece] for t, piece in run["chunks"]],
                    "duration_s": round(run["chunks"][-1][0] - t0, 3)}
    texts = [out[lab]["text"] for lab in labels]
    out["identical_text"] = all(t == texts[0] for t in texts)
    Path(__file__).with_name(name).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:16s} {v['n']} tokens in {v['duration_s']:.2f} s ({v['tps']:.1f} tok/s)")
    print(f"{name}: identical text across the rows: {out['identical_text']}")


race(("27b-none", "27b-mtp-n3", "27b-df2-n7"), "race_code.json")
race(("l70-none-ngl52", "l70-d1-n8-ngl50"), "race_l70.json")
race(("27b-none", "27b-df2-n7", "ysk-speed"), "race_ysk.json")
