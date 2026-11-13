#!/usr/bin/env python3
"""Find the most GPU layers (-ngl) of the 27B that fit a smaller card's VRAM budget, per guesser setting.

  calibrate.py            # writes offload_ngl.json (read by bench.py) and calibrate.log

Budget = card VRAM minus 1 GiB left for Windows and the display (a card that also drives the desktop): 16 GB card ->
15,360 MiB, 12 GB card -> 11,264 MiB, measured as nvidia-smi memory.used above the desktop's reading after one short
generation (so it includes the CUDA context, KV cache and compute buffers, not just the weights).
The guesser always sits fully on the GPU, so it costs the big model some GPU layers: that is the honest trade.
"""
import json, time
import bench as b

BUDGET = {"16gb": 15360, "12gb": 11264}
GUESSERS = {"none": [], "mtp-n3": ["mtp", 3], "ngram": ["ngram", 0]}


def probe(ngl, g):
    extra = []
    if g and g[0] == "mtp":
        extra = b.mtp(g[1])
    elif g and g[0] == "ngram":
        extra = ["--spec-type", "ngram-mod"]
    label = f"cal-{g[0] if g else 'none'}-ngl{ngl}"
    b.ROWS[label] = b.srv(b.T27, *extra, ngl=ngl)
    r = b.run_row(label, ["code"], 1, 0.0, 64, 42)
    return r.get("vram_delta_mib") if r.get("status") == "ok" else None


def main():
    out, log = {}, []
    for tier, budget in BUDGET.items():
        for name, g in GUESSERS.items():
            lo, hi, best = (40, 65, None) if tier == "16gb" else (16, 56, None)
            while lo <= hi:                       # binary search the largest ngl under budget
                mid = (lo + hi) // 2
                d = probe(mid, g)
                log.append(f"{tier} {name} ngl={mid} delta={d}")
                print(log[-1], flush=True)
                if d is not None and d <= budget:
                    best, lo = (mid, d), mid + 1
                else:
                    hi = mid - 1
            out[f"{tier}-{name}"] = [best[0], g] if best else None
            log.append(f"=> {tier} {name}: ngl {best}")
            print(log[-1], flush=True)
    (b.B / "offload_ngl.json").write_text(json.dumps({k: v for k, v in out.items() if v}, indent=1))
    (b.B / "calibrate.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
