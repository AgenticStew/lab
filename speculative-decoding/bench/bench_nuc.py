#!/usr/bin/env python3
"""The same benchmark on a second, real card: an RTX 3080 Ti 12 GB in a NUC 12 Extreme (i9-12900, 64 GB).

  bench_nuc.py <label> ... [--runs N] [--prompts ...] [--max-tokens N]     # rows below; results in results/<label>/
  bench_nuc.py --calibrate                                                  # GPU layers of the 27B that fit 12 GB

How it runs: llama.cpp b12189 (Windows CUDA 13.5 release build) on the NUC under C:\\spec-test, started over SSH
(host alias "nuc") and bound to the NUC's 127.0.0.1:8091. An SSH tunnel maps it to this
PC's 127.0.0.1:8091, so bench.py's client, prompts and metrics are unchanged. VRAM comes from the NUC's own nvidia-smi.
Each row's server is stopped with taskkill. The model files are the same pinned files as on the 5090 (../nuc/setup.ps1).
"""
import json, subprocess, sys
import bench as b

NUC = "nuc"
W = r"C:\spec-test"
CARD_MIB = 12288
MARGIN_MIB = 350           # stay clear of Windows' shared-memory spill near the top of the card


def ssh(cmd, timeout=60):
    return subprocess.run(["ssh", NUC, cmd], capture_output=True, text=True, timeout=timeout)


def vram_used():
    try:
        out = ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


_stop = b.stop


def stop(p):
    ssh("taskkill /F /IM llama-server.exe", timeout=30)
    _stop(p)


b.vram_used = vram_used
b.stop = stop


def tunnel():
    """One persistent local forward 8091 -> NUC 127.0.0.1:8091 (idempotent)."""
    probe = subprocess.run(["bash", "-c", "ss -ltn | grep -q '127.0.0.1:8091 '"])
    if probe.returncode != 0:
        subprocess.run(["ssh", "-f", "-N", "-o", "ExitOnForwardFailure=yes", "-L", "8091:127.0.0.1:8091", NUC], check=True)


def nsrv(model, *extra, ngl="99", ctx="8192"):
    args = [rf"{W}\llama\llama-server.exe", "-m", rf"{W}\models\{model}", "-ngl", str(ngl), "-c", str(ctx),
            "--parallel", "1", "-fa", "on", "--no-cache-prompt", "--cache-ram", "0", "--jinja", "--fit", "off",
            "-t", "8", "--host", "127.0.0.1", "--port", "8091", *extra]
    return ["ssh", NUC, " ".join(args)]


Q27, Q9 = "Qwen3.8-27B-UD-Q4_K_M.gguf", "Qwen3.5-9B-Q4_K_M.gguf"
L8, L1 = "llama3.1-8b-q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q8_0.gguf"
MTP = ["-md", rf"{W}\models\mtp-Qwen3.8-27B-Q4_0.gguf", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-ngld", "99"]
DF2 = ["-md", rf"{W}\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "--spec-type", "draft-dflash", "--spec-draft-n-max", "7",
       "-ngld", "99"]

b.ROWS.update({
    "nuc-9b-none": nsrv(Q9),
    "nuc-9b-mtp-n2": nsrv(Q9, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"),
    "nuc-l8-none": nsrv(L8),
    "nuc-l8-d1-n12": nsrv(L8, "-md", rf"{W}\models\{L1}", "--spec-type", "draft-simple", "--spec-draft-n-max", "12",
                          "-ngld", "99"),
})
GUESS27 = {"none": [], "mtp-n3": MTP, "df2-n7": DF2}
CAL = b.B / "nuc_ngl.json"
if CAL.exists():
    for name, ngl in json.loads(CAL.read_text()).items():
        b.ROWS[f"nuc-27b-{name}"] = nsrv(Q27, *GUESS27[name], ngl=ngl)


def calibrate():
    """Largest -ngl whose peak VRAM (a 128-token story run) stays under the card minus a margin, per guesser."""
    out = {}
    for name, extra in GUESS27.items():
        lo, hi, best = 16, 60, None
        while lo <= hi:
            mid = (lo + hi) // 2
            lab = f"cal-nuc-{name}-ngl{mid}"
            b.ROWS[lab] = nsrv(Q27, *extra, ngl=mid)
            r = b.run_row(lab, ["story"], 1, 0.0, 128, 42)
            peak = r.get("vram_peak_mib")
            ok = r.get("status") == "ok" and peak is not None and peak <= CARD_MIB - MARGIN_MIB
            print(f"cal {name} ngl={mid} peak={peak} MiB -> {'fits' if ok else 'too big'}", flush=True)
            if ok:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        out[name] = best
        print(f"=> {name}: ngl {best}", flush=True)
    CAL.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    tunnel()
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        b.main()
