#!/usr/bin/env python3
"""Speculative decoding on one RTX 5090: run one llama-server configuration ("row") and record it.

  bench.py <label> [<label> ...]      # run rows in order, write results/<label>/row.json
  bench.py --list                     # show the rows
  bench.py <label> --prompts code,explain --runs 3 --temp 0

Each row: start llama-server, wait for /health, one warm-up, then N measured runs per prompt (greedy, thinking off,
512-token cap), stop the server. Decode speed = (tokens - 1) / (last chunk time - first chunk time), measured by the
client like a user sees it; the server's own figure is stored too. Draft stats come from the server's timings
(draft_n, draft_n_accepted). VRAM = peak nvidia-smi memory.used minus the reading before the server started, plus
llama.cpp's own buffer sizes from its log. A row that never answers is recorded as did-not-start with the log tail.
"""
import argparse, hashlib, json, os, re, signal, statistics, subprocess, sys, threading, time
from pathlib import Path

import requests

B = Path(__file__).resolve().parent
RES = B / "results"
H = Path.home()
LLAMA = H / "src/llama.cpp-b12189/build/bin/llama-server"
M = H / "video-studio/models"
T27 = next((H / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots").glob("*/Qwen3.8-27B-UD-Q4_K_M.gguf"))
MTP27 = M / "fmt-test/gguf-mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
DF2 = M / "specdec/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
D08 = M / "specdec/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
D2 = M / "specdec/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf"
T9 = M / "specdec/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
PORT = 8091
PROMPTS = json.loads((B / "prompts.json").read_text())


def srv(model, *extra, ngl="99", ctx="8192"):
    return [str(LLAMA), "-m", str(model), "-ngl", str(ngl), "-c", str(ctx), "--parallel", "1", "-fa", "on",
            "--no-cache-prompt", "--cache-ram", "0", "--jinja", "--fit", "off", "-t", "16",
            "--host", "127.0.0.1", "--port", str(PORT), *extra]


def dm(path, n, kind="draft-simple", pmin=None):
    a = ["-md", str(path), "--spec-type", kind, "--spec-draft-n-max", str(n), "-ngld", "99"]
    return a + (["--spec-draft-p-min", str(pmin)] if pmin is not None else [])


def mtp(n, path=MTP27, pmin=None):
    a = (["-md", str(path)] if path else []) + ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
    a += ["-ngld", "99"] if path else []
    return a + (["--spec-draft-p-min", str(pmin)] if pmin is not None else [])


ROWS = {}
# --- the 27B on the full card: no guesser vs each kind of guesser
ROWS["27b-none"] = srv(T27)
for n in (2, 4, 8, 16):
    ROWS[f"27b-d08-n{n}"] = srv(T27, *dm(D08, n))
    ROWS[f"27b-d2-n{n}"] = srv(T27, *dm(D2, n))
for n in (1, 2, 3, 4, 6):
    ROWS[f"27b-mtp-n{n}"] = srv(T27, *mtp(n))
for n in (4, 7, 10, 15):
    ROWS[f"27b-df2-n{n}"] = srv(T27, *dm(DF2, n, kind="draft-dflash"))
# the UD GGUF also carries its own MTP layer (blk.64.nextn): plain --spec-type draft-mtp, no -md
ROWS["27b-mtp-n3-bare"] = srv(T27, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3")
ROWS["27b-ngram-mod"] = srv(T27, "--spec-type", "ngram-mod")
ROWS["27b-ngram-simple"] = srv(T27, "--spec-type", "ngram-simple", "--spec-draft-n-max", "64")
for n in (3, 4):   # built-in guesser plus copy-from-the-text, both on
    ROWS[f"27b-mtp-n{n}+ngram"] = srv(T27, "-md", str(MTP27), "--spec-type", "draft-mtp,ngram-mod",
                                      "--spec-draft-n-max", str(n), "-ngld", "99")
# --- smaller cards: the 9B (fits 8 GB) with its built-in MTP head, or the 0.8B guesser
ROWS["9b-none"] = srv(T9)
for n in (1, 2, 3, 4):
    ROWS[f"9b-mtp-n{n}"] = srv(T9, *mtp(n, path=None))
for n in (2, 4, 8):
    ROWS[f"9b-d08-n{n}"] = srv(T9, *dm(D08, n))


# YASAKANI (github.com/yasakani-dev/yasakani @758ef28, 13 Nov 2026): its own OpenAI-compatible server, built for Linux by
# ~/video-studio/setup/install-yasakani.sh (self-check passed). Same prompts, greedy, thinking off, 512 tokens.
YSK_CLI = H / "video-studio/tools/yasakani/.venv/bin/yasakani"
YSK_LIB = H / "src/yasakani/build-linux/libyskn.so"


def ysk(model_dir):
    return [YSK_CLI, "serve", "--model-dir", M / model_dir, "--dll", YSK_LIB, "--host", "127.0.0.1", "--port", str(PORT),
            "--max-ctx", "32768"]


ROWS["ysk-speed"] = ysk("yasakani-speed")
ROWS["ysk-quality"] = ysk("yasakani-quality")


def add_offload_rows():
    """27B squeezed into a 12 GB / 16 GB card: -ngl picked per budget (see OFFLOAD.md), guesser on the GPU."""
    f = B / "offload_ngl.json"
    if not f.exists():
        return
    for k, v in json.loads(f.read_text()).items():   # e.g. {"16gb-none": [44, []], "16gb-mtp-n3": [40, ["mtp", 3]]}
        ngl, g = v
        extra = []
        if g and g[0] == "mtp":
            extra = mtp(g[1])
        elif g and g[0] == "d08":
            extra = dm(D08, g[1])
        elif g and g[0] == "df2":
            extra = dm(DF2, g[1], kind="draft-dflash")
        elif g and g[0] == "ngram":
            extra = ["--spec-type", "ngram-mod"]
        ROWS[f"27b-{k}"] = srv(T27, *extra, ngl=ngl)


add_offload_rows()


def vram_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            v = vram_used()
            if v:
                self.peak = max(self.peak, v)
            time.sleep(0.5)


# Qwen3.8-27B model card's recommended sampling (huggingface.co/Qwen/Qwen3.8-27B, README lines 252-253)
SAMPLING = {
    "greedy": {},
    "qwen-instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                      "chat_template_kwargs": {"enable_thinking": False}},
    "qwen-thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                      "chat_template_kwargs": {"enable_thinking": True}},
}
MODE = {"sampling": "greedy", "ysk": False}


# Warm-up prompt shared by every row since 13 Nov ~07:00: unrelated to the five tasks, so a guesser that remembers text
# across requests (ngram-mod's shared pool) can't learn a measured answer from it. Earlier rows warmed up on their first
# test prompt (64 tokens); that only matters for ngram-mod rows, which were re-run with --fresh.
WARM = [{"role": "user", "content": "Name three primary colors, one per line."}]


def chat(prompt, temp, max_tokens, seed, extra=None):
    msgs = PROMPTS[prompt]["messages"] if prompt in PROMPTS else WARM
    body = {"messages": msgs, "stream": True, "max_tokens": max_tokens, "temperature": temp,
            "seed": seed, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    if temp == 0:
        body["top_k"] = 1
    if MODE["ysk"]:     # YASAKANI: count tokens from usage (its chunks under-count), and no loop stopper (llama.cpp has none)
        body.update(stream_options={"include_usage": True}, loop_guard=False)
    if MODE["sampling"] != "greedy":
        body.pop("top_k", None)
        body.update(SAMPLING[MODE["sampling"]])
    body.update(extra or {})
    t0 = time.perf_counter()
    chunks, timings, text, finish, usage = [], None, [], None, None
    with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            j = json.loads(data)
            now = time.perf_counter() - t0
            if j.get("timings"):
                timings = j["timings"]
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                dl = ch.get("delta") or {}
                piece = dl.get("content") or dl.get("reasoning_content")   # thinking text streams separately
                if piece:
                    chunks.append([round(now, 6), piece])
                    text.append(piece)
                finish = ch.get("finish_reason") or finish
    text = "".join(text)
    t = timings or {}
    n = t.get("predicted_n") or (usage or {}).get("completion_tokens") or len(chunks)
    span = chunks[-1][0] - chunks[0][0] if len(chunks) > 1 else 0
    tps = (n - 1) / span if span > 0 else None
    d_n, d_acc = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    steps = (n - 1) - d_acc if d_n else None          # each verify step yields accepted + 1 tokens
    return {"tps": tps, "server_tps": t.get("predicted_per_second"), "n": n, "ttft": chunks[0][0] if chunks else None,
            "prompt_n": t.get("prompt_n"), "draft_n": d_n, "draft_acc": d_acc,
            "accept_rate": (d_acc / d_n) if d_n else None,
            "mean_len": ((n - 1) / steps) if steps else None, "finish": finish,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text, "chunks": chunks,
            "usage": usage, "n_chunks": len(chunks)}


BUF = re.compile(r"(CUDA0|CPU|CUDA_Host|CPU_Mapped)\s+(model|KV|compute|RS|output) buffer size =\s+([\d.]+) MiB")


def run_row(label, prompts, runs, temp, max_tokens, seed):
    cmd = ROWS[label]
    MODE["ysk"] = "yasakani" in " ".join(map(str, cmd))
    out = RES / label
    out.mkdir(parents=True, exist_ok=True)
    # 13 Nov: a leftover llama-server on PORT answered /health for every row of a whole queue (each row's own server
    # failed to bind), so 23 minutes of "results" all measured that one plain server. Refuse to start on a busy port.
    import socket
    with socket.socket() as s_:
        if s_.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"{label}: port {PORT} is already in use (a leftover server?): stop it first")
    log = open(out / "server.log", "w")
    base = vram_used()
    env = dict(os.environ)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    row = {"label": label, "cmd": [str(c) for c in cmd], "vram_base_mib": base, "temp": temp, "max_tokens": max_tokens,
           "seed": seed, "sampling": MODE["sampling"], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ready = False
    while time.time() - t0 < 600:
        if p.poll() is not None:
            break
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).json().get("status") == "ok":
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    row["load_s"] = round(time.time() - t0, 1)
    time.sleep(1.0)
    if ready and p.poll() is not None:          # health answered, but not by our server
        ready = False
    if not ready:
        row["status"] = "did-not-start"
        log.flush()
        row["log_tail"] = (out / "server.log").read_text(errors="replace").splitlines()[-25:]
        stop(p)
        (out / "row.json").write_text(json.dumps(row, indent=1))
        print(f"{label}: DID NOT START", flush=True)
        return row
    vp = VramPeak()
    vp.start()
    try:
        chat("_warmup", temp, 64, seed)                                          # warm-up
        row["runs"] = {}
        for pr in prompts:
            # sampled modes: a different seed per run, so the median covers several different replies
            rs = [chat(pr, temp, max_tokens, seed + (i if MODE["sampling"] != "greedy" else 0)) for i in range(runs)]
            row["runs"][pr] = rs
            tps = statistics.median([r["tps"] for r in rs if r["tps"]])
            acc = rs[0]["accept_rate"]
            print(f"{label:22s} {pr:8s} {tps:7.1f} tok/s  n={rs[0]['n']:4d}"
                  + (f"  accept {acc*100:5.1f}%  len {rs[0]['mean_len']:.2f}" if acc is not None else ""), flush=True)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = f"error: {e!r}"
        print(f"{label}: ERROR {e!r}", flush=True)
    finally:
        vp.stop.set()
        row["vram_peak_mib"] = vp.peak
        row["vram_delta_mib"] = (vp.peak - base) if base else None
        stop(p)
        log.close()
    row["buffers"] = [m.groups() for m in BUF.finditer((out / "server.log").read_text(errors="replace"))]
    (out / "row.json").write_text(json.dumps(row, indent=1))
    return row


def run_row_fresh(label, prompts, runs, temp, max_tokens, seed):
    """Every measured run on a newly started server: for guessers that remember text across requests (ngram-mod)."""
    merged = None
    for pr in prompts:
        for i in range(runs):
            tmp = f"{label}--{pr}-{i}"
            ROWS[tmp] = ROWS[label]
            r = run_row(tmp, [pr], 1, temp, max_tokens, seed + (i if MODE["sampling"] != "greedy" else 0))
            if merged is None:
                merged = {k: v for k, v in r.items() if k != "runs"}
                merged.update(label=label, runs={}, fresh=True, vram_delta_mib=r.get("vram_delta_mib"))
            if r.get("status") != "ok":
                merged["status"] = r.get("status")
                continue
            merged["runs"].setdefault(pr, []).extend(r["runs"][pr])
            merged["vram_delta_mib"] = max(merged.get("vram_delta_mib") or 0, r.get("vram_delta_mib") or 0)
            # the per-run folder is kept only for its server log
            (RES / tmp / "row.json").rename(RES / tmp / "fresh_run.json")
    out = RES / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "row.json").write_text(json.dumps(merged, indent=1))
    for pr, rs in merged["runs"].items():
        print(f"{label:22s} {pr:8s} {statistics.median([x['tps'] for x in rs]):7.1f} tok/s (fresh server per run)",
              flush=True)
    return merged


def stop(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help="append to the result folder name (e.g. -t1 for a temperature run)")
    ap.add_argument("--sampling", default="greedy", choices=list(SAMPLING))
    ap.add_argument("--fresh", action="store_true", help="start a new server for every measured run")
    a = ap.parse_args()
    MODE["sampling"] = a.sampling
    if a.list:
        for k, v in ROWS.items():
            print(k, " ".join(map(str, v)))
        return
    for lab in a.labels:
        if lab not in ROWS:
            sys.exit(f"unknown row {lab}")
    for lab in a.labels:
        if a.suffix:
            ROWS[lab + a.suffix] = ROWS[lab]
        (run_row_fresh if a.fresh else run_row)(lab + a.suffix, a.prompts.split(","), a.runs, a.temp, a.max_tokens, a.seed)


if __name__ == "__main__":
    main()
