#!/usr/bin/env python3
"""Launch one engine row, time its start, run the harness, stop it, write results/<label>/row.json.

  row.py <label> [--suite speed,prefill,quality] [--keep]   # --keep: leave the server up (for footage/long tests)
  row.py --list

A row that never answers /v1/models (or exits) is recorded as did-not-start with the last log lines: failures are results.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

import requests

B = Path(__file__).resolve().parent
H = Path.home()
M = H / "models/fmt-test"           # paths below are my machine's: edit them to yours
GGUF = next((H / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots").glob("*/Qwen3.8-27B-UD-Q4_K_M.gguf"))
MTP_GGUF = M / "gguf-mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
LLAMA = H / "src/llama.cpp/build/bin/llama-server"
VLLM = H / "tools/vllm/.venv/bin/vllm"
TABBY_PY = H / "tools/exllamav3/.venv/bin/python"
TABBY = H / "tools/exllamav3/src/tabbyAPI"
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

LLAMA_A = [str(LLAMA), "-m", str(GGUF), "-ngl", "99", "-c", "32768", "--parallel", "1", "-fa", "on", "-ctk", "f16",
           "-ctv", "f16", "--no-cache-prompt", "--cache-ram", "0", "--jinja", "--fit", "off", "--host", "127.0.0.1", "--port", "8080",
           "--alias", "qwen38", "--metrics"]
MTP_ARGS = ["-md", str(MTP_GGUF), "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.1",
            "-ngld", "99"]


def vllm(model, *extra):
    return [str(VLLM), "serve", str(model), "--served-model-name", "qwen38", "--max-model-len", "32768",
            "--max-num-seqs", "1", "--language-model-only", "--no-enable-prefix-caching", "--reasoning-parser", "qwen3",
            "--gpu-memory-utilization", "@UTIL@", "--host", "127.0.0.1", "--port", "8000",
            *(["--kv-cache-dtype", "bfloat16"] if "--kv-cache-dtype" not in extra else []), *extra]


MTP3 = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
TFR = H / "tools/tensorfold/recipe"
TF_ENV = {"HOST": "127.0.0.1", "PORT": "30000", "CONTEXT": "32768", "PARALLEL": "1", "TF_QWEN_KV_PAGES": "0",
          "PROMPT_CACHE_GIB": "0", "KV_CACHE": "bf16", "ALIAS": "qwen38-mlx4"}

def YSK(model_dir):
    return [str(H / "yasakani-venv/bin/yasakani"), "serve", "--model-dir", str(H / "models" / model_dir),
            "--dll", str(H / "src/yasakani/build-linux/libyskn.so"), "--host", "127.0.0.1", "--port", "8765", "--max-ctx", "32768"]


ROWS = {
    # Table A: no speculation, 32K, FP16/BF16 KV, batch 1
    "gguf-A": dict(cmd=LLAMA_A, port=8080, extra={**THINK_OFF, "cache_prompt": False}),
    "exl3-A": dict(tabby={"draft_mode": "disabled"}, port=5000, extra=THINK_OFF),
    "awq-A": dict(cmd=vllm(M / "awq"), port=8000, extra=THINK_OFF),
    "gptq-A": dict(cmd=vllm(M / "gptq"), port=8000, extra=THINK_OFF),
    "mxfp4-A": dict(cmd=vllm(M / "mxfp4"), port=8000, extra=THINK_OFF),
    "mxfp4-A-b12x": dict(cmd=vllm(M / "mxfp4", "--linear-backend", "b12x"), port=8000, extra=THINK_OFF),
    "mxfp4-A-marlin": dict(cmd=vllm(M / "mxfp4", "--linear-backend", "marlin"), port=8000, extra=THINK_OFF),
    "mxfp4-mtp-b12x": dict(cmd=vllm(M / "mxfp4", "--linear-backend", "b12x", *MTP3), port=8000, extra=THINK_OFF),
    "mxfp4-mtp-marlin": dict(cmd=vllm(M / "mxfp4", "--linear-backend", "marlin", *MTP3), port=8000, extra=THINK_OFF),
    "nvfp4-A": dict(cmd=vllm(M / "nvfp4"), port=8000, extra=THINK_OFF),
    "nvfp4-A-eager": dict(cmd=vllm(M / "nvfp4", "--enforce-eager"), port=8000, extra=THINK_OFF),
    "fp8-A": dict(cmd=vllm(M / "fp8"), port=8000, extra=THINK_OFF),
    # Table B: best practical (drafting on, or a memory variant)
    "gguf-mtp": dict(cmd=LLAMA_A + MTP_ARGS, port=8080, extra={**THINK_OFF, "cache_prompt": False}),
    "exl3-mtp": dict(tabby={"draft_mode": "mtp"}, port=5000, extra=THINK_OFF),
    "exl3-dflash2": dict(tabby={"draft_mode": "model", "draft_model_name": "dflash2-exl3"}, port=5000, extra=THINK_OFF),
    "awq-mtp": dict(cmd=vllm(M / "awq", *MTP3), port=8000, extra=THINK_OFF),
    "gptq-mtp": dict(cmd=vllm(M / "gptq", *MTP3), port=8000, extra=THINK_OFF),
    "mxfp4-mtp": dict(cmd=vllm(M / "mxfp4", *MTP3), port=8000, extra=THINK_OFF),
    "nvfp4-fp8kv": dict(cmd=vllm(M / "nvfp4", "--kv-cache-dtype", "fp8"), port=8000, extra=THINK_OFF),
    "nvfp4-mtp": dict(cmd=vllm(M / "nvfp4", *MTP3), port=8000, extra=THINK_OFF),
    "nvfp4-fp8kv-mtp": dict(cmd=vllm(M / "nvfp4", "--kv-cache-dtype", "fp8", *MTP3), port=8000, extra=THINK_OFF),
    "gptq-A-eager": dict(cmd=vllm(M / "gptq", "--enforce-eager"), port=8000, extra=THINK_OFF),
    "fp8-A-8k": dict(cmd=[c if c != "32768" else "8192" for c in vllm(M / "fp8")], port=8000, extra=THINK_OFF),
    "fp8-eager": dict(cmd=vllm(M / "fp8", "--enforce-eager"), port=8000, extra=THINK_OFF),
    "tf-dflash2-32k": dict(cmd=[str(TFR / "serve.sh")], env=TF_ENV, port=30000, extra=THINK_OFF),
    "tf-nodraft-32k": dict(cmd=[str(TFR / "serve-nodraft.sh")], env=TF_ENV, port=30000, extra=THINK_OFF),
    "tf-nodrafts-32k": dict(cmd=[str(TFR / "serve-nodrafts.sh")], env=TF_ENV, port=30000, extra=THINK_OFF),
    "tf-default": dict(cmd=[str(TFR / "serve.sh")], env={"HOST": "127.0.0.1", "PORT": "30000", "ALIAS": "qwen38-mlx4"}, port=30000, extra=THINK_OFF),
    "tf-nodraft-262k": dict(cmd=[str(TFR / "serve-nodraft.sh")], env={"HOST": "127.0.0.1", "PORT": "30000", "ALIAS": "qwen38-mlx4",
                                                                       "TF_QWEN_KV_PAGES": "0", "PARALLEL": "1", "PROMPT_CACHE_GIB": "0"}, port=30000, extra=THINK_OFF),
    "gguf-A-ub2048": dict(cmd=LLAMA_A + ["-b", "2048", "-ub", "2048"], port=8080, extra={**THINK_OFF, "cache_prompt": False}),
    # long context (separate segment): biggest window that loads, then cold 100K / 250K prompts
    "gguf-262k-q8": dict(cmd=[c if c != "32768" else "262144" for c in LLAMA_A[:LLAMA_A.index("-ctk")]] + ["-ctk", "q8_0", "-ctv", "q8_0"]
                         + LLAMA_A[LLAMA_A.index("-ctv") + 2:], port=8080, extra={**THINK_OFF, "cache_prompt": False}),
    "exl3-262k-q8": dict(tabby={"draft_mode": "disabled", "ctx": 262144, "cache_mode": "8,8"}, port=5000, extra=THINK_OFF),
    "fp8-eager-fp8kv": dict(cmd=vllm(M / "fp8", "--enforce-eager", "--kv-cache-dtype", "fp8"), port=8000, extra=THINK_OFF),
    # YASAKANI (13 Nov, github.com/yasakani-dev/yasakani @758ef28): its own compressed files + 5090 kernels + DFlash 2 drafting.
    # Windows-only upstream; built for Linux by ../yasakani-linux/install.sh (self-check passed). 32K like the others;
    # its loop stopper off (no other engine has one; the harness counts loops itself).
    "ysk-speed": dict(cmd=YSK("yasakani-speed"), port=8765, extra={**THINK_OFF, "loop_guard": False}),
    "ysk-quality": dict(cmd=YSK("yasakani-quality"), port=8765, extra={**THINK_OFF, "loop_guard": False}),
}


def tabby_config(label, over):
    cfg = B / "launch" / f"tabby-{label}.yml"
    cfg.parent.mkdir(exist_ok=True)
    dm = over.get("draft_mode", "disabled")
    lines = ["network:", "  host: 127.0.0.1", "  port: 5000", "  disable_auth: true",
             "logging:", "  log_prompt: false", "  log_generation_params: false",
             "model:", f"  model_dir: {M}", "  model_name: exl3-4.00bpw", "  backend: exllamav3",
             f"  max_seq_len: {over.get('ctx', 32768)}", f"  cache_size: {over.get('ctx', 32768)}",
             f"  cache_mode: {over.get('cache_mode', 'FP16')}", "  max_batch_size: 1", "  vision: false",
             "draft_model:", f"  draft_mode: {dm}", f"  draft_model_dir: {M}"]
    if over.get("draft_model_name"):
        lines.append(f"  draft_model_name: {over['draft_model_name']}")
    if dm == "mtp":                      # MTP-3 like every other engine; DFlash2 keeps its drafter's own block size
        lines.append(f"  draft_num_tokens: {over.get('draft_num_tokens', 3)}")
    cfg.write_text("\n".join(lines) + "\n")
    return [str(TABBY_PY), str(TABBY / "main.py"), "--config", str(cfg)]


def smi_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("--suite", default="speed,prefill,quality")
    ap.add_argument("--harness-args", default="")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=float, default=1500)
    a = ap.parse_args()
    if a.list or not a.label:
        print("\n".join(ROWS))
        return
    key = a.label
    for suf in ("-ident", "-drift", "-q", "-pf", "-r2"):
        if key not in ROWS and key.endswith(suf):
            key = key[: -len(suf)]
    row = ROWS[key]
    out = B / "results" / a.label
    out.mkdir(parents=True, exist_ok=True)
    cmd = tabby_config(a.label, row["tabby"]) if "tabby" in row else row["cmd"]
    env = dict(os.environ)
    env["PATH"] = f"{VLLM.parent}:{env['PATH']}"
    env.update(row.get("env", {}))
    env.setdefault("MAX_JOBS", "4")   # FlashInfer JIT-compiles SM120 FP4 kernels on first use; 16+ parallel cicc jobs hit the OOM killer
    base = f"http://127.0.0.1:{row['port']}/v1"
    baseline = smi_used()
    total = 32607
    util = round(min(0.94, (total - baseline - 1600) / total), 3)    # what the desktop leaves, minus ~1.6 GiB vLLM sees as used (own CUDA context, WSL)
    cmd = [str(util) if c == "@UTIL@" else c for c in cmd]
    info = {"label": a.label, "cmd": cmd, "baseline_mib": baseline, "gpu_util": util, "t_launch": time.time(), "status": "starting"}
    FLAG = Path("/tmp/bench_timing.flag")
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(a.label + " startup")             # load time is measured too
    logf = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                            cwd=str(TABBY) if "tabby" in row else str(B))
    print(f"[{a.label}] launched pid {proc.pid}; baseline {baseline} MiB; {' '.join(cmd)[:200]}", flush=True)
    peak = baseline
    while True:
        time.sleep(0.5)
        try:
            peak = max(peak, smi_used())
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None:
            info.update(status="did-not-start", exit=proc.returncode)
            break
        try:
            hc = requests.get(f"http://127.0.0.1:{row['port']}/health", timeout=2).status_code
            if hc in (200, 404) and requests.get(f"{base}/models", timeout=2).status_code == 200:
                info.update(status="ready", load_s=round(time.time() - info["t_launch"], 1))
                break
        except requests.RequestException:
            pass
        if time.time() - info["t_launch"] > a.timeout:
            info.update(status="timeout")
            break
    time.sleep(2)
    if FLAG.exists():
        FLAG.unlink()
    info["after_load_mib"] = smi_used()
    info["peak_load_mib"] = max(peak, info["after_load_mib"])
    print(f"[{a.label}] {info['status']} load={info.get('load_s')} s, after load {info['after_load_mib']} MiB "
          f"(+{info['after_load_mib'] - baseline})", flush=True)
    if info["status"] == "ready":
        h = [str(B / ".venv/bin/python"), str(B / "harness.py"), "--base", base, "--label", a.label, "--suite", a.suite,
             "--extra", json.dumps(row["extra"])] + (["--model", "exl3-4.00bpw"] if "tabby" in row else []) \
            + (a.harness_args.replace(";", " ").split() if a.harness_args else [])
        info["harness_rc"] = subprocess.run(h).returncode
        try:
            info["metrics_text"] = requests.get(f"http://127.0.0.1:{row['port']}/metrics", timeout=5).text[-20000:]
        except requests.RequestException:
            pass
    info["log_tail"] = (out / "server.log").read_text(errors="replace")[-6000:]
    if not a.keep or info["status"] != "ready":
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=60)
        except ProcessLookupError:
            pass                                   # the server already exited (a failed start)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for _ in range(60):
            if smi_used() < baseline + 600:
                break
            time.sleep(1)
    info["t_end"] = time.time()
    (out / "row.json").write_text(json.dumps(info, indent=1))
    print(f"[{a.label}] done: {info['status']}", flush=True)


if __name__ == "__main__":
    main()
