#!/usr/bin/env python3
"""One harness for every engine (llama.cpp, TabbyAPI/ExLlamaV3, vLLM, TensorFold): OpenAI-compatible streaming chat.

  harness.py --base http://127.0.0.1:8080/v1 --label gguf-A --suite speed,prefill,quality [--extra JSON] [--runs 3]

Writes results/<label>/: speed.jsonl, prefill.jsonl, quality.jsonl (one line per request: timestamps of every streamed
chunk, usage, text), smi.csv (nvidia-smi at 10 Hz for the whole row), summary.json (medians), and appends to
results/index.jsonl. Metrics (identical for every engine):
  decode tok/s = (completion_tokens - 1) / (t_last_chunk - t_first_chunk)
  TTFT         = request sent -> first content chunk;  prefill tok/s = prompt_tokens / TTFT (cold: unique nonce prefix)
  e2e tok/s    = completion_tokens / wall time
Greedy, thinking off, prompt cache off or defeated (every request starts with a fresh nonce line).
"""
import argparse, json, os, re, statistics as st, subprocess, sys, tempfile, threading, time, uuid
from pathlib import Path

import requests

B = Path(__file__).resolve().parent
P = json.loads((B / "prompts.json").read_text())
PL = json.loads((B / "prompts_long.json").read_text()) if (B / "prompts_long.json").exists() else {}  # long-context segment only
SAMPLING = {"temperature": 0, "top_p": 1, "seed": 0}


class Smi:
    """nvidia-smi sampler (10 Hz) -> csv with wall-clock seconds."""
    COLS = "memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu"

    def __init__(self, path):
        self.path, self.stop_ev = Path(path), threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        p = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.COLS}", "--format=csv,noheader,nounits", "-lms", "100"],
                             stdout=subprocess.PIPE, text=True, bufsize=1)
        with self.path.open("a") as f:
            for line in p.stdout:
                f.write(f"{time.time():.3f},{line.strip()}\n")
                if self.stop_ev.is_set():
                    break
        p.terminate()

    def __enter__(self):
        if not self.path.exists():
            self.path.write_text("t," + self.COLS + "\n")
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        self.th.join(timeout=3)


def nonce_msgs(messages):
    m = json.loads(json.dumps(messages))
    m[0]["content"] = f"[request {uuid.uuid4().hex[:12]}]\n" + m[0]["content"]
    return m


def chat(base, model, messages, max_tokens, extra, stream=True, timeout=1800):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream, **SAMPLING, **extra}
    if stream:
        body["stream_options"] = {"include_usage": True}
    rec = {"t_send": time.time(), "chunks": [], "usage": None, "timings": None, "finish": None, "error": None}
    try:
        r = requests.post(f"{base}/chat/completions", json=body, stream=stream, timeout=timeout)
        if r.status_code != 200:
            rec["error"] = f"HTTP {r.status_code}: {r.text[:500]}"
            return rec
        r.encoding = "utf-8"               # SSE responses often omit the charset; requests would guess latin-1
        if not stream:
            d = r.json()
            rec["t_last"] = time.time()
            rec["text"] = d["choices"][0]["message"].get("content") or ""
            rec["reasoning"] = d["choices"][0]["message"].get("reasoning_content") or ""
            rec["usage"], rec["finish"] = d.get("usage"), d["choices"][0].get("finish_reason")
            return rec
        text, reasoning = [], []
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            now = time.time()
            if d.get("usage"):
                rec["usage"] = d["usage"]
            if d.get("timings"):
                rec["timings"] = d["timings"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                c = delta.get("content") or ""
                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if c:
                    rec["chunks"].append([round(now - rec["t_send"], 5), c])
                    text.append(c)
                if rc:
                    reasoning.append(rc)
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
        rec["t_last"] = time.time()
        rec["text"], rec["reasoning"] = "".join(text), "".join(reasoning)
    except Exception as e:  # noqa: BLE001 - a failed request is a result
        rec["error"] = repr(e)
    return rec


def loop_flag(text):
    """True if some 8-word sequence repeats >= 4 times in the second half (greedy repetition, cf. YASAKANI's 424->303)."""
    w = text.split()
    half = w[len(w) // 2:]
    seen = {}
    for i in range(len(half) - 8):
        k = " ".join(half[i:i + 8])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 4:
            return True
    return False


def metrics(rec):
    if rec.get("error") or not rec["chunks"]:
        return {}
    u = rec["usage"] or {}
    n = u.get("completion_tokens")
    t_first, t_last = rec["chunks"][0][0], rec["chunks"][-1][0]
    m = {"ttft": round(t_first, 4), "completion_tokens": n, "prompt_tokens": u.get("prompt_tokens"),
         "wall": round(rec["t_last"] - rec["t_send"], 4), "n_chunks": len(rec["chunks"]), "loop": loop_flag(rec["text"])}
    if n and n > 1 and t_last > t_first:
        m["decode_tps"] = round((n - 1) / (t_last - t_first), 2)
        m["e2e_tps"] = round(n / m["wall"], 2)
    if u.get("prompt_tokens"):
        m["prefill_tps"] = round(u["prompt_tokens"] / t_first, 1)
    return m


def spec_snapshot(base):
    """cumulative drafted/accepted tokens from the server's Prometheus /metrics (llama.cpp, vLLM), or None"""
    try:
        t = requests.get(base.rsplit("/v1", 1)[0] + "/metrics", timeout=3).text
    except requests.RequestException:
        return None
    out = {}
    for k, name in (("drafted", "spec_decode_num_draft_tokens_total"), ("accepted", "spec_decode_num_accepted_tokens_total")):
        m = re.findall(rf"^[^#\n]*{name}(?:\{{[^}}]*\}})?\s+([0-9.e+]+)$", t, re.M)
        if m:
            out[k] = sum(float(x) for x in m)
    return out or None


def log(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def suite_speed(a, out, ids=("prose", "code", "json", "edit")):
    res = {}
    for pid in ids:
        p = P["speed"][pid]
        vals = []
        for i in range(a.warmups + a.runs):
            s0 = spec_snapshot(a.base)
            rec = chat(a.base, a.model, nonce_msgs(p["messages"]), p["max_tokens"], a.extra)
            s1 = spec_snapshot(a.base) if s0 else None
            m = metrics(rec)
            if s0 and s1:
                m["spec"] = {k: s1.get(k, 0) - s0.get(k, 0) for k in s1}
            warm = i < a.warmups
            log(out / "speed.jsonl", {"prompt": pid, "run": i, "warmup": warm, **m, "finish": rec["finish"],
                                       "error": rec["error"], "t_send": rec["t_send"], "chunks": rec["chunks"],
                                       "timings": rec["timings"], "text": rec.get("text", ""),
                                       "reasoning_chars": len(rec.get("reasoning") or "")})
            print(f"  speed {pid:6s} run {i}{' (warm-up)' if warm else ''}: {m.get('decode_tps')} tok/s "
                  f"n={m.get('completion_tokens')} ttft={m.get('ttft')} loop={m.get('loop')} {rec['error'] or ''}", flush=True)
            if not warm and m.get("decode_tps"):
                vals.append(m)
        if vals:
            res[pid] = {"decode_tps": st.median(v["decode_tps"] for v in vals),
                        "e2e_tps": st.median(v["e2e_tps"] for v in vals),
                        "completion_tokens": st.median(v["completion_tokens"] for v in vals),
                        "runs": [v["decode_tps"] for v in vals], "loops": sum(v["loop"] for v in vals)}
    return res


def suite_prefill(a, out, ids=("2k", "8k", "32k")):
    res = {}
    for pid in ids:
        p = P["prefill"].get(pid) or PL[pid]
        n_meas = a.runs if pid != "32k" else max(2, a.runs - 1)
        if pid.startswith("long"):              # one short warm-up, then one cold measured run (a 250K prompt takes minutes)
            chat(a.base, a.model, nonce_msgs(P["prefill"]["2k"]["messages"]), 8, a.extra)
            n_meas = 0
        vals, found = [], 0
        for i in range(1 + n_meas):
            rec = chat(a.base, a.model, nonce_msgs(p["messages"]), p["max_tokens"], a.extra)
            m = metrics(rec)
            hit = p["answer"] in (rec.get("text") or "")
            if pid.startswith("long"):
                i = 1
            log(out / "prefill.jsonl", {"prompt": pid, "run": i, "warmup": i == 0, **m, "hit": hit, "error": rec["error"],
                                         "timings": rec["timings"], "text": rec.get("text", ""),
                                         "reasoning_chars": len(rec.get("reasoning") or "")})
            print(f"  prefill {pid:4s} run {i}: ttft={m.get('ttft')} {m.get('prefill_tps')} tok/s "
                  f"prompt={m.get('prompt_tokens')} hit={hit} {rec['error'] or ''}", flush=True)
            if i > 0 and m.get("ttft"):
                vals.append(m)
                found += hit
        if vals:
            res[pid] = {"ttft": st.median(v["ttft"] for v in vals), "prefill_tps": st.median(v["prefill_tps"] for v in vals),
                        "prompt_tokens": vals[0]["prompt_tokens"], "needle": f"{found}/{len(vals)}"}
    return res


def suite_ident(a, out):
    """Fixed prompts with NO nonce, 256 tokens, greedy: compare a drafted row's text with its plain row (drafting should
    not change a greedy answer, up to batch-shape rounding)."""
    res = {}
    for pid in ("prose", "json", "code"):
        p = P["speed"][pid]
        rec = chat(a.base, a.model, p["messages"], 256, a.extra)
        log(out / "ident.jsonl", {"prompt": pid, "text": rec.get("text", ""), "error": rec["error"], **metrics(rec)})
        res[pid] = len(rec.get("text") or "")
        print(f"  ident {pid}: {res[pid]} chars", flush=True)
    return res


def extract_code(text, entry):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    return code


def run_py(src, timeout=20):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def suite_quality(a, out):
    q, res = P["quality"], {}
    from jsonschema import Draft202012Validator
    # HumanEval-20: hidden tests
    ok = 0
    for t in q["humaneval"]:
        msg = [{"role": "user", "content": "Complete this Python function. Reply with the complete function (with any "
                "imports it needs) in one ```python code block and nothing else.\n\n```python\n" + t["prompt"] + "```"}]
        rec = chat(a.base, a.model, msg, 768, a.extra, stream=False)
        code = extract_code(rec.get("text") or "", t["entry_point"])
        if f"def {t['entry_point']}" not in code:
            code = t["prompt"] + code
        passed, err = run_py(t["prompt"].split("def ")[0] + "\n" + code + "\n\n" + t["test"] + f"\ncheck({t['entry_point']})\n")
        ok += passed
        log(out / "quality.jsonl", {"test": "humaneval", "id": t["task_id"], "pass": passed, "err": err,
                                     "finish": rec["finish"], "text": rec.get("text"), "error": rec["error"]})
    res["humaneval"] = {"pass": ok, "n": len(q["humaneval"])}
    print(f"  humaneval {ok}/{len(q['humaneval'])}", flush=True)
    # GSM8K-20
    ok = 0
    for t in q["gsm8k"]:
        msg = [{"role": "user", "content": t["question"] + "\n\nSolve it step by step, briefly. End with a final line "
                "in the form 'Answer: <number>'."}]
        rec = chat(a.base, a.model, msg, 1024, a.extra, stream=False)
        txt = rec.get("text") or ""
        mm = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", txt)
        got = (mm[-1] if mm else "").replace(",", "")
        right = False
        try:
            right = got != "" and abs(float(got) - float(t["answer"])) < 1e-6
        except ValueError:
            pass
        ok += right
        log(out / "quality.jsonl", {"test": "gsm8k", "id": t["idx"], "pass": right, "got": got, "want": t["answer"],
                                     "finish": rec["finish"], "text": txt, "error": rec["error"]})
    res["gsm8k"] = {"pass": ok, "n": len(q["gsm8k"])}
    print(f"  gsm8k {ok}/{len(q['gsm8k'])}", flush=True)
    # MMLU-200
    ok = 0
    for t in q["mmlu"]:
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", t["choices"]))
        msg = [{"role": "user", "content": f"{t['question']}\n\n{opts}\n\nAnswer with the letter only."}]
        rec = chat(a.base, a.model, msg, 8, a.extra, stream=False)
        m = re.search(r"\b([ABCD])\b", (rec.get("text") or "").strip())
        right = bool(m and m.group(1) == t["answer"])
        ok += right
        log(out / "quality.jsonl", {"test": "mmlu", "id": t["idx"], "pass": right, "got": m and m.group(1),
                                     "want": t["answer"], "text": rec.get("text"), "error": rec["error"]})
    res["mmlu"] = {"pass": ok, "n": len(q["mmlu"])}
    print(f"  mmlu {ok}/{len(q['mmlu'])}", flush=True)
    # JSON schema x3
    ok = 0
    for t in q["json"]:
        rec = chat(a.base, a.model, [{"role": "user", "content": t["prompt"]}], 1500, a.extra, stream=False)
        txt = (rec.get("text") or "").strip()
        valid, why = False, ""
        try:
            arr = json.loads(txt)
            v = Draft202012Validator(t["schema"])
            errs = [e.message for el in arr for e in v.iter_errors(el)] if isinstance(arr, list) else ["not a list"]
            valid = isinstance(arr, list) and len(arr) == 10 and not errs
            why = f"len={len(arr) if isinstance(arr, list) else '-'} errs={errs[:2]}"
        except Exception as e:  # noqa: BLE001
            why = repr(e)[:200]
        ok += valid
        log(out / "quality.jsonl", {"test": "json", "id": t["seed"], "pass": valid, "why": why, "text": txt,
                                     "error": rec["error"]})
    res["json"] = {"pass": ok, "n": len(q["json"])}
    print(f"  json {ok}/{len(q['json'])}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", default="speed,prefill,quality")
    ap.add_argument("--speed-ids", default="prose,code,json,edit")
    ap.add_argument("--prefill-ids", default="2k,8k,32k")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--extra", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    a.extra = json.loads(a.extra)
    if not a.model:
        a.model = requests.get(f"{a.base}/models", timeout=10).json()["data"][0]["id"]
    out = B / "results" / a.label
    out.mkdir(parents=True, exist_ok=True)
    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"label": a.label}
    summary.update({"model": a.model, "base": a.base, "extra": a.extra, "note": a.note or summary.get("note", "")})
    print(f"[{a.label}] model={a.model} suites={a.suite}", flush=True)
    with Smi(out / "smi.csv"):
        for s in a.suite.split(","):
            t0 = time.time()
            timing = s in ("speed", "prefill")          # other GPU jobs on this PC wait while this flag exists
            if timing:
                Path("/tmp/bench_timing.flag").write_text(a.label)
            elif Path("/tmp/bench_timing.flag").exists():
                Path("/tmp/bench_timing.flag").unlink()
            if s == "speed":
                summary["speed"] = {**summary.get("speed", {}), **suite_speed(a, out, a.speed_ids.split(","))}
            elif s == "prefill":
                summary["prefill"] = {**summary.get("prefill", {}), **suite_prefill(a, out, a.prefill_ids.split(","))}
            elif s == "ident":
                summary["ident"] = suite_ident(a, out)
            elif s == "quality":
                summary["quality"] = suite_quality(a, out)
            summary.setdefault("suite_times", {})[s] = [t0, time.time()]
            summ_path.write_text(json.dumps(summary, indent=1))
    if Path("/tmp/bench_timing.flag").exists():
        Path("/tmp/bench_timing.flag").unlink()
    log(B / "results/index.jsonl", {"t": time.time(), "label": a.label, "suite": a.suite, "summary": summary})
    print(json.dumps({k: summary.get(k) for k in ("speed", "prefill", "quality")}, indent=1))


if __name__ == "__main__":
    main()
