"""MATH-500: Qwen3.5-0.8B (public weights) vs GPT-6 Astra's p50c edit, with the authors' own evaluation protocol
(read from lm_understand-logs/exp8_math/code/{evaluate.py,mathqa.py}; reimplemented here, not run):

- user turn = problem + "\\n" + "Please reason step by step, and put your final answer within \\boxed{}."
- think:   chat template enable_thinking=True,  temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5,
           max 32,768 new tokens, seed 0 (their SAMPLING_THINK35, the post's protocol)
- nothink: chat template enable_thinking=False, temperature 0.7, top_p 0.8,  top_k 20, presence_penalty 1.5,
           max 8,192 new tokens, seed 0 (their SAMPLING_NOTHINK35: the model card's non-thinking benchmark setting,
           which their evaluate.py notes they considered because "thinking looped at 0.8B")
- answer = the last \\boxed{...} after the last </think> (whole text if none); graded with math-verify against the
  MATH-500 'answer' field. A response cut at the token cap counts as wrong unless it boxed the right answer anyway.

  python run_math.py --model base --modes think,nothink --n 500 [--samples 1] [--weights DIR]
Weights default to models/qwen3.5-0.8B (base) and models/qwen3.5-0.8B-astra-p50c (edit) next to this script (see README.md).
Writes results/<model>_<mode>.jsonl.gz (per problem: tokens, finish, pred, correct, full text) and
results/<model>_<mode>.summary.json.
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {"base": HERE / "models/qwen3.5-0.8B", "edit": HERE / "models/qwen3.5-0.8B-astra-p50c"}
WEIGHTS = {"base": "Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17",
           "edit": "Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17 + p50c edit (rebuild_edit.py)"}
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
MODES = {
    "think": dict(thinking=True, sp=dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768, seed=0)),
    "nothink": dict(thinking=False, sp=dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=8192, seed=0)),
}


def last_boxed(s: str):
    """hendrycks last_boxed_only_string + remove_boxed (brace-matched); '\\boxed 5' read to '$' or line end."""
    if "</think>" in s:
        s = s.split("</think>")[-1]
    idx = max(s.rfind("\\boxed"), -1)
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    rest = s[idx:]
    if rest.startswith("\\boxed "):
        tail = rest[len("\\boxed "):]
        return tail.split("$")[0].split("\n")[0].strip()
    i = rest.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(rest)):
        if rest[j] == "{":
            depth += 1
        elif rest[j] == "}":
            depth -= 1
            if depth == 0:
                return rest[i + 1:j]
    return None


def correct(pred, gold) -> bool:
    if pred is None:
        return False
    from math_verify import parse, verify
    try:
        g, p = parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")
        return bool(g) and bool(p) and bool(verify(g, p))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--weights", help="model folder (default: models/... next to this script)")
    ap.add_argument("--modes", default="think")
    ap.add_argument("--n", type=int, default=500, help="first N MATH-500 problems (dataset order)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.78)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-seqs", type=int, default=50, help="concurrency cap: the KV cache holds ~52 full 32k answers")
    ap.add_argument("--chunk", type=int, default=100, help="save results after every chunk of problems")
    a = ap.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / ".hf_datasets"))
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=str(HERE / ".hf_datasets"))
    rows = [ds[i] for i in range(min(a.n, len(ds)))]
    weights = Path(a.weights) if a.weights else MODELS[a.model]
    llm = LLM(model=str(weights), dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=32768 + 2048,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=a.max_seqs)
    tok = llm.get_tokenizer()
    import torch
    gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)"
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for mode in a.modes.split(","):
        cfg = MODES[mode]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=cfg["thinking"]) for r in rows]
        sp = SamplingParams(n=a.samples, **cfg["sp"])
        name = f"{a.model}_{mode}{a.tag}"
        t0 = time.time()
        recs, ntok = [], 0
        part = out_dir / f"{name}.partial.jsonl"
        part.unlink(missing_ok=True)
        for c0 in range(0, len(rows), a.chunk):
            outs = llm.generate(prompts[c0:c0 + a.chunk], sp, use_tqdm=False)
            with open(part, "a") as pf:
                for r, o in zip(rows[c0:c0 + a.chunk], outs):
                    for k, c in enumerate(o.outputs):
                        pred = last_boxed(c.text)
                        n = len(c.token_ids)
                        ntok += n
                        rec = {"id": r["unique_id"], "subject": r["subject"], "level": r["level"], "gold": r["answer"], "sample": k,
                               "tokens": n, "finish": c.finish_reason, "cut": c.finish_reason == "length",
                               "closed_think": "</think>" in c.text, "pred": pred, "correct": correct(pred, r["answer"]),
                               "text": c.text}
                        recs.append(rec)
                        pf.write(json.dumps({k2: v for k2, v in rec.items() if k2 != "text"}) + "\n")
            el = time.time() - t0
            print(f"PROGRESS {name} {len(recs)}/{len(rows) * a.samples} acc={100 * sum(x['correct'] for x in recs) / len(recs):.1f} "
                  f"cut={100 * sum(x['cut'] for x in recs) / len(recs):.1f} tok/s={ntok / el:.0f} elapsed={el / 60:.1f}min", flush=True)
        wall = time.time() - t0
        with gzip.open(out_dir / f"{name}.jsonl.gz", "wt") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        n = len(recs)
        fin = [x for x in recs if not x["cut"]]
        summ = {"model": a.model, "weights": WEIGHTS[a.model] if not a.weights else weights.name, "mode": mode, "sampling": cfg["sp"], "thinking": cfg["thinking"],
                "problems": len(rows), "samples": a.samples, "responses": n,
                "accuracy": round(100 * sum(x["correct"] for x in recs) / n, 2),
                "cut_share": round(100 * sum(x["cut"] for x in recs) / n, 2),
                "accuracy_on_finished": round(100 * sum(x["correct"] for x in fin) / max(1, len(fin)), 2),
                "mean_tokens": round(ntok / n, 1),
                "median_tokens_correct": sorted(x["tokens"] for x in recs if x["correct"])[len([x for x in recs if x["correct"]]) // 2] if any(x["correct"] for x in recs) else None,
                "wall_s": round(wall, 1), "tokens_per_s": round(ntok / wall, 1),
                "gpu": gpu, "date": time.strftime("%Y-%m-%d %H:%M %Z")}
        (out_dir / f"{name}.summary.json").write_text(json.dumps(summ, indent=1))
        print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
