"""results/*.jsonl.gz -> RESULTS.md: my RTX 5090 MATH-500 test of Qwen3.5-0.8B vs GPT-6 Astra's p50c edit (and the
thinking-off control), next to the authors' reported numbers.

  python3 analyze.py
"""
import gzip
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
AUTH = {("base", "think"): (7.9, 93.0), ("edit", "think"): (48.1, 33.1)}


def load(name):
    p = R / f"{name}.jsonl.gz"
    return [json.loads(l) for l in gzip.open(p, "rt")] if p.exists() else []


def summ(name):
    p = R / f"{name}.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def tail_repeats(text: str, n_lines: int = 30) -> bool:
    """True if the last n non-empty lines are dominated by one repeated line (a loop)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()][-n_lines:]
    if len(lines) < 10:
        return False
    top = max(set(lines), key=lines.count)
    return lines.count(top) >= 0.5 * len(lines)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    out = ["# My RTX 5090 test: Qwen3.5-0.8B vs Astra's edit (MATH-500)", ""]
    out += ["Run 2 Nov 2026 on my test PC: RTX 5090 32 GB, Ryzen 9 9950X3D, 64 GB DDR5-6000 (2 x 32 GB), Windows 11 + WSL2 Ubuntu 24.04,",
            "vLLM 0.33.0, torch 2.14 (CUDA 13.0), bf16.",
            "- **Models:** `Qwen/Qwen3.5-0.8B` @ 2fc06364715b967f1860aea9cf38778875588b17 (the public release), and the same weights with Astra's p50c",
            "  edit rebuilt by `rebuild_edit.py` from `lm_understand-logs/exp8_math/runs/p50c/edit` (both changed tensors match the",
            "  authors' `edited_sha256`, so this is the model they evaluated).",
            "- **Protocol** (the authors' `evaluate.py` / `mathqa.py`, reimplemented, not run): user turn = problem + newline + \"Please reason step by step,",
            "  and put your final answer within \\boxed{}.\"; last `\\boxed{}` after `</think>` graded with math-verify.",
            "  - **think** (the post's protocol): thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32,768 new tokens, seed 0.",
            "  - **nothink** (control): thinking off, temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, max 8,192 tokens, seed 0 (the",
            "    authors' own `SAMPLING_NOTHINK35`, the model card's non-thinking benchmark setting).",
            "- **One sample per problem** (the authors averaged 4 on MATH-500), all 500 problems. Expect a few points of sampling noise.", ""]
    rows = []
    for m in ("base", "edit"):
        for mode in ("think", "nothink"):
            s = summ(f"{m}_{mode}")
            if s:
                a = AUTH.get((m, mode))
                rows.append(f"| {'Original' if m == 'base' else 'Edited (p50c)'} | {mode} | {s['accuracy']:.1f}% | {s['cut_share']:.1f}% | "
                            f"{s['accuracy_on_finished']:.1f}% | {s['mean_tokens']:,.0f} | {s['median_tokens_correct'] or 0:,} | "
                            f"{(f'{a[0]}% / {a[1]}%') if a else '—'} | {s['wall_s'] / 60:.0f} min · {s['tokens_per_s']:,.0f} tok/s |")
    out += ["## Results", "",
            "| Model | Mode | Accuracy | Cut at token cap | Accuracy on finished answers | Mean tokens | Median tokens (correct) | Authors (acc / cut) | Time on the 5090 |",
            "|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    bt, et = load("base_think"), load("edit_think")
    if bt and et:
        by = {r["id"]: r for r in bt}
        resc = [r for r in et if r["correct"] and not by[r["id"]]["correct"]]
        brok = [r for r in et if not r["correct"] and by[r["id"]]["correct"]]
        resc_cut = [r for r in resc if by[r["id"]]["cut"]]
        fin_both = [r for r in et if not by[r["id"]]["cut"]]
        out += ["## Paired (same 500 problems, thinking on)", "",
                f"- Rescued (original wrong → edited right): **{len(resc)}**; broken (right → wrong): **{len(brok)}**; exact McNemar p = {mcnemar_p(len(resc), len(brok)):.2g}.",
                f"- Of the rescues, **{len(resc_cut)}** were answers the original model never finished (cut at 32,768 tokens).",
                f"- Problems the original finished: {len(fin_both)}; accuracy on them: original "
                f"{100 * sum(by[r['id']]['correct'] for r in fin_both) / max(1, len(fin_both)):.1f}% → edited {100 * sum(r['correct'] for r in fin_both) / max(1, len(fin_both)):.1f}%.",
                f"- Cut-off original answers whose last 30 lines are one repeated line: {sum(tail_repeats(r['text']) for r in bt if r['cut'])} of {sum(r['cut'] for r in bt)}.",
                f"- Edited answers still cut off and looping the same way: {sum(tail_repeats(r['text']) for r in et if r['cut'])} of {sum(r['cut'] for r in et)}.", ""]
        p807 = [(r, by[r["id"]]) for r in et if r["id"].endswith("precalculus/807.json")]
        if p807:
            e, b = p807[0]
            lb = [l.strip() for l in b["text"].splitlines() if l.strip()]
            out += ["## The cold-open problem (test/precalculus/807: (0, 3) to polar)", "",
                    f"- Original: {b['tokens']:,} tokens, {'cut off' if b['cut'] else 'finished'}, answer {b['pred']!r}, correct={b['correct']}. Last line: `{lb[-1][:90] if lb else ''}`",
                    f"- Edited: {e['tokens']:,} tokens, {'cut off' if e['cut'] else 'finished'}, answer {e['pred']!r}, correct={e['correct']}.", ""]
    bn = summ("base_nothink")
    if bn:
        out += ["## Control: thinking off", "",
                f"- Original model, thinking off (Qwen's default mode for the 0.8B): **{bn['accuracy']:.1f}%**, cut off {bn['cut_share']:.1f}% at 8,192 tokens.",
                "- This is my own quick control with one sample, not the authors' protocol; it isn't a debunk of the post (the post's question is about",
                "  thinking mode), but it shows how much of the gap a simpler loop fix closes.", ""]
        en = summ("edit_nothink")
        if en:
            out += [f"- Astra's edit, thinking off: **{en['accuracy']:.1f}%** (cut off {en['cut_share']:.1f}%): the edit adds nothing in that mode.",
                    f"- Tokens per answer: thinking off ~{bn['mean_tokens']:,.0f} vs the edit with thinking on ~{summ('edit_think')['mean_tokens']:,.0f}.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
