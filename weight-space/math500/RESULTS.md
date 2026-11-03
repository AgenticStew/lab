# My RTX 5090 test: Qwen3.5-0.8B vs Astra's edit (MATH-500)

Run 2 Nov 2026 on my test PC: RTX 5090 32 GB, Ryzen 9 9950X3D, 64 GB DDR5-6000 (2 x 32 GB), Windows 11 + WSL2 Ubuntu 24.04,
vLLM 0.33.0, torch 2.14 (CUDA 13.0), bf16.
- **Models:** `Qwen/Qwen3.5-0.8B` @ 2fc06364715b967f1860aea9cf38778875588b17 (the public release), and the same weights with Astra's p50c
  edit rebuilt by `rebuild_edit.py` from `lm_understand-logs/exp8_math/runs/p50c/edit` (both changed tensors match the
  authors' `edited_sha256`, so this is the model they evaluated).
- **Protocol** (the authors' `evaluate.py` / `mathqa.py`, reimplemented, not run): user turn = problem + newline + "Please reason step by step,
  and put your final answer within \boxed{}."; last `\boxed{}` after `</think>` graded with math-verify.
  - **think** (the post's protocol): thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32,768 new tokens, seed 0.
  - **nothink** (control): thinking off, temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, max 8,192 tokens, seed 0 (the
    authors' own `SAMPLING_NOTHINK35`, the model card's non-thinking benchmark setting).
- **One sample per problem** (the authors averaged 4 on MATH-500), all 500 problems. Expect a few points of sampling noise.

## Results

| Model | Mode | Accuracy | Cut at token cap | Accuracy on finished answers | Mean tokens | Median tokens (correct) | Authors (acc / cut) | Time on the 5090 |
|---|---|---|---|---|---|---|---|---|
| Original | think | 7.2% | 93.6% | 100.0% | 31,012 | 5,896 | 7.9% / 93.0% | 54 min · 4,804 tok/s |
| Original | nothink | 45.6% | 33.0% | 67.5% | 3,801 | 693 | — | 4 min · 8,296 tok/s |
| Edited (p50c) | think | 49.8% | 32.6% | 73.9% | 21,958 | 15,348 | 48.1% / 33.1% | 35 min · 5,283 tok/s |
| Edited (p50c) | nothink | 45.0% | 33.8% | 66.8% | 3,892 | 712 | — | 4 min · 8,386 tok/s |

## Paired (same 500 problems, thinking on)

- Rescued (original wrong → edited right): **214**; broken (right → wrong): **1**; exact McNemar p = 8.2e-63.
- Of the rescues, **214** were answers the original model never finished (cut at 32,768 tokens).
- Problems the original finished: 32; accuracy on them: original 100.0% → edited 96.9%.
- Cut-off original answers whose last 30 lines are one repeated line: 284 of 468.
- Edited answers still cut off and looping the same way: 13 of 163.

## The cold-open problem (test/precalculus/807: (0, 3) to polar)

- Original: 32,768 tokens, cut off, answer None, correct=False. Last line: `*   Wait, is there any possibility`
- Edited: 9,248 tokens, finished, answer '(3, \\frac{\\pi}{2})', correct=True.

## Control: thinking off

- Original model, thinking off (Qwen's default mode for the 0.8B): **45.6%**, cut off 33.0% at 8,192 tokens.
- This is my own quick control with one sample, not the authors' protocol; it isn't a debunk of the post (the post's question is about
  thinking mode), but it shows how much of the gap a simpler loop fix closes.

- Astra's edit, thinking off: **45.0%** (cut off 33.8%): the edit adds nothing in that mode.
- Tokens per answer: thinking off ~3,801 vs the edit with thinking on ~21,958.
