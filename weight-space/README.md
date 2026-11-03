# Weight-Space Programming: my rerun on an RTX 5090

Code and raw results for my video on **"Sparks of Weight-Space Programming"** by Takehi, Teoh, Zhang and Khattab (MIT CSAIL,
blog post, 2 Nov 2026): https://rikiyat.github.io/blog/weight-space-programming/. In it, GPT-6 Astra was given the raw
weights of other neural networks and edited or wrote them directly, with no training.

I reran two of the experiments from the authors' public logs (https://github.com/RikiyaT/lm_understand-logs):

| Folder | What I ran | Result |
|---|---|---|
| [`math500/`](math500/RESULTS.md) | Astra's edit of Qwen3.5-0.8B (Experiment 8): all 500 MATH-500 problems, thinking on and off | Thinking on: original **7.2%**, edited **49.8%** (authors: 7.9% and 48.1%). Thinking off, no edit: **45.6%** |
| [`breakout/`](breakout/RESULTS.md) | Astra's one-shot Breakout network (Experiment 5): 1,000 games of MinAtar Breakout, plus a scripted rule | Network **502.9** (authors: 517.4). A ~30-line scripted rule: **533.9** |

**Test PC:** NVIDIA GeForce RTX 5090 (32 GB), AMD Ryzen 9 9950X3D, 64 GB DDR5-6000 (2 × 32 GB), Windows 11 + WSL2
(Ubuntu 24.04).

## What the MATH-500 test shows
- **The edit is real.** `rebuild_edit.py` rebuilds it from the logs and both edited layers match the authors' sha256, so
  this is the exact model they evaluated. With thinking on, it rescued 214 problems and broke 1.
- **It fixed loops, not maths skill.** All 214 rescues were answers the original never finished: it ran to the
  32,768-token limit, usually repeating itself. The original with thinking switched off (Qwen's own non-thinking settings)
  scores 45.6% with no edit, and the edit adds nothing in that mode (45.0%).

## Setup (both tests)

```bash
git clone https://github.com/RikiyaT/lm_understand-logs     # into this folder
git -C lm_understand-logs checkout 4d09ba6                    # the commit I used (29 Oct 2026)
```

The scripts read the authors' data (the edit, the Breakout network) from that clone; nothing of theirs is copied here.

## MATH-500 (`math500/`)
Needs a CUDA GPU (any card that fits a 0.8B model; a full run is ~1 h 40 min on the 5090) and
`pip install -r requirements.txt`.

```bash
cd math500
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/qwen3.5-0.8B
mkdir -p models/qwen3.5-0.8B-astra-p50c
cp models/qwen3.5-0.8B/*.json models/qwen3.5-0.8B/*.jinja models/qwen3.5-0.8B/merges.txt models/qwen3.5-0.8B/vocab.json models/qwen3.5-0.8B-astra-p50c/
python rebuild_edit.py models/qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors \
    ../lm_understand-logs/exp8_math/runs/p50c/edit \
    models/qwen3.5-0.8B-astra-p50c/model.safetensors-00001-of-00001.safetensors   # prints "sha256 OK" for both layers
./run_all.sh            # or: python run_math.py --model base --modes think,nothink --n 500
python analyze.py       # results/ -> RESULTS.md
```

`results/` holds my run: one `.summary.json` per model and mode, and every answer in full (`.jsonl.gz`: tokens, whether it
hit the cap, the extracted answer, the grade, the model's whole output). To read the original model's loop on the problem
from the start of the video:

```bash
python -c "import gzip,json; r=[json.loads(l) for l in gzip.open('results/base_think.jsonl.gz','rt') if 'precalculus/807' in l][0]; print(r['tokens'], r['text'][-1500:])"
```

## Breakout (`breakout/`)
CPU only. `pip install numpy && pip install --no-deps MinAtar==1.0.15`, then see [`breakout/RESULTS.md`](breakout/RESULTS.md).

## Notes
- **One sample per problem.** The authors averaged four samples on MATH-500, so expect a few points of noise between runs.
- **vLLM reruns aren't bit-for-bit.** The sampling seed is fixed, but vLLM's results shift slightly with batch size, so a
  rerun gives the same scores within noise, not identical answers. (In a 4-problem check, 2 answers came out identical.)
- **131,072 weights.** The edit touches 64 columns of 1,024 numbers in each of two layers (2 × 64 × 1,024 = 131,072, the
  figure in the post). After rounding to bfloat16, 130,188 of them differ from the original, as the authors' exp8 README
  notes.
- **My own reimplementation.** `run_math.py` reimplements the authors' `evaluate.py` / `mathqa.py` protocol (prompt,
  sampling settings, answer extraction, math-verify grading). I read their code but didn't run it.

## Credits and licences
- Logs, the edit data and the Breakout network: Takehi, Teoh, Zhang, Khattab, `lm_understand-logs`, CC BY 4.0
  (their scripts are MIT). The recordings in `breakout/results/` are games played by their network.
- Qwen3.5-0.8B: Qwen team, Apache-2.0.
- MATH-500 (`HuggingFaceH4/MATH-500`, from Hendrycks et al.'s MATH): the reference answers appear in `math500/results/`.
- `breakout/scripted_rule.py` follows MinAtar's ball physics (Kenny Young and Tian Tian, GPL-3.0), so that one file is
  GPL-3.0 (`breakout/LICENSE-MinAtar`).
