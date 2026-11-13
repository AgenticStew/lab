# Speculative decoding on an RTX 5090 (and an RTX 3080 Ti), plus YASAKANI on Linux

Video: "Did One Developer Just Make My RTX 5090 8x Faster?" (Nov 2026), and three Shorts.

Speculative decoding: a small, fast guesser writes a few tokens ahead and the big model checks them all in one pass, keeping
the ones it agrees with. This folder has the harness, the prompts and every raw run behind the video's numbers, and the
script that builds YASAKANI (an engine built for Windows) for Linux.

## Hardware and software
- RTX 5090 32 GB (Palit), driver 621.01, stock 575 W · Ryzen 9 9950X3D · 64 GB DDR5-6000 · Windows 11 + WSL2 (Ubuntu 24.04)
- RTX 3080 Ti 12 GB · Core i9-12900 · 64 GB DDR4-3200 (NUC 12 Extreme)
- llama.cpp b12189 (CUDA 13.3) · YASAKANI v0.1.1 (commit 758ef28), built for Linux with `yasakani-linux/install.sh`

## Main results: this harness (`bench/`; tokens a second, Qwen3.8-27B UD-Q4_K_M, greedy, thinking off, 512 tokens, median of 3)
| Job | no guesser | MTP (3 ahead) | DFlash 2 | YASAKANI fastest (Linux build) |
|---|---|---|---|---|
| new code | 76.5 | 186.5 | 285.7 | 597.2 |
| edit | 76.8 | 202.4 | 315.0 | 777.2 |
| explain | 76.9 | 127.1 | 118.8 | 278.7 |
| story | 76.6 | 113.5 | 123.8 | 236.5 |
| JSON | 76.6 | 207.1 | 346.2 | 1,182.4 |

## The 1,277 tok/s figure: YASAKANI on a code edit (`formats-harness/`)
The video's and the Shorts' 1,277 tok/s is the **edit** job of my model-formats harness, a different prompt from the "edit"
row above. It asks the model to fix an off-by-one bug and return the complete corrected file, so most of the reply is copied
from the prompt: the best case for any guesser. Same RTX 5090, 512 tokens, greedy, thinking off, median of 3:

| Engine (formats-harness row) | prose | code | JSON | edit |
|---|---|---|---|---|
| llama.cpp, no guesser (`gguf-A`) | 77.6 | 78.0 | 76.8 | 76.4 |
| llama.cpp + MTP (`gguf-mtp`) | 127.8 | 172.5 | 194.6 | 208.7 |
| ExLlamaV3 + DFlash 2 (`exl3-dflash2`) | 139.1 | 226.8 | 305.4 | 409.3 |
| TensorFold + DFlash 2 (`tf-dflash2-32k`) | 243.7 | 414.2 | 405.7 | 1,569.3 |
| **YASAKANI fastest, my Linux build (`ysk-speed`)** | **269.3** | **446.8** | **629.4** | **1,277.3** |
| YASAKANI quality, my Linux build (`ysk-quality`) | 251.2 | 424.4 | 511.6 | 1,190.7 |

YASAKANI's three edit runs were 1,268.9, 1,284.2 and 1,277.3 tok/s (512 tokens in about 0.40 s), and its reply was identical,
character for character, to plain llama.cpp's, which ran at 76.1 tok/s in the same session (`gguf-A-r2`). The developer's
own figure, about 1,222 tok/s, comes from his edit test on his Windows PC: a different prompt, so the two aren't a
like-for-like comparison. Raw runs: `formats-harness/results/ysk-speed/speed.jsonl`; quality suite (HumanEval 19/20, GSM8K
18/20, MMLU 160/200, JSON 3/3; the normal Q4_K_M model 19, 18, 163, 3): `formats-harness/results/ysk-speed-q/`.

Everything else (the small 0.8B and 2B guessers, n-gram, the Llama 8B and 70B runs, the 3080 Ti, a 16 GB cap, the 9B,
sampled settings, thinking mode, VRAM costs, how far ahead to guess) is in `bench/RESULTS.md`; the YASAKANI runs, its quality
check and the comparison with TensorFold are in `bench/YASAKANI.md`.

## What you need
- Linux (I used Ubuntu 24.04 under WSL2), an NVIDIA GPU, Python 3.10+ with `requests` (`pip install requests`).
- llama.cpp b12189 built with CUDA (`LLAMA_SERVER` / `LLAMA_BENCH` point at it; the default is `~/src/llama.cpp-b12189/build/bin/`).
- The model files: `download.sh` fetches every GGUF at the exact revision in `model_sources.json` into `$MODELS/specdec`
  (default `~/models`) and checks each SHA-256. Llama 3.1 8B and 70B come from Ollama (`ollama pull llama3.1:8b llama3.1:70b`);
  `bench_llama.py` reads Ollama's blob files directly. The RTX 3080 Ti box was set up with `nuc/setup.ps1` (Windows).
- For the YASAKANI rows: `yasakani-linux/install.sh`, then one model folder in `~/models` (see the end of that script).

## Rerun
- `bench/bench.py <row> [--runs 3]` starts llama-server (or `yasakani serve`) for a row, warms up, runs the five prompts in
  `bench/prompts.json` and writes `bench/results/<row>/row.json` (every streamed chunk with its arrival time, the text, and
  the server's draft counters). `bench.py --list` shows the rows; model paths are at the top of the file.
- `bench/queue8.sh` is the full quiet-machine run behind the video; `bench/queue_ysk.sh` and `queue_ysk2.sh` are the
  YASAKANI runs. Run a quiet check first (`llama-bench -p 0 -n 256` should give ~78 tok/s for this model on a 5090): with
  Windows apps open, everything ran ~9% slow.
- `bench/spread_check.py` flags rows whose repeat runs disagree by more than 5%.
- `replay/race_data.py` makes the replay files the video's races are drawn from.
- `wsl-vs-win/README.md`: how much WSL2 costs (at most ~5%: the GPU was busy 95.5% of each token).

## YASAKANI on Linux
YASAKANI (github.com/yasakani-dev/yasakani, Apache-2.0, by its developer; I'm not affiliated) supports only Windows 11. Its
source already has the Linux code paths; two Windows-only file calls in `engine35.cpp` (`_fseeki64`, `_ftelli64`) map to
`fseeko` / `ftello` with one compiler flag, so the source stays untouched. `yasakani-linux/install.sh` clones it, builds
`libyskn.so` with CUDA 13.3 and installs its Python package; `yasakani selfcheck` then reproduced the developer's reference
output on all three prompts. Built and tested under WSL2 on an RTX 5090 only: at commit 758ef28 (the runs in this folder) and
again from a fresh clone at 3d436bf (14 Nov 2026), where the self-check also passed.
- `formats-harness/` is the harness from my model-formats video (four 512-token jobs plus prefill and a quick quality
  suite). The YASAKANI rows (`ysk-*`), the llama.cpp anchor and the comparison rows (TensorFold, ExLlamaV3 + DFlash 2,
  llama.cpp + MTP) behind the 1,277 tok/s edit figure and the quality check are in `formats-harness/results/`.
