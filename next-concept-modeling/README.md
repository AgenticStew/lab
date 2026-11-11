# NCM vs DUO on an RTX 5090

A rerun of the speed and quality comparison in "Next-Concept Modeling" (arXiv 2610.14162, Hazy Research): NCM against
DUO on LM1B, both from their released checkpoints, on one RTX 5090. The authors report NCM 21x faster than DUO on an
H100. This folder holds the code, the raw logs and the results behind the video.

## What was measured

| table | model | precision | KV cache | CUDA graphs | median wall s | passes/batch |
|---|---|---|---|---|---|---|
| A | NCM, eager (as released) | fp32 | off | off | 9.03 | 11 |
| A | DUO, SDPA attention | fp32 | off | off | 225.37 | 129 |
| B | NCM, KV cache via PR 5 | bf16 | on | on | 5.65 | 12 |
| B | DUO, SDPA attention | bf16 | off | on | 159.37 | 129 |
| C | NCM, penalties off | bf16 | on | on | 4.30 | 12 |

1,024 sequences, batch 16, 128 positions, 1 warm-up batch, median of 3 repeats, seed 42.
NCM over DUO: 24.9x (Table A), 28.2x (Table B).

Quality (generative perplexity under GPT-2 large, NCM's eval_merge.py scorer, 5 seeds x 256 samples, fp32):

| run | gen PPL | entropy |
|---|---|---|
| NCM, released penalties | 93.64 | 4.32 |
| NCM, penalties off | 78.79 | 4.12 |
| DUO, ancestral (default) | 100.26 | 4.30 |
| DUO, greedy | 87.96 | 4.18 |

Full tables, caveats and the output checks: `RESULTS.md`. Every number comes from `results.json` and `logs/`.

## Setup

- RTX 5090 32 GB, Ryzen 9 9950X3D, 64 GB DDR5-6000, Windows 11 + WSL2 Ubuntu 24.04
- Container `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` (PyTorch 2.10.0+cu128, Python 3.12.3); `logs/env.txt`,
  `logs/pip-freeze.txt`
- NCM: github.com/HazyResearch/ncm at c702742 plus the PR 5 KV-cache change (`apply_patch.py`), checkpoints
  huggingface.co/hazyresearch/ncm-checkpoints at d342ce8
- DUO: github.com/s-sahoo/duo at 7c9b498, attention on SDPA (`duo_sdpa_patch.py`), LM1B checkpoint `lm1b_Duo.ckpt`
  from the folder linked by github.com/david3684/flm (download by hand into `dl/duo/`)
- Recorded checksums: `logs/sha256-recorded.txt`, `logs/src-sha256-check.txt`

## Running it

Copy this folder to `bench/` inside an empty working folder and run from that folder. `bench/run_container.sh`
starts a throwaway container with only `dl/`, `results/` and `bench/` (read-only) mounted, then runs
`steps/step2.sh` to `steps/step6.sh` inside it:

- `bash bench/run_container.sh phase1`: downloads, installs and dry runs (network on)
- `bash bench/run_container.sh phase2`: cuts the network, then the timed rows and the quality rows, and stops the
  container

The checkpoints load only after the network is cut. `bench_ncm_duo.py` is the timing harness, `make_configs.py` writes
the per-row configs, `check_outputs.py` and `compare_ids.py` produce the output checks, and `write_results_md.py`
builds `RESULTS.md`.

## Licences

`duo_sdpa_patch.py` reproduces one small helper (`apply_rotary_emb_torch`) from flash-attn 2.x, which is BSD-3-Clause:
see `LICENSE-flash-attn`. NCM is MIT and DUO is Apache-2.0; neither repo's code is copied here, the steps fetch both
at the pinned commits.

"TESTPLAN step N" in the comments is the video's test plan: step 1 starts the container, steps 2-6 are `steps/step2.sh` to `steps/step6.sh`.
