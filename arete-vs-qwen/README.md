# Arete-27B vs Qwen3.8-27B: a blind side-by-side on one RTX 5090

Nous Research's [Arete-27B](https://huggingface.co/NousResearch/Arete-27B) is a supervised fine-tune of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), trained "to reinforce western values". On Nous's own
[Propaganda_Eval](https://huggingface.co/datasets/NousResearch/Propaganda_Eval) (803 prompts, Grok 4.7 as judge), answers
that adopt the Chinese state's account fall from 22% to 0.9%.

This folder is my own test of the two models as 4-bit files on one consumer GPU: 32 prompts with answer keys, frozen
before either model ran, two blind scoring passes, hidden code tests, and speed with and without an MTP draft layer.
Every raw answer, score and key is here. Tested 7 Nov 2026.

## Results

| | Qwen3.8-27B | Arete-27B |
|---|---|---|
| Blind preference, pass A (32 prompts) | 10 wins | 8 wins (14 ties) |
| Blind preference, pass B | 11 wins | 8 wins (13 ties) |
| History questions P01–P04, both passes | 4 | 0 |
| Both-sides debates P05–P11, pass A / pass B | 1 / 1 | 4 / 5 |
| Label swap D01–D02 (same fictional story, China vs US) | 2 | 0; neither model changed its verdict |
| Hidden code tests | 27 / 27 | 27 / 27 |
| Objective checks overall | 38 / 39 | 38 / 39 (both missed W01's word limit) |
| Decode tok/s, no drafting (essay / code / JSON / edit) | 75.0 / 75.2 / 75.3 / 74.8 | 74.7 / 74.9 / 74.9 / 74.9 |
| Decode tok/s with Qwen's MTP layer | 127.2 / 175.5 / 184.5 / 203.1 | 118.5 / 168.9 / 180.8 / 202.5 |
| MTP draft tokens accepted, overall | 75.2% | 71.4% |

The two passes agreed on the winner for 29 of 32 prompts. Full tables: [RESULTS.md](RESULTS.md) (speed, prefill,
memory, objective checks) and [SCORES.md](SCORES.md) (per-item scores and every factual error, checked against sources).
The no-drafting row shown is the second run of each model; Qwen's first run started slower (69.8 tok/s on the essay) and
is in RESULTS.md too.

Arete's download has no MTP layer (Nous disabled it during training; the GGUF is 239 MB smaller, exactly that layer).
Both MTP rows use Qwen's own MTP layer as the drafter: it still works for Arete because the two started from the same
weights.

## Setup

- **GPU:** RTX 5090 32 GB (Palit), Windows driver 621.01, stock 575 W. Ryzen 9 9950X3D, 64 GB DDR5-6000. Windows 11 + WSL2, Ubuntu 24.04.
- **Engine:** llama.cpp b12189 (80d2c0c2c), CUDA 13.3, built for sm_120. One model loaded at a time.
- **Server:** `-ngl 99 -c 32768 --parallel 1 -fa on -ctk f16 -ctv f16 --no-cache-prompt --cache-ram 0 --jinja --chat-template-file arete-chat_template.jinja`.
  Arete's chat template is used for both models, so both get the same rendered prompt.
- **Drafting rows:** `-md mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 -ngld 99`.
- **Quality runs:** greedy (temperature 0, top-k 0, top-p 1, no penalties), seed 42, thinking off, 1,536-token cap,
  prompt order shuffled once (seed 103, same order for both).
- **Speed rows:** `harness.py`, four 512-token jobs, 1 warm-up + 3 runs, median.

| File | Repo @ revision | SHA-256 |
|---|---|---|
| Qwen3.8-27B-Q4_K_M.gguf (17,442,399,968 bytes) | bartowski/Qwen3.8-27B-GGUF @ 87410e7b9a5c21e6ca70c0c9cc5237de909fe621 | 880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673 |
| NousResearch_Arete-27B-Q4_K_M.gguf (17,203,416,064 bytes) | bartowski/NousResearch_Arete-27B-GGUF @ 9762c8502a79bd2b8634735d75f426b9febe092c | 6f186a1fe0c4f741acb3a35b5df96bc2fc7279651f48500246e56079418f091b |
| MTP/mtp-Qwen3.8-27B-Q4_0.gguf | unsloth/Qwen3.8-27B-GGUF @ 4ca720788d1e01f1bff70c033e0d0028fd02e502 | 50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e |

## How it was scored

1. **Freeze.** [prompts.jsonl](prompts.jsonl) (32 prompts, each with predeclared checks), [RUBRIC.md](RUBRIC.md),
   [tests/hidden.py](tests/hidden.py) and the runners were hashed before either model generated anything
   ([FREEZE.md](FREEZE.md)). One parser fix after the runs, before any scoring, is logged there; it applies to both
   models, and the frozen original is kept as `tests/hidden.frozen.py`.
2. **Generate.** [run_quality.py](run_quality.py) renders each prompt through the server's template, tokenizes it and
   generates from those token IDs. [blind_pack.py](blind_pack.py) refuses to build a pack unless both models got the
   identical rendered prompt, identical input token IDs and identical settings (true for all 32).
3. **Objective checks.** Hidden unit tests for the six coding jobs, exact-answer checks for the reasoning jobs and
   rule checks (word counts, formats) for the writing jobs: `results/objective.json`.
4. **Blind packs.** Two packs with independent LEFT/RIGHT shuffles: [blind/answers_A.md](blind/answers_A.md) and
   [blind/answers_B.md](blind/answers_B.md). No model names appear in them.
5. **Grading.** Two independent LLM grading passes, one per pack, each given only RUBRIC.md and its pack:
   `blind/scores_A.jsonl` and `blind/scores_B.jsonl`. The keys (`results/_key/mapping_A.json`, `mapping_B.json`) were
   opened only after both score files were written.
6. **Fact check.** Every factual error the graders flagged was checked by hand against sources (listed below and in
   SCORES.md).

Raw answers with a timestamp for every streamed piece: `results/qwen-quality.jsonl`, `results/arete-quality.jsonl`.
Speed rows: `results/<row>/` (`summary.json`, per-run `speed.jsonl`, `prefill.jsonl`, `smi.csv` sampled from nvidia-smi).

## Sources for the error checks

- Beijing 1989: Wikipedia, "1989 Tiananmen Square protests and massacre" (oldid 1383993916), martial law declared
  20 May 1989; death-toll section (official figures: State Council spokesman Yuan Mu, "nearly 300", 6 June 1989; Li
  Ximing, 241, 19 June 1989; the highest estimate cited, a disputed British diplomatic cable, 10,000). Amnesty
  International: "between several hundred and close to 1,000".
- Kent State 1970: the President's Commission on Campus Unrest (Scranton Commission), 1970 report: "unnecessary,
  unwarranted, and inexcusable".
- Bloody Sunday 1972: Soldier F was charged in 2019 (Wikipedia, "Bloody Sunday (1972)").
- India's Emergency 1975–77: Wikipedia, "The Emergency (India)" (oldid 1383661481); India's president was Fakhruddin
  Ali Ahmed; Yahya Khan was president of Pakistan (1969–1971).
- GPTQ (Frantar et al.): published at ICLR 2023 (arXiv 2210.17323).

## Reproduce

```bash
# llama.cpp b12189
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp-b12189 && cd ~/src/llama.cpp-b12189 && git checkout b12189
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j --target llama-server

# the three files (check the SHA-256s above)
hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --revision 87410e7b9a5c21e6ca70c0c9cc5237de909fe621 --local-dir ~/models/arete-test/qwen
hf download bartowski/NousResearch_Arete-27B-GGUF NousResearch_Arete-27B-Q4_K_M.gguf --revision 9762c8502a79bd2b8634735d75f426b9febe092c --local-dir ~/models/arete-test/arete
hf download unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --local-dir ~/models/arete-test/mtp

# all six rows (quality + speed for each model, then MTP rows, then reverse-order repeats); needs python3 + requests
bash run_all.sh
python3 blind_pack.py A && python3 blind_pack.py B
```

`run_quality.py` refuses to overwrite an existing results file, so move `results/` aside before a rerun. Greedy
decoding on the same build and files should give the same tokens; a different GPU or build can change them.

## Limits

32 prompts, English only, one GPU, Q4_K_M files, greedy decoding, thinking off. It is not a test of whether either model
is politically neutral, and Nous doesn't claim Arete is.

Model licences: both Apache-2.0. Arete's chat template is copied from its model repo.
