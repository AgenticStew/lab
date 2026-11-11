# RESULTS: NCM vs DUO on the RTX 5090 (phase 2, run 4, 10 Nov 2026)

Written by `bench/write_results_md.py` from `results/results.json` (copied to `bench/logs/results.full.json`) and the
root `results.json` (slots, derived values with their expressions). Raw logs: `bench/logs/`.

## Timed rows (1,024 sequences, batch 16, 128 positions, 1 warm-up batch, median of 3 repeats, seed 42)

| table | model | precision | KV cache | CUDA graphs | median wall s | passes/batch | ms/pass (median) | outside-pass share % |
|---|---|---|---|---|---|---|---|---|
| A | NCM, eager (as released) | fp32 | off | off | 9.033 | 11 | 7.14 | 38.0 |
| A | DUO, SDPA attention | fp32 | off | off | 225.371 | 129 | 13.20 | 51.4 |
| B | NCM, KV cache via #5 | bf16 | on | on | 5.649 | 12 | 3.07 | 59.6 |
| B | DUO, SDPA attention | bf16 | off | on | 159.368 | 129 | 5.45 | 71.1 |
| C | NCM, penalties off (bf16, KV, graphs) | bf16 | on | on | 4.298 | 12 | 3.06 | 47.0 |

- NCM ÷ DUO wall-clock: Table A 24.9×, Table B 28.2× (authors, H100: 21×). B vs A ratio change 13%.
- Passes: DUO 129 ÷ NCM 12 = 10.75×; a DUO pass costs 1.8× an NCM pass (bf16 + graphs). DUO spends 71% of its time outside its passes (NCM 60%), which is why the total exceeds passes × pass cost.
- NCM counted 11 passes per batch in Table A (eager) and 12 in Table B (KV cache + graphs).
- Settings gain: NCM 1.6×, DUO 1.4×. Every repeat produced identical ids within its row.
- Penalties off: NCM bf16 time 5.65 s to 4.30 s (24% of NCM's time).

## Identical outputs, Table A vs Table B (NCM)

0.0% of sequences identical; 29.9560546875% of token positions identical. Caveat: both tables
used the same --seed, but NCM samples its codes at every level, so any numeric difference between fp32 eager and
bf16 + KV cache + graphs (and a different pass count, 11 vs 12) sends the random draws apart. This measures
path divergence under sampling, not bf16 error on a fixed draw. The quality scores below were taken at fp32 (as R7 says).

## Quality (generative perplexity, GPT-2 large, genppl_flm via NCM's eval_merge.py; 5 seeds × 256 samples; fp32)

| run | gen PPL mean | ± sd (seeds) | entropy |
|---|---|---|---|
| NCM, released penalties | 93.64 | 1.79 | 4.3161 |
| NCM, penalties off | 78.79 | 0.64 | 4.1189 |
| DUO, ancestral (default) | 100.26 | 2.07 | 4.3031 |
| DUO, greedy | 87.96 | 1.79 | 4.1788 |

Paper (H100): NCM 93.05 / H 4.31, DUO 97.30 / H 4.30. NCM landed close; DUO ancestral landed about 3% higher than
the paper's DUO figure. Under greedy sampling DUO beats NCM on gen PPL with lower entropy.

## Output checks (bench/check_outputs.py)

| file | samples | 8+ token run | dominant 4-gram | distinct-1 (seed 42) | distinct-2 (seed 42) |
|---|---|---|---|---|---|
| ncm-pen | 1280 | 0 | 0 | 0.078 | 0.506 |
| ncm-nopen | 1280 | 45 | 4 | 0.070 | 0.462 |
| duo-ancestral | 1280 | 2 | 0 | 0.198 | 0.712 |
| duo-greedy | 1280 | 1 | 0 | 0.159 | 0.631 |

NCM's pickles show every `hyps` string empty: eval_shard.py's `dec()` cuts at the first EOS, and BERT's [CLS] (id 101)
is both the EOS and the first token of every packed LM1B sample. The scorer (genppl_flm) decodes `raw_ids` with
specials kept, as it does for DUO, and the raw ids hold real packed text, so the scores are unaffected (FACTCHECK
C038). The empty-string counts in output_checks are that artefact, not empty generations.
Penalties off: 45 of 1,280 NCM samples hold a run of 8+ identical tokens (0 with penalties on).
Distinct-1 over a seed is lower for NCM than DUO; it is not in the script.
