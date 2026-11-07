# RESULTS: Arete-27B vs Qwen3.8-27B on one RTX 5090 (7 Nov 2026)

**Setup:**
- llama.cpp b12189 (80d2c0c2c), built for sm_120 with CUDA 13.3.
- Bartowski Q4_K_M files (SHA-256-verified), one model loaded at a time.
- 32K context, F16 KV, flash attention on, one slot.
- `--no-cache-prompt --cache-ram 0`; Arete's chat template used for both; thinking off.
- Speed rows: the formats-video harness (`harness.py`, same 4 jobs, 512 tokens, 1 warm-up + 3 runs, median).
- MTP rows: the same Qwen MTP head file for both models: `mtp-Qwen3.8-27B-Q4_0.gguf`, `--spec-type draft-mtp`, n-max 3, p-min 0.1.

## Speed, decode tok/s (median of 3)

| Row | Essay | Code | JSON | File edit |
|---|---:|---:|---:|---:|
| Qwen, no drafting (ran 1st) | 69.8 | 71.1 | 74.9 | 74.9 |
| Arete, no drafting (ran 2nd) | 75.2 | 75.2 | 75.1 | 75.1 |
| Arete, no drafting (repeat, 5th) | 74.7 | 74.9 | 74.9 | 74.9 |
| Qwen, no drafting (repeat, 6th) | 75.0 | 75.2 | 75.3 | 74.8 |
| Qwen + its MTP layer | 127.2 | 175.5 | 184.5 | 203.1 |
| Arete + Qwen's MTP layer | 118.5 | 168.9 | 180.8 | 202.5 |

Qwen's first no-drafting row started slower (69.8 on the essay, rising to 74.9 by the last job); its repeat row ran 75.0. Read the no-drafting speed as a tie at about 75 tok/s.

## Draft acceptance (from the server's /metrics counters, 3 measured runs per job)

| Row | Essay | Code | JSON | File edit | Overall |
|---|---:|---:|---:|---:|---:|
| qwen-mtp | 48.7% | 79.2% | 87.1% | 100.0% | 75.2% |
| arete-mtp | 43.4% | 76.3% | 83.9% | 100.0% | 71.4% |

## Prefill (cold, unique nonce prefix)

| Row | 2K tok/s (TTFT) | 8K tok/s (TTFT) |
|---|---:|---:|
| qwen-A | 3,139 (0.64 s) | 3,570 (2.29 s) |
| arete-A | 2,981 (0.68 s) | 3,692 (2.21 s) |

## Memory (device total, nvidia-smi; Windows desktop included, about 2.6 GiB idle)

- qwen: 21207 MiB after load; 21,719 MiB peak during the speed and prefill rows
- arete: 21272 MiB after load; 21,096 MiB peak during the speed and prefill rows

## Objective checks (hidden tests, frozen)

- qwen: 38/39 overall; code 27/27 hidden tests; W01 word count 103 words (limit 80–100)
- arete: 38/39 overall; code 27/27 hidden tests; W01 word count 121 words (limit 80–100)

**Token cap (1,536) hit:** Qwen ['P05', 'P06', 'P07', 'P08', 'P11']; Arete ['C02', 'P07', 'P08', 'P11'].
**Input token IDs:** identical for all 32 prompts (blind_pack.py asserts it).
