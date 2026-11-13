# YASAKANI head-to-head (13 Nov 2026, 17:38-18:05 UTC, quiet machine)

YASAKANI v0.1.1 (github.com/yasakani-dev/yasakani @758ef28), built for Linux under WSL2 by
`../yasakani-linux/install.sh`; `yasakani selfcheck` passed (bit-identical to the author's Windows reference).
Its own compressed Qwen3.8-27B files (fastest 10.3 GB, quality 12.4 GB) + its retrained DFlash 2 drafter, 32K context,
loop stopper off. Quiet check: llama-bench tg256 median 78.0; same-session llama.cpp anchor 27b-none-ysk-session 75.7-76.0
tok/s with replies identical (sha) to the afternoon quiet re-run. Queues: queue_ysk.sh, queue_ysk2.sh (the first queue's
later rows hit "address already in use": a YASAKANI server holds its port a few seconds after it stops; re-run with a wait).

## This video's harness (bench.py: 5 tasks, greedy, 512 tokens, thinking off, median of 3; tok/s from usage counts)

| Task | llama.cpp none | llama.cpp MTP n3 | llama.cpp DFlash 2 | YASAKANI fastest | YASAKANI quality |
|---|---|---|---|---|---|
| code | 76.5 | 186.5 | 285.7 | 597.2 | 606.6 |
| edit | 76.8 | 202.4 | 315.0 | 777.2 | 766.1 |
| explain | 76.9 | 127.1 | 118.8 | 278.7 | 248.0 |
| story | 76.6 | 113.5 | 123.8 | 236.5 | 221.4 |
| json | 76.6 | 207.1 | 346.2 | 1,182.4 | 1,132.5 |

YASAKANI fastest vs llama.cpp DFlash 2: 2.1x code, 2.5x edit, 2.3x explain, 1.9x story, 3.4x JSON (vs no guesser: 7.8x
code, 15.4x JSON, 3.1x story). Its replies differ from the GGUF's except JSON (identical text): the weights are compressed
differently. VRAM +15.2 GiB at 32K; load 13 s.

## Model-formats harness (`../formats-harness`: decode tok/s, median of 3)

| Row | prose | code | json | edit | prefill 8K |
|---|---|---|---|---|---|
| gguf-A (no drafting) | 77.6 | 78.0 | 76.8 | 76.4 | 3,883 |
| gguf-A-r2 (same session) | 76.5 | 76.3 | 76.3 | 76.2 | - |
| gguf-mtp | 127.8 | 172.5 | 194.6 | 208.7 | 3,342 |
| exl3-dflash2 | 139.1 | 226.8 | 305.4 | 409.3 | 2,867 |
| tf-dflash2-32k (TensorFold) | 243.7 | 414.2 | 405.7 | 1,569.3 | 4,783 |
| **ysk-speed** | 269.3 | 446.8 | 629.4 | 1,277.3 | 3,907 |
| **ysk-quality** | 251.2 | 424.4 | 511.6 | 1,190.7 | 3,940 |

Quality suite (HumanEval 20 / GSM8K 20 / MMLU 200 / JSON 3, greedy, thinking off): ysk-speed 19/18/160/3, ysk-quality
19/19/162/3; GGUF Q4_K_M 19/18/163/3, EXL3 18/18/160/3, TensorFold 19/18/161/3: no measurable loss on these small sets.
His self-reported 1,222 tok/s code-edit figure matches the kind of number measured here (edit 1,277); TensorFold is still
faster on that edit test (1,569), YASAKANI is faster on prose, code and JSON.

Caveats: WSL2 build, not his Windows DLL (self-check identical; WSL2 cost llama.cpp <= ~5%, ../wsl-vs-win); 512-token
replies; one machine.
