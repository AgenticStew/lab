# Lab

The code and raw results behind the tests I run for my YouTube videos, one folder per video (newest first). Each
folder says what was run, on what hardware, and how to rerun it.

| Folder | Video |
|---|---|
| [`speculative-decoding/`](speculative-decoding/) | Speculative decoding: four kinds of guesser tested on an RTX 5090 and an RTX 3080 Ti, plus YASAKANI (built for Windows) built and benchmarked on Linux, Ubuntu under WSL2 (Nov 2026) |
| [`next-concept-modeling/`](next-concept-modeling/) | Stanford's Next-Concept Modeling claims 21x faster text generation than DUO. I reran NCM against DUO on an RTX 5090 (Nov 2026) |
| [`arete-vs-qwen/`](arete-vs-qwen/) | Nous retrained Qwen3.8-27B to drop China's party line. I tested Arete-27B against the original, blind, on an RTX 5090 (Nov 2026) |
| [`weight-space/`](weight-space/) | GPT-6 Astra rewired a small AI's weights with no training. I rebuilt the edit and reran it on an RTX 5090 (Nov 2026) |

The Local AI Engineer course has its own repo: https://github.com/AgenticStew/local-ai-engineer-course
