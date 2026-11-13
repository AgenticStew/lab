#!/usr/bin/env bash
# 13 Nov evening: YASAKANI (WSL2 build, self-check passed) head-to-head on a quiet machine (Windows apps closed).
# Same prompts and settings as the videos' llama.cpp rows; a same-session llama.cpp anchor in each harness.
set -uo pipefail
cd "$(dirname "$0")"
T27=$(python3 -c "import bench; print(bench.T27)")
LB=${LLAMA_BENCH:-~/src/llama.cpp-b12189/build/bin/llama-bench}
FMT="$(pwd)/../formats-harness"
date -Is; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
ss -ltn | grep -E ":(8091|8080|8765) " && { echo "a server port is busy: stop it first"; exit 1; }
# 0. quiet check (median of the reps: WSL clock jumps fake an occasional ~96 tok/s rep)
$LB -m "$T27" -ngl 99 -fa 1 -p 0 -n 256 -r 5 -o json > results/quiet_check_ysk.json 2>/dev/null
python3 -c "import json,statistics as s; d=json.load(open('results/quiet_check_ysk.json'))[0]; print('QUIET CHECK tg256 median', round(s.median(d['samples_ts']),1), [round(x,1) for x in d['samples_ts']])"
# 1. speculative-decoding prompts (5 tasks, greedy, 512 tokens, thinking off, 3 runs): anchor + both YASAKANI models
python3 bench.py 27b-none --runs 3 --suffix=-ysk-session
python3 bench.py ysk-speed ysk-quality --runs 3
# 2. model-formats prompts: anchor (speed only) + YASAKANI speed/prefill and quality suites
cd "$FMT"
.venv/bin/python row.py gguf-A-r2 --suite speed
.venv/bin/python row.py ysk-speed --suite speed,prefill
.venv/bin/python row.py ysk-speed-q --suite quality
.venv/bin/python row.py ysk-quality --suite speed,prefill
.venv/bin/python row.py ysk-quality-q --suite quality
date -Is; echo QUEUE DONE
