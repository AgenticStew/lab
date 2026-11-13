#!/usr/bin/env bash
# Every RTX 5090 row in results/, run on a quiet machine (Windows apps closed: with them open, every decode ran ~9% slow).
# Run order matters only for the 70B (last: its 42 GB load empties the page cache).
set -uo pipefail
cd "$(dirname "$0")"
T27=$(python3 -c "import bench; print(bench.T27)")
LB=${LLAMA_BENCH:-~/src/llama.cpp-b12189/build/bin/llama-bench}
date -Is; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
# 0. quiet check: pure decode should be ~78 tok/s
$LB -m "$T27" -ngl 99 -fa 1 -p 0 -n 256 -r 3 -o json > results/quiet_check.json 2>/dev/null
python3 -c "import json; d=json.load(open('results/quiet_check.json'))[0]; print('QUIET CHECK tg256', round(d['avg_ts'],1), [round(x,1) for x in d['samples_ts']])"
# 1. verify cost: one pass with 1 to 32 tokens
$LB -m "$T27" -ngl 99 -fa 1 -p 1,2,3,4,5,6,7,8,12,16,32 -n 32 -r 5 -o json > results/verify_cost_27b.json 2> results/verify_cost_27b.log && echo "verify cost done"
# 2. the main grid: no guesser, MTP, DFlash 2
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --runs 3
# 3. how far ahead to guess (MTP 1 to 6)
python3 bench.py 27b-mtp-n1 27b-mtp-n2 27b-mtp-n4 27b-mtp-n6 --runs 1
# 4. the small 0.8B guesser, and the n-gram copy guesser on a fresh server per run
python3 bench.py 27b-d08-n8 --runs 3
python3 bench.py 27b-ngram-simple --fresh --runs 3
# 5. Qwen's recommended chat settings, and thinking mode
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --sampling qwen-instruct --suffix=-t07 --runs 3 --prompts code,explain,story
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --sampling qwen-thinking --suffix=-think --runs 2 --prompts code,explain --max-tokens 1024
# 6. the 9B with MTP, and Llama 3.1 8B + Llama 3.2 1B
python3 bench.py 9b-none 9b-mtp-n2 --runs 3
python3 bench_llama.py l8-none l8-d1-n12 --runs 3
# 7. the 27B squeezed into 16 GB
python3 bench.py 27b-16gb-none 27b-16gb-mtp-n3 --runs 2 --prompts code,explain,story
# 8. Llama 3.1 70B, too big for the card (last: slowest)
python3 bench_llama.py l70-none-ngl52 l70-d1-n8-ngl50 --runs 3 --prompts code,explain
date -Is
echo QUEUE8_DONE
