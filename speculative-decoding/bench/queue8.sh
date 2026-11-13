#!/usr/bin/env bash
# 13 Nov afternoon, take 2 (queue6 hit a leftover server; guard added to bench.run_row): re-run every 5090 number the videos use on a quiet machine (user closed the Windows apps).
# Morning results archived in results_s1_morning/ (that session ran ~9% slow: server 69.7 vs 75.8 re-run, llama-bench 77.9).
set -uo pipefail
cd "$(dirname "$0")"
T27=$(python3 -c "import bench; print(bench.T27)")
LB=~/src/llama.cpp-b12189/build/bin/llama-bench
date -Is; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
# 0. quiet check: pure decode should be ~78 tok/s
$LB -m "$T27" -ngl 99 -fa 1 -p 0 -n 256 -r 3 -o json > results/quiet_check.json 2>/dev/null
python3 -c "import json; d=json.load(open('results/quiet_check.json'))[0]; print('QUIET CHECK tg256', round(d['avg_ts'],1), [round(x,1) for x in d['samples_ts']])"
# 1. verify cost (frame 5 chart, Short frame 3)
$LB -m "$T27" -ngl 99 -fa 1 -p 1,2,3,4,5,6,7,8,12,16,32 -n 32 -r 5 -o json > results/verify_cost_27b.json 2> results/verify_cost_27b.log && echo "verify cost done"
# 2. the main grid (frames 1, 4, 13, 26; Short)
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --runs 3
# 3. guess-ahead sweep (frame 17)
python3 bench.py 27b-mtp-n1 27b-mtp-n2 27b-mtp-n4 27b-mtp-n6 --runs 1
# 4. the small 0.8B guesser (frame 14) and the n-gram copy guesser on fresh servers (frame 13)
python3 bench.py 27b-d08-n8 --runs 3
python3 bench.py 27b-ngram-simple --fresh --runs 3
# 5. real chat settings + thinking (frame 18)
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --sampling qwen-instruct --suffix=-t07 --runs 3 --prompts code,explain,story
python3 bench.py 27b-none 27b-mtp-n3 27b-df2-n7 --sampling qwen-thinking --suffix=-think --runs 2 --prompts code,explain --max-tokens 1024
# 6. the 9B (frame 23) and Llama 8B + 1B (frame 15)
python3 bench.py 9b-none 9b-mtp-n2 --runs 3
python3 bench_llama.py l8-none l8-d1-n12 --runs 3
# 7. 16 GB cap (frame 22)
python3 bench.py 27b-16gb-none 27b-16gb-mtp-n3 --runs 2 --prompts code,explain,story
# 8. the 70B (frame 24), last: slowest
python3 bench_llama.py l70-none-ngl52 l70-d1-n8-ngl50 --runs 3 --prompts code,explain
date -Is
( cd ../demo && python3 ~/video-studio/kits/explainer/tools/tmuxcap.py plan_bench.py take-bench3 ) > results/take_bench3.log 2>&1 && echo "take-bench3 done"
echo QUEUE8_DONE
