#!/usr/bin/env bash
# TESTPLAN step 5, network off: the timed rows, one per call, no GPU tools running alongside.
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_HOME=/dl/hf
if python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)" 2>/dev/null; then
  echo "STOP: the network is still on"
  exit 3
fi
B=/bench/bench_ncm_duo.py
python -I $B --model ncm --cfg /work/cfg/lm1b_b16_pen.yaml --n 1024 --batch 16 --seq-len 128 --precision fp32 --kv-cache off --graphs off --penalties config --l0 unigram --seed 42 --warmup 1 --repeats 3 --count-passes --save-ids --table A --out /results/results.json 2>&1 | tee /results/row-A-ncm.txt
python -I $B --model duo --ckpt /dl/duo/lm1b_Duo.ckpt --n 1024 --batch 16 --seq-len 128 --steps 128 --precision fp32 --kv-cache off --graphs off --noise-removal ancestral --seed 42 --warmup 1 --repeats 3 --count-passes --log-attn --table A --out /results/results.json 2>&1 | tee /results/row-A-duo.txt
python -I $B --model ncm --cfg /work/cfg/lm1b_b16_pen.yaml --n 1024 --batch 16 --seq-len 128 --precision bf16 --kv-cache on --kv-source pr5 --graphs on --penalties config --l0 unigram --seed 42 --warmup 1 --repeats 3 --count-passes --save-ids --table B --out /results/results.json 2>&1 | tee /results/row-B-ncm.txt
python -I $B --model duo --ckpt /dl/duo/lm1b_Duo.ckpt --n 1024 --batch 16 --seq-len 128 --steps 128 --precision bf16 --kv-cache off --graphs on --noise-removal ancestral --seed 42 --warmup 1 --repeats 3 --count-passes --log-attn --table B --out /results/results.json 2>&1 | tee /results/row-B-duo.txt
python -I $B --model ncm --cfg /work/cfg/lm1b_b16_nopen.yaml --n 1024 --batch 16 --seq-len 128 --precision bf16 --kv-cache on --kv-source pr5 --graphs on --penalties off --l0 unigram --seed 42 --warmup 1 --repeats 3 --count-passes --table C --out /results/results.json 2>&1 | tee /results/row-C-ncm.txt
python -I /bench/compare_ids.py /results/results.json --table-a A --table-b B --model ncm | tee /results/ncm-identical.txt
