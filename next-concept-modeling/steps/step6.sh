#!/usr/bin/env bash
# TESTPLAN step 6, network off: the quality rows. Deviations: (a) eval_shard.py's real option names plus its
# required --out, (b) its pickles scored by the repo's own eval_merge.py, (g) every output checked for loops.
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_HOME=/dl/hf NCM_ROOT=/work/ncm
cd /work/ncm
python -I scripts/eval_shard.py --config /work/cfg/lm1b_b16_pen.yaml --checkpoint-dir /dl/ncm/lm1b/small --out /results/ncm-pen.pkl 2>&1 | tee /results/gen-ncm-pen.txt
python -I scripts/eval_merge.py --config /work/cfg/lm1b_b16_pen.yaml --glob /results/ncm-pen.pkl 2>&1 | tee /results/quality-ncm-pen.txt
B=/bench/bench_ncm_duo.py
python -I $B --model duo --ckpt /dl/duo/lm1b_Duo.ckpt --quality --scorer genppl_flm --ref /dl/gpt2-large --samples 256 --seeds 42 0 1 2 3 --batch 16 --seq-len 128 --steps 128 --precision fp32 --kv-cache off --graphs off --noise-removal ancestral --count-truncated --log-attn --table Q --out /results/results.json 2>&1 | tee /results/row-Q-duo-ancestral.txt
python -I $B --model duo --ckpt /dl/duo/lm1b_Duo.ckpt --quality --scorer genppl_flm --ref /dl/gpt2-large --samples 256 --seeds 42 0 1 2 3 --batch 16 --seq-len 128 --steps 128 --precision fp32 --kv-cache off --graphs off --noise-removal greedy --count-truncated --log-attn --table Q --out /results/results.json 2>&1 | tee /results/row-Q-duo-greedy.txt
python -I scripts/eval_shard.py --config /work/cfg/lm1b_b16_nopen.yaml --checkpoint-dir /dl/ncm/lm1b/small --out /results/ncm-nopen.pkl 2>&1 | tee /results/gen-ncm-nopen.txt
python -I scripts/eval_merge.py --config /work/cfg/lm1b_b16_nopen.yaml --glob /results/ncm-nopen.pkl 2>&1 | tee /results/quality-ncm-nopen.txt
python -I /bench/check_outputs.py /results/results.json /results/ncm-pen.pkl /results/ncm-nopen.pkl /results/duo-ancestral.pkl /results/duo-greedy.pkl | tee /results/output-checks.txt
