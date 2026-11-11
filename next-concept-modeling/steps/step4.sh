#!/usr/bin/env bash
# TESTPLAN step 4, network still on: configs, then the dry runs on random weights (no checkpoint is unpickled).
# Deviation (c): eval.ref_model points at the downloaded gpt2-large in both copies.
set -euo pipefail
export HF_HOME=/dl/hf
python -I /bench/make_configs.py /dl/ncm/lm1b/small/config.yaml /work/cfg --set batch_size=16 --set eval.ref_model=/dl/gpt2-large --name lm1b_b16_pen.yaml
python -I /bench/make_configs.py /dl/ncm/lm1b/small/config.yaml /work/cfg --set batch_size=16 --set eval.ref_model=/dl/gpt2-large --set generation.frequency_penalty=0 --set generation.frequency_deficit=0 --set "generation.adjacency_penalty=[]" --name lm1b_b16_nopen.yaml
diff /dl/ncm/lm1b/small/config.yaml /work/cfg/lm1b_b16_pen.yaml | tee /results/config-diff-pen.txt || true
diff /dl/ncm/lm1b/small/config.yaml /work/cfg/lm1b_b16_nopen.yaml | tee /results/config-diff-nopen.txt || true
python -I /bench/bench_ncm_duo.py --dry-run --model ncm --cfg /work/cfg/lm1b_b16_pen.yaml --n 32 --batch 16 --seq-len 128 --seed 42 2>&1 | tee /results/dryrun-ncm.txt
python -I /bench/bench_ncm_duo.py --dry-run --model duo --ckpt /dl/duo/lm1b_Duo.ckpt --n 32 --batch 16 --seq-len 128 --steps 128 --seed 42 2>&1 | tee /results/dryrun-duo.txt
echo "Dry runs done. Check results/dryrun-*.txt (a file new in HF_HOME needs your OK); then on the host: docker network disconnect bridge ncm-bench"
