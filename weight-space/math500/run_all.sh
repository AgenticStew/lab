#!/usr/bin/env bash
# The full test: MATH-500 (500 problems, 1 sample each, seed 0), base vs the p50c edit, thinking on and off.
# About 1 h 40 min on an RTX 5090. Set PY to the Python that has vLLM installed.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
for m in base edit; do
  echo "=== $m start $(date +%T)"
  $PY run_math.py --model $m --modes think,nothink --n 500 --mem 0.85 --max-seqs 50 --chunk 125 > logs_$m.log 2>&1
  echo "=== $m exit $? $(date +%T)"
  grep -E "SUMMARY|Error" logs_$m.log
done
echo ALL_DONE
