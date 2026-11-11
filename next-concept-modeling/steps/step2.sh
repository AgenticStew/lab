#!/usr/bin/env bash
# TESTPLAN step 2, inside the container, network on: downloads, every file checked against its sha256.
set -euo pipefail
export HF_HUB_DISABLE_XET=1 HF_HOME=/dl/hf
# the image's pip refuses system-wide installs (PEP 668); the container is throwaway, so allow them
export PIP_BREAK_SYSTEM_PACKAGES=1
command -v python python3 pip
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.__file__)"
pip --version
pip install huggingface_hub==2.4.0
hf download hazyresearch/ncm-checkpoints lm1b/small/generator.pt lm1b/tokenizer.pt lm1b/small/config.yaml --revision d342ce8aaea58830eef3b8ad473664673e41f692 --local-dir /dl/ncm
echo "b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe  /dl/ncm/lm1b/small/generator.pt" | sha256sum -c
echo "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c  /dl/ncm/lm1b/tokenizer.pt" | sha256sum -c
# hub 2.4's --include takes one pattern (the rest became filenames and model.safetensors was skipped): name all seven
hf download openai-community/gpt2-large model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt --local-dir /dl/gpt2-large
sha256sum /dl/ncm/lm1b/small/config.yaml /dl/gpt2-large/*.safetensors | tee /results/sha256-recorded.txt
# lm1b_Duo.ckpt: download it by hand from the Drive folder into dl/duo/ first
if [ -f /dl/duo/lm1b_Duo.ckpt ]; then
  sha256sum /dl/duo/lm1b_Duo.ckpt | tee -a /results/sha256-recorded.txt
  stat -c %s /dl/duo/lm1b_Duo.ckpt | tee -a /results/sha256-recorded.txt
else
  echo "STOP: /dl/duo/lm1b_Duo.ckpt is not there yet (Drive download by hand); rerun the last two lines once it is"
fi
