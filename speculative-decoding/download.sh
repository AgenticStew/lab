#!/usr/bin/env bash
# Downloads every model file the speculative-decoding test uses, pinned to the revisions in model_sources.json,
# then checks each file's sha256 against Hugging Face's LFS hash.
set -euo pipefail
HF=${HF:-hf}                                  # huggingface_hub CLI
OUT=${MODELS:-$HOME/models}/specdec
export HF_HUB_DISABLE_XET=1
dl() { # repo rev file
  HF_HUB_DISABLE_XET=1 $HF download "$1" "$3" --revision "$2" --local-dir "$OUT/$(basename $1)"
}
# small files first so the drafts are testable early
dl unsloth/Qwen3.5-0.8B-GGUF 6ab4614 Qwen3.5-0.8B-Q8_0.gguf
dl z-lab/Qwen3.8-27B-DFlash2-GGUF 2d9571f Qwen3.8-27B-DFlash2-Q4_K_M.gguf
dl unsloth/Qwen3.5-2B-GGUF f6d5376 Qwen3.5-2B-Q8_0.gguf
dl unsloth/Qwen3.5-9B-MTP-GGUF 9716a63 Qwen3.5-9B-Q4_K_M.gguf
dl unsloth/Qwen3.8-27B-GGUF 4af51b3 Qwen3.8-27B-UD-Q4_K_M.gguf
dl unsloth/Qwen3.8-27B-GGUF 4af51b3 MTP/mtp-Qwen3.8-27B-Q4_0.gguf
dl bartowski/Llama-3.2-1B-Instruct-GGUF 067b946 Llama-3.2-1B-Instruct-Q8_0.gguf
# Llama 3.1 8B and 70B: ollama pull llama3.1:8b llama3.1:70b (bench_llama.py reads Ollama's blob files)
python3 -I - "$OUT" "$(dirname "$0")/model_sources.json" <<'PY'
import hashlib, json, os, sys
out, src = sys.argv[1], json.load(open(sys.argv[2]))
bad = 0
for key, v in src.items():
    repo, f = key.split("/", 2)[1], key.split("/", 2)[2]
    p = os.path.join(out, repo, f)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    ok = h.hexdigest() == v["sha256"]
    bad += not ok
    print(("OK  " if ok else "BAD ") + p)
sys.exit(1 if bad else 0)
PY
echo ALL_DONE
