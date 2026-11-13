#!/usr/bin/env python3
"""Classic draft-model pairs on a standard-attention model family (Llama 3.1 + Llama 3.2 1B), same harness as bench.py.

  bench_llama.py <label> ... [--runs N] [--prompts ...]

Llama 3.1 8B and 70B are Ollama's own Q4_K_M GGUF blobs (read directly by llama.cpp); the 1B guesser is
bartowski/Llama-3.2-1B-Instruct-GGUF Q8_0 (see ../model_sources.json). All three share Llama 3's tokenizer.
The 70B (42.5 GB) can't fit 32 GB, so -ngl keeps as many layers on the GPU as fit; the guesser rows give up GPU
layers to make room for the guesser, so both use about the same VRAM.
"""
import bench as b

OL = "/usr/share/ollama/.ollama/models/blobs/"
L8 = OL + "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
L70 = OL + "sha256-de20d2cf2dc430b1717a8b07a9df029d651f3895dbffec4729a3902a6fe344c9"
L1 = b.M / "specdec/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"

b.ROWS["l8-none"] = b.srv(L8)
b.ROWS["l8-ngram-mod"] = b.srv(L8, "--spec-type", "ngram-mod")
b.ROWS["l8-ngram-simple"] = b.srv(L8, "--spec-type", "ngram-simple", "--spec-draft-n-max", "64")
for n in (2, 4, 6, 8, 12):
    b.ROWS[f"l8-d1-n{n}"] = b.srv(L8, *b.dm(L1, n))
for ngl in (44, 46, 48, 50, 52):
    b.ROWS[f"l70-none-ngl{ngl}"] = b.srv(L70, ngl=ngl)
    for n in (4, 8):
        b.ROWS[f"l70-d1-n{n}-ngl{ngl}"] = b.srv(L70, *b.dm(L1, n), ngl=ngl)

if __name__ == "__main__":
    b.main()
