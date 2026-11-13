#!/usr/bin/env bash
# Build YASAKANI (github.com/yasakani-dev/yasakani, Apache-2.0) for Linux. Upstream supports Windows 11 only; its source
# already has the POSIX code paths. The one change: two Windows-only file calls in engine35.cpp (_fseeki64, _ftelli64) are
# mapped to fseeko / ftello with a compiler flag, so the source stays untouched. CUDA flags mirror the MSVC branch of its
# CMakeLists.txt (-O3 --fmad=false). Tested 13 Nov 2026: commit 758ef28, CUDA 13.3, GCC 13.3, Ubuntu 24.04 under WSL2,
# RTX 5090 (the kernels target compute capability 12.0 only). `yasakani selfcheck` reproduced the developer's reference.
#
#   SRC=~/src/yasakani VENV=~/yasakani-venv NVCC=/usr/local/cuda/bin/nvcc ./install.sh
set -euo pipefail
SRC=${SRC:-$HOME/src/yasakani}
VENV=${VENV:-$HOME/yasakani-venv}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
[ -d "$SRC" ] || git clone https://github.com/yasakani-dev/yasakani.git "$SRC"
cd "$SRC"
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_FLAGS="-O3 --fmad=false" -DCMAKE_CXX_FLAGS="-O2 -D_ftelli64=ftello -D_fseeki64=fseeko"
cmake --build build-linux -j "$(nproc)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q "$SRC/python"
cat <<MSG
engine: $SRC/build-linux/libyskn.so
model:  hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --local-dir ~/models/yasakani-speed   (or -quality)
check:  $VENV/bin/yasakani selfcheck --model-dir ~/models/yasakani-speed --dll $SRC/build-linux/libyskn.so
serve:  $VENV/bin/yasakani serve --model-dir ~/models/yasakani-speed --dll $SRC/build-linux/libyskn.so --port 8765
MSG
