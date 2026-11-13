#!/usr/bin/env bash
# Re-run of queue_ysk.sh's rows that hit "address already in use" (a YASAKANI server holds its port for a few seconds after
# it stops), plus the llama.cpp anchor whose --suffix flag was mistyped. Waits until each port has no sockets at all.
set -uo pipefail
cd "$(dirname "$0")"
FMT="$(pwd)/../formats-harness"
free() { for i in $(seq 120); do ss -tan | grep -q -E "[:.]$1 " || { sleep 3; return; }; sleep 1; done; echo "port $1 still busy"; }
date -Is; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
free 8091; python3 bench.py 27b-none --runs 3 --suffix=-ysk-session
free 8091; python3 bench.py ysk-quality --runs 3
cd "$FMT"
free 8080; .venv/bin/python row.py gguf-A-r2 --suite speed
free 8765; .venv/bin/python row.py ysk-speed-q --suite quality
free 8765; .venv/bin/python row.py ysk-quality --suite speed,prefill
free 8765; .venv/bin/python row.py ysk-quality-q --suite quality
date -Is; echo QUEUE DONE
