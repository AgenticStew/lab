#!/usr/bin/env bash
# Sequential run of every row, one model loaded at a time. Usage: run_all.sh [row ...]  (default: all rows)
# Rows: qwen arete qwen-mtp arete-mtp arete-r2 qwen-r2
set -u
B="$(cd "$(dirname "$0")" && pwd)"
P="$(dirname "$B")"
LLAMA=~/src/llama.cpp-b12189/build/bin/llama-server
M=~/models/arete-test
MTP=~/models/arete-test/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TPL="$B/arete-chat_template.jinja"
EXTRA='{"chat_template_kwargs": {"enable_thinking": false}, "cache_prompt": false}'
mkdir -p "$B/logs"

model_of() { case "$1" in qwen*) echo "$M/qwen/Qwen3.8-27B-Q4_K_M.gguf" ;; arete*) echo "$M/arete/NousResearch_Arete-27B-Q4_K_M.gguf" ;; esac; }

serve() {  # $1 row
  local args=(-m "$(model_of "$1")" -ngl 99 -c 32768 --parallel 1 -fa on -ctk f16 -ctv f16 --no-cache-prompt --cache-ram 0
              --jinja --chat-template-file "$TPL" --fit off --host 127.0.0.1 --port 8080 --alias contestant --metrics)
  case "$1" in *-mtp) args+=(-md "$MTP" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 -ngld 99) ;; esac
  echo "[$1] $LLAMA ${args[*]}" > "$B/logs/$1-server.log"
  "$LLAMA" "${args[@]}" >> "$B/logs/$1-server.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && return 0
    kill -0 $SPID 2>/dev/null || { echo "[$1] server died"; return 1; }
    sleep 1
  done
  echo "[$1] server not healthy after 180 s"; return 1
}

stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 3; }

row() {
  local r=$1
  echo "=== $r $(date -Iseconds)"
  serve "$r" || { stop; return 1; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$B/logs/$r-vram-loaded.txt"
  case "$r" in
    qwen|arete)
      python3 "$B/run_quality.py" --tag "$r" || echo "[$r] quality failed"
      python3 "$B/harness.py" --base http://127.0.0.1:8080/v1 --label "$r-A" --suite speed,prefill --prefill-ids 2k,8k \
        --extra "$EXTRA" --note "b12189, no drafting, 32K ctx, F16 KV" ;;
    *-mtp)
      python3 "$B/harness.py" --base http://127.0.0.1:8080/v1 --label "$r" --suite speed \
        --extra "$EXTRA" --note "b12189, Qwen MTP head as drafter (-md mtp-Qwen3.8-27B-Q4_0.gguf), n-max 3, p-min 0.1" ;;
    *-r2)
      python3 "$B/harness.py" --base http://127.0.0.1:8080/v1 --label "$r" --suite speed \
        --extra "$EXTRA" --note "reverse-order repeat, no drafting" ;;
  esac
  stop
}

ROWS=("$@"); [ ${#ROWS[@]} -eq 0 ] && ROWS=(qwen arete qwen-mtp arete-mtp arete-r2 qwen-r2)
for r in "${ROWS[@]}"; do row "$r"; done
echo "=== done $(date -Iseconds)"
