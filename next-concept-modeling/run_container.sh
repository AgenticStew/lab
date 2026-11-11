#!/usr/bin/env bash
# Host-side wrapper for TESTPLAN steps 1-6. Run it on the host.
#   bash bench/run_container.sh all      # one command: phase1, wait for your go, phase2, stop
#   bash bench/run_container.sh phase1   # step 1 (container, network on), steps 2-4 (downloads, installs, dry runs)
#   bash bench/run_container.sh phase2   # cuts the network, then steps 5-6 (timed rows, quality rows), stops the container
# In `all` mode the wrapper writes its state to results/.ctl/state and waits between phases (and after a failure)
# for a file you write once you have read the logs: results/.ctl/go (next phase), results/.ctl/retry (rerun the
# phase after a fix to bench/steps) or results/.ctl/abort (stop the container and exit). It only ever runs the
# bench/steps scripts. Ctrl-C stops it at any time; `docker stop ncm-bench` removes the container.
# Same container as TESTPLAN step 1 (no $HOME, no secrets, dl + results mounted, bench read-only), kept alive
# between phases with sleep instead of an interactive shell; --rm removes it when it stops.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
NAME=ncm-bench
CTL=results/.ctl

up() {
  if [ ! -f dl/duo/lm1b_Duo.ckpt ]; then echo "STOP: dl/duo/lm1b_Duo.ckpt is missing"; exit 3; fi
  mkdir -p dl results "$CTL"
  if [ -n "$(docker ps -q -f "name=^${NAME}\$")" ]; then
    echo "reusing the running $NAME container"
  else
    docker run --gpus all --rm -d --name "$NAME" -v "$PWD/dl:/dl" -v "$PWD/results:/results" -v "$PWD/bench:/bench:ro" -w /work "$IMAGE" sleep infinity
  fi
}

step() {  # step N: run bench/steps/stepN.sh in the container, log to results/stepN.log; nonzero on failure
  docker exec "$NAME" bash "/bench/steps/step$1.sh" 2>&1 | tee "results/step$1.log"
  return "${PIPESTATUS[0]}"
}

phase1() { step 2 || return 1; step 3 || return 1; step 4 || return 1; }

phase2() {
  docker network disconnect bridge "$NAME" 2>/dev/null || echo "(network already disconnected)"
  step 5 || return 1
  step 6 || return 1
}

wait_for_go() {  # $1 = state to report; returns 0 for go, 1 for retry; exits on abort
  echo "$1" > "$CTL/state"
  echo "[$(date +%H:%M:%S)] $1: waiting for you (results/.ctl/go | retry | abort)"
  while true; do
    if [ -f "$CTL/abort" ]; then rm -f "$CTL"/abort; echo "abort"; docker stop "$NAME" >/dev/null || true; exit 1; fi
    if [ -f "$CTL/go" ]; then rm -f "$CTL"/go; return 0; fi
    if [ -f "$CTL/retry" ]; then rm -f "$CTL"/retry; return 1; fi
    sleep 5
  done
}

run_phase() {  # run_phase 1|2, repeating after a failure until you say go or abort
  while true; do
    echo "running-phase$1" > "$CTL/state"
    if "phase$1"; then
      wait_for_go "phase$1-ok" && return 0
    else
      wait_for_go "phase$1-failed" && return 0
    fi
  done
}

case "${1:-}" in
  all)
    up
    rm -f "$CTL"/go "$CTL"/retry "$CTL"/abort
    run_phase 1
    run_phase 2
    docker stop "$NAME" >/dev/null || true
    echo "done" > "$CTL/state"
    echo "all done: results are in results/."
    ;;
  phase1)
    up
    phase1
    echo "phase1 done: check results/. The container stays up for phase2."
    ;;
  phase2)
    phase2
    docker stop "$NAME"
    echo "phase2 done: check results/."
    ;;
  *)
    echo "usage: bash bench/run_container.sh all|phase1|phase2"; exit 2
    ;;
esac
