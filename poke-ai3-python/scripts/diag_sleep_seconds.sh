#!/usr/bin/env bash
# busy-spin 検証: --sleep-seconds 0.0(現状) と 0.001 で各2連続 run の s/ep を比較。
set -euo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=60; OUT=/tmp/diag_ss.pt
make build
runpair() {  # $1=sleep_seconds
  for r in 1 2; do
    cp "$T/shared_init.pt" "$OUT"
    t0=$(date +%s)
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --sleep-seconds "$1" --checkpoint-path "$OUT" > /dev/null 2>&1
    t1=$(date +%s)
    echo "  sleep_seconds=$1 RUN$r : $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
  done
}
echo "===== sleep_seconds=0.0 (現状) ====="; runpair 0.0
echo "===== sleep_seconds=0.001 ====="; runpair 0.001
echo "===== DONE ====="
