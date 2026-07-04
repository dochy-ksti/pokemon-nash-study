#!/usr/bin/env bash
# else分岐: 純スピン(sleep呼ばない, -1) vs sleep(0) で各2連続 run を比較。
set -euo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=60; OUT=/tmp/diag_spin.pt
make build
runpair() {  # $1=sleep_seconds $2=label
  for r in 1 2; do
    cp "$T/shared_init.pt" "$OUT"
    t0=$(date +%s)
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --sleep-seconds "$1" --checkpoint-path "$OUT" > /dev/null 2>&1
    t1=$(date +%s)
    echo "  [$2] RUN$r : $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
  done
}
echo "===== 純スピン (sleep_seconds=-1, time.sleep呼ばない) ====="; runpair -1 "純スピン"
echo "===== sleep(0) (sleep_seconds=0.0) ====="; runpair 0.0 "sleep0"
echo "===== DONE ====="
