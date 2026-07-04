#!/usr/bin/env bash
# 1本目/2本目を cProfile で取得し、関数別に何が遅くなるか比較する。
set -euo pipefail
cd "$(dirname "$0")/.."

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=40
P1=/tmp/run1.prof
P2=/tmp/run2.prof

ARGS="--num-games 32 --sim-concurrency 16 --sims 64 --search-turn-min 4 \
  --search-turn-max $MAX --depth-skew 2.0 --no-random --no-crit --stage 3b \
  --max-epochs $EPOCHS"

make build

prof() {  # $1=label $2=out
  cp "$T/shared_init.pt" "/tmp/prof_${1}.pt"
  echo ">>> [$(date +%H:%M:%S)] $1 開始 (cProfile)"
  t0=$(date +%s)
  uv run python -m cProfile -o "$2" -m poke_ai3_train.train_loop \
    $ARGS --checkpoint-path "/tmp/prof_${1}.pt" > /dev/null 2>&1
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1 終了: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
}

prof RUN1 "$P1"
prof RUN2 "$P2"
echo "===== prof DONE ($P1 / $P2) ====="
