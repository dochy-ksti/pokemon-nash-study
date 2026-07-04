#!/usr/bin/env bash
# 単一 max の s/ep を「スクリプトの1本目」として測る。
# 用法: bench_one_max.sh <MAX> [EPOCHS]
# 各 max を別々のスクリプト起動で呼ぶことで、全て「1本目=速い regime」で揃える。
set -euo pipefail
cd "$(dirname "$0")/.."

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX="$1"; EPOCHS="${2:-100}"
OUT=/tmp/bench_one_${MAX}.pt

cp "$T/shared_init.pt" "$OUT"
echo "===== [$(date +%H:%M:%S)] search-max=$MAX : ${EPOCHS}ep 計測 (単独1本目) ====="
t0=$(date +%s)
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
  --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
  --checkpoint-path "$OUT" > /dev/null 2>&1
t1=$(date +%s); dt=$((t1 - t0))
echo "  max=$MAX : ${dt}s / ${EPOCHS}ep = $(echo "scale=3; $dt/$EPOCHS" | bc) s/ep"
