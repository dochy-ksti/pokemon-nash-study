#!/usr/bin/env bash
# battle_seed を固定して4連続 run。全部同速なら「1本目速い」は seed 由来の対戦差。
# 1速い-3遅いのままなら data 非依存のプロセス順効果。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=45; OUT=/tmp/diag_seed.pt; SEED=12345
make build
echo "===== FIXED SEED=$SEED ====="
for r in 1 2 3 4; do
  cp "$T/shared_init.pt" "$OUT"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --battle-seed "$SEED" --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "  FIXED RUN$r : $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
done
echo "===== RANDOM SEED (対照) ====="
for r in 1 2; do
  cp "$T/shared_init.pt" "$OUT"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "  RANDOM RUN$r : $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
done
echo "===== DONE ====="
