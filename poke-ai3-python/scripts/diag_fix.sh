#!/usr/bin/env bash
# FxHasher 適用後の検証: 4連続 run で s/ep が揃って速いままか(衝突seed依存が消えたか)。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=45; OUT=/tmp/diag_fix.pt
make build
for r in 1 2 3 4; do
  cp "$T/shared_init.pt" "$OUT"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "  RUN$r : $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
done
echo "===== DONE ====="
