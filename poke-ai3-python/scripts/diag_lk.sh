#!/usr/bin/env bash
# lookahead カウンタ: 2連続 run で decisions/rollouts/plies/forced の比率を比較。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=45; OUT=/tmp/diag_lk.pt
export POKE_AI3_ROOT_DIAG=1
make build
run() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] $1 開始"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --checkpoint-path "$OUT" > /dev/null 2> "$T/lk_${1}.err"
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1 終了: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
  echo "--- $1 lk_diag 最終行 ---"; grep 'lk_diag' "$T/lk_${1}.err" | tail -1
}
run RUN1
run RUN2
echo "===== DONE ====="
