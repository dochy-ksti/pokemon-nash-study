#!/usr/bin/env bash
# RootTask 計測: 2連続 run で obs/sec が落ちるかを見る。stderr に root_diag が出る。
set -euo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=50; OUT=/tmp/diag_root.pt
export POKE_AI3_ROOT_DIAG=1
make build
run() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] $1 開始"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --checkpoint-path "$OUT" > /dev/null 2> "$T/root_diag_${1}.err"
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1 終了: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
}
run RUN1
run RUN2
echo "===== DONE ====="
