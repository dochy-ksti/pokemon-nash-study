#!/usr/bin/env bash
# RUN1(速)とRUN2(遅)を同条件で call-graph 記録し、関数単位で「どのコードがどれだけ
# 実行されたか」を比較する。-n でサンプル数(=サイクル比例)を出すので絶対比較できる。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=30; OUT=/tmp/diag_pb.pt
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
make build

rec() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] record $1"
  t0=$(date +%s)
  perf record -F 499 -g --call-graph dwarf,16384 -o "$T/pb_${1}.data" -- \
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
}

rec RUN1
rec RUN2

for L in RUN1 RUN2; do
  echo "===== $L flat self (top30, サンプル数付き) ====="
  perf report -i "$T/pb_${L}.data" --stdio -g none -n 2>/dev/null \
    | grep -vE '^#|^$' | head -30
  echo
done
echo "===== DONE ====="
