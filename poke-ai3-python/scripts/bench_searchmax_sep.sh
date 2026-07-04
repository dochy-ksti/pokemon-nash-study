#!/usr/bin/env bash
# 純生成コスト s/ep ベンチ (位置効果対策版):
#  1) 先頭に捨て暖機 run を1本 → GPU を定常状態へ。
#  2) パリンドローム順 6 7 8 8 7 6 で各 max を2回ずつ計測。
#     → 各 max の平均実行位置が揃い、サーマルドリフトが相殺される。
#  3) 各計測の前に sleep 60 で条件を均一化。
# 選抜・h2h なし=学習生成のみ。!!! GPU 専有で実行すること !!!
set -euo pipefail
cd "$(dirname "$0")/.."

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
EPOCHS=100
OUT=/tmp/bench_sep.pt

run_train() {  # $1=max $2=epochs
  cp "$T/shared_init.pt" "$OUT"
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$1" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$2" \
    --checkpoint-path "$OUT" > /dev/null 2>&1
}

make build

echo "===== 捨て暖機 (max7, 80ep) ====="
run_train 7 80

for MAX in 6 7 8 8 7 6; do
  sleep 60
  echo "===== search-max=$MAX : ${EPOCHS}ep 計測 ====="
  t0=$(date +%s)
  run_train "$MAX" "$EPOCHS"
  t1=$(date +%s); dt=$((t1 - t0))
  echo "  max=$MAX : ${dt}s / ${EPOCHS}ep = $(echo "scale=3; $dt/$EPOCHS" | bc) s/ep"
done
echo "===== DONE ====="
