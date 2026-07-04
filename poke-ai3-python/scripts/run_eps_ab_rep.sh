#!/usr/bin/env bash
# epochs-per-step A/B 再現ラン (random+crit, sims64)。別シード独立ラン。
# 前回 (20260630_1251) は eps5=RC64 流用だったが、再現性確認のため両アームとも
# 新規 funnel を別シードで回す。epochs-per-step は学習を変えない (--snapshot-every のみ)
# ので、両アームの学習は同分布 = この A/B は実質「ラン間ノイズ + 選抜粒度差」の再測定。
# 前回 E10 平均 +2.4 / eps5 -2.4 (差 4.8 Elo) が別シードで再現するかを見る。
# 各 funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
# epochs-per-step / warmup-steps 以外完全一致の COMMON (前回と同一)。
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit --sims 64)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel E5b (epochs-per-step 5, warmup-steps 10) ====="
uv run python scripts/ckpt_tournament.py funnel --tag E5b "${COMMON[@]}" \
  --epochs-per-step 5 --warmup-steps 10

echo "===== [$(date +%H:%M)] funnel E10b (epochs-per-step 10, warmup-steps 5) ====="
uv run python scripts/ckpt_tournament.py funnel --tag E10b "${COMMON[@]}" \
  --epochs-per-step 10 --warmup-steps 5

echo "===== [$(date +%H:%M)] rate (E10b vs E5b) random+crit ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/E5b_finalists.json" "$T/E10b_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
