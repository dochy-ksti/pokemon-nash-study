#!/usr/bin/env bash
# 教師あり学習の内部パス回数 epochs 4 vs 3 の A/B (random+crit, sims64, search8)。
# --train-supervised-epochs は 1 バッチの生成データを教師あり学習で何パスなめるか。
# 学習の軌跡そのものを変える真のノブ (生成は不変、learn step だけ変わる)。
# epochs=4 (baseline) は RC64 流用。E3 アームだけ新規 funnel で --train-supervised-epochs 3。
# COMMON は RC64 と epochs 以外完全一致。時間プローブは省略 (learn 側ノブのため生成不変)。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
SCR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/2c45769d-b60f-4b86-906a-3c8c7d88b007/scratchpad

COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sims 64 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 --warmup-steps 10 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit)

echo "===== [$(date +%H:%M:%S)] make build ====="
make build

# --- E3 funnel (--train-supervised-epochs 3) ---
echo "===== [$(date +%H:%M:%S)] funnel E3 (--train-supervised-epochs 3) ====="
f0=$(date +%s)
uv run python scripts/ckpt_tournament.py funnel --tag E3 "${COMMON[@]}" \
  --train-supervised-epochs 3
echo "===== [$(date +%H:%M:%S)] funnel E3 wall=$(( $(date +%s) - f0 ))s ====="

# --- rate (RC64 流用 epochs=4 + E3 epochs=3) ---
echo "===== [$(date +%H:%M:%S)] rate (RC64 + E3) random+crit ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/RC64_finalists.json" "$T/E3_finalists.json"

echo "===== [$(date +%H:%M:%S)] DONE ====="
