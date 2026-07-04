#!/usr/bin/env bash
# threshold + batch充填 A/B を 0624 funnel 手法で実行する driver。
# 両手法とも search A (depth-skew 2.0 / search 4-8 / sims 32 / stage 3b) で共通。
# 差分は学習スループット設定のみ:
#   A (baseline): num-games 32 / max-batch 256 / threshold 32  (epochs-per-step 20)
#   B (new)     : num-games 64 / max-batch 512 / threshold 128 (epochs-per-step 5)
# epochs-per-step を threshold に反比例させ snapshot 粒度を examples 等価に正規化。
# warmup-steps=10 は step 単位 → A=200ep / B=50ep = 両者 6400 traj で examples 等価。
# 各 funnel は --resume 可能。A → B → rate を直列実行。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python (uv run / train-loop の cwd)

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
# 学習側 search 条件 (両手法共通) + 選抜パラメータ + eval 設定。
SEARCH=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sims 32 --sim-concurrency 16 \
  --peaks-per-rr 3 --finalists-target 3 --warmup-steps 10 \
  --n-per-side 512 --num-games 256 --stage 3b)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel A (baseline g32/b256/th32) ====="
uv run python scripts/ckpt_tournament.py funnel --tag A "${SEARCH[@]}" \
  --train-num-games 32 --train-max-batch-size 256 --train-trajectories-threshold 32 \
  --epochs-per-step 20 --max-added-epochs 4000

echo "===== [$(date +%H:%M)] funnel B (new g64/b512/th128) ====="
uv run python scripts/ckpt_tournament.py funnel --tag B "${SEARCH[@]}" \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --epochs-per-step 5 --max-added-epochs 1000

echo "===== [$(date +%H:%M)] rate (A vs B) ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --funnel-json "$T/A_finalists.json" "$T/B_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
