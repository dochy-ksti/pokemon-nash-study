#!/usr/bin/env bash
# random+crit 学習 (RC) vs no-random/no-crit 学習 (A, 既存 v2 finalists) のクロスルール比較。
# RC は A と同一 COMMON (shared_init から / depth-skew 2.0 / search 4-8 / sims 32 /
# g64 b512 th128 / minibatch 256 / epochs-per-step 5 / block 50 / warmup 10 / finalists 3)
# に --random --crit のみ追加して funnel 学習・選抜する。
# rate は A_finalists.json vs RC_finalists.json を 2 ルールで実行:
#   (1) --random --crit       … random+crit ルールでの強さ
#   (2) --no-random --no-crit  … 決定論ルールでの強さ
# RC funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sims 32 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 --warmup-steps 10 \
  --n-per-side 512 --num-games 256 --stage 3b)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel RC (random+crit) ====="
uv run python scripts/ckpt_tournament.py funnel --tag RC "${COMMON[@]}" --random --crit

echo "===== [$(date +%H:%M)] rate (1) random+crit ルール ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/A_finalists.json" "$T/RC_finalists.json"

echo "===== [$(date +%H:%M)] rate (2) no-random/no-crit ルール ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --no-random --no-crit \
  --funnel-json "$T/A_finalists.json" "$T/RC_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
