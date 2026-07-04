#!/usr/bin/env bash
# depth_skew A/B を現手法(全履歴比較・warmup・finalists 3個)で再実行する driver。
# A 完了 → B → rate を一つずつ直列実行。各 funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python (uv run / train-loop の cwd)

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
COMMON=(--start "$T/shared_init.pt" --peaks-per-rr 3 --finalists-target 3 \
  --warmup-steps 10 --epochs-per-step 20 --max-added-epochs 4000 \
  --n-per-side 512 --num-games 64 --stage 3b)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel A (depth_skew=2.0 / search 4-8) ====="
uv run python scripts/ckpt_tournament.py funnel --tag A "${COMMON[@]}" \
  --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8

echo "===== [$(date +%H:%M)] funnel B (depth_skew=1.0 / search 6-12) ====="
uv run python scripts/ckpt_tournament.py funnel --tag B "${COMMON[@]}" \
  --depth-skew 1.0 --search-turn-min 6 --search-turn-max 12

echo "===== [$(date +%H:%M)] rate (A vs B) ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 64 --stage 3b \
  --funnel-json "$T/A_finalists.json" "$T/B_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
