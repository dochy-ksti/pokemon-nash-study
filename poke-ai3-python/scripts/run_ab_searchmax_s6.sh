#!/usr/bin/env bash
# 先読み削減 A/B 続き: 削減版 S6(search-max 6)を funnel → S8(=A) vs S7 vs S6 を3手法 rate。
# depth-skew 2.0 / search-min 4 は固定。現行 funnel 3v3 BT で判定。
# S6 funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python (uv run / train-loop の cwd)

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
COMMON=(--start "$T/shared_init.pt" --peaks-per-rr 3 --finalists-target 3 \
  --warmup-steps 10 --epochs-per-step 20 --max-added-epochs 4000 \
  --n-per-side 512 --num-games 64 --stage 3b --depth-skew 2.0 --search-turn-min 4)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel S6 (search-max 6, 削減版) ====="
uv run python scripts/ckpt_tournament.py funnel --tag S6 "${COMMON[@]}" \
  --search-turn-max 6

echo "===== [$(date +%H:%M)] rate (S8[=A] vs S7 vs S6, n-per-side 1024) ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 1024 --num-games 64 --stage 3b \
  --funnel-json "$T/A_finalists.json" "$T/S7_finalists.json" "$T/S6_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
