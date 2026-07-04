#!/usr/bin/env bash
# epochs-per-step A/B (random+crit, sims64 環境)。
# 仮説検証: snapshot 間隔 epochs-per-step を 5→10 にすると出来上がる方策の強さが変わるか。
# eps5 アーム = 既存 RC64_finalists を流用 (RC64 は sims64/eps5/warmup10/random+crit/
#   shared_init 始点 = まさに eps5 ベースライン)。
# eps10 アーム = RC64 と同 COMMON で --epochs-per-step 10 のみ差分。
#   warmup は epoch 基準で揃える: eps10 は step 間隔が倍なので warmup-steps 5 (=50ep) で
#   eps5 の warmup-steps 10 (=50ep) と等価。train-block-epochs 50 は 10 の倍数で OK。
# rate は E10_finalists vs RC64_finalists を random+crit で実行。
# E10 funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
# RC64 と epochs-per-step / warmup-steps 以外完全一致の COMMON。
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit --sims 64)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel E10 (epochs-per-step 10, warmup-steps 5) ====="
uv run python scripts/ckpt_tournament.py funnel --tag E10 "${COMMON[@]}" \
  --epochs-per-step 10 --warmup-steps 5

echo "===== [$(date +%H:%M)] rate (E10 vs RC64=eps5) random+crit ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/RC64_finalists.json" "$T/E10_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
