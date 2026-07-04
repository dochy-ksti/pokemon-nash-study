#!/usr/bin/env bash
# minibatch-size A/B を 0624 funnel 手法で実行する driver。
# 両手法とも採用設定 g64/b512/th128 + search A (depth-skew 2.0 / search 4-8 / sims 32
# / stage 3b) で共通。差分は学習の --minibatch-size のみ。
# 比較する minibatch は環境変数で差し替え可能 (既定は step3b の 64 vs 256):
#   MBS_A (既定 64) / MBS_B (既定 256)
#   例: step3c の 256 vs 512 → MBS_A=256 MBS_B=512 bash scripts/run_ab_minibatch.sh
# threshold は両手法 128 で同一なので epochs-per-step=5 も共通 (examples 等価)。
# 各 funnel は --resume 可能。A → B → rate を直列実行。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

MBS_A=${MBS_A:-64}
MBS_B=${MBS_B:-256}
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
# 採用設定 + search 条件 (両手法共通) + 選抜パラメータ + eval 設定。
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sims 32 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 --warmup-steps 10 \
  --n-per-side 512 --num-games 256 --stage 3b)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel A (minibatch $MBS_A) ====="
uv run python scripts/ckpt_tournament.py funnel --tag A "${COMMON[@]}" \
  --train-minibatch-size "$MBS_A"

echo "===== [$(date +%H:%M)] funnel B (minibatch $MBS_B) ====="
uv run python scripts/ckpt_tournament.py funnel --tag B "${COMMON[@]}" \
  --train-minibatch-size "$MBS_B"

echo "===== [$(date +%H:%M)] rate (A vs B) ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --funnel-json "$T/A_finalists.json" "$T/B_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
