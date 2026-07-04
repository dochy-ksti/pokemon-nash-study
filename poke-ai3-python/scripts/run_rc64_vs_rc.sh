#!/usr/bin/env bash
# 学習時教師 sims A/B (random+crit 環境)。
# 仮説: 乱数環境では学習時の探索教師 sims を増やすと方策が強くなる。
# RC (sims32, 既存流用) vs RC64 (sims64, 新規)。差分は --sims のみ。
# funnel eval は policy-only なので sims は学習教師の質だけに効く。
# rate は RC_finalists vs RC64_finalists を 2 ルールで実行 (random+crit / no-rng)。
# RC64 funnel は --resume 可能。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
# RC と sims 以外完全一致の COMMON。
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 --warmup-steps 10 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit)

echo "===== [$(date +%H:%M)] make build (native 鮮度確保) ====="
make build

echo "===== [$(date +%H:%M)] funnel RC64 (sims64, random+crit) ====="
uv run python scripts/ckpt_tournament.py funnel --tag RC64 "${COMMON[@]}" --sims 64

echo "===== [$(date +%H:%M)] rate (1) random+crit ルール ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/RC_finalists.json" "$T/RC64_finalists.json"

echo "===== [$(date +%H:%M)] rate (2) no-random/no-crit ルール ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --no-random --no-crit \
  --funnel-json "$T/RC_finalists.json" "$T/RC64_finalists.json"

echo "===== [$(date +%H:%M)] DONE ====="
