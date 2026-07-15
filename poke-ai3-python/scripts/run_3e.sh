#!/usr/bin/env bash
# Stage 3e (= 3c + Crunch/Dark Pulse) の厳密 Nash を解き、3c と軌跡比較する。
#
# 各段は出力ファイルが既にあればスキップするので、中断後の再実行は安全 (idempotent)。
# solve は最大 RSS ~11GB を使うため、2 段は必ず逐次実行する (並列にすると ~22GB 必要)。
#
#   setsid nohup bash poke-ai3-python/scripts/run_3e.sh > /tmp/psro/run_3e.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."
DATA=../data/poke-ai3/nash_geo
mkdir -p "$DATA"

if [ -f "$DATA/nash_geo_h26_3e.npz" ]; then
  echo "[driver] skip solve (nash_geo_h26_3e.npz exists)"
else
  echo "[driver] solving stage 3e (H=26, discount=0.99)"
  uv run python scripts/run_nash_geo_h26.py --stage 3e --hp-buckets 26
fi

if [ -f "$DATA/trajectory_3c_vs_3e.json" ]; then
  echo "[driver] skip trajectory (trajectory_3c_vs_3e.json exists)"
else
  echo "[driver] trajectory compare 3c vs 3e"
  uv run python scripts/run_nash_trajectory_compare.py \
    --stages 3c 3e --games-per-config 2000 --seed 20260715 \
    --output "$DATA/trajectory_3c_vs_3e.json"
fi

echo "[driver] done"
