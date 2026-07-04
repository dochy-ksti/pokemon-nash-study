#!/usr/bin/env bash
# search-turn-max / sims 3way 実験 (random+crit, baseline=RC64 流用)。
# 3条件: baseline(search-max 8, sims 64) / A(search-max 9, sims 64) / B(search-max 8, sims 128)。
# 学習教師の計算ノブを増やすと強さが上がるか、時間コストはどれだけかを測る。
# search-turn-max も sims も学習教師の質だけに効く (funnel eval は policy-only で未使用)。
#
# 構成:
#   PHASE 1 タイミングプローブ: 3条件を各 100ep だけ shared_init から逐次実行し、
#     wall-clock と batch_stats(累計 real)から examples/s を apples-to-apples 取得。
#     probe ckpt は使い捨て (scratchpad)。
#   PHASE 2 強さ funnel: A(s9) と B(sims128) を新規 funnel。baseline は RC64 流用。
#   PHASE 3 rate: A + B + RC64 の 9 finalists を 1 プール random+crit n512。
# funnel は --resume 可能。段階的方針: まず各 1 ラン、微妙な差は後で再現ラン。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
SCR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/2c45769d-b60f-4b86-906a-3c8c7d88b007/scratchpad
PROBE_DIR="$SCR/probe_3way"
mkdir -p "$PROBE_DIR"

# --- 共通学習設定 (RC64 と sims/search-max 以外完全一致) ---
GEN=(--num-games 64 --sim-concurrency 16 --search-turn-min 4 --depth-skew 2.0 \
  --random --crit --stage 3b --max-batch-size 512 --trajectories-threshold 128 \
  --minibatch-size 256)
# funnel 共通 (RC64 と同一の選抜/eval パラメータ)。
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 --warmup-steps 10 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit)

echo "===== [$(date +%H:%M:%S)] make build ====="
make build

# --- PHASE 1: タイミングプローブ (各 100ep, 逐次) ---
probe() {  # $1=label $2=search_max $3=sims
  local label=$1 smax=$2 sims=$3
  local ck="$PROBE_DIR/${label}.pt"
  rm -f "$PROBE_DIR/${label}"*.pt
  echo "##### [$(date +%H:%M:%S)] PROBE $label (search-max $smax, sims $sims) start #####"
  local t0 t1
  t0=$(date +%s)
  uv run train-loop "${GEN[@]}" --search-turn-max "$smax" --sims "$sims" \
    --max-epochs 100 --checkpoint-path "$ck" --snapshot-every 1000
  t1=$(date +%s)
  echo "##### [$(date +%H:%M:%S)] PROBE $label done: wall=$((t1 - t0))s (100ep) #####"
}
echo "===== [$(date +%H:%M:%S)] PHASE 1 timing probes ====="
probe baseline 8 64
probe A_s9     9 64
probe B_sims128 8 128

# --- PHASE 2: 強さ funnel (A, B) ---
echo "===== [$(date +%H:%M:%S)] PHASE 2 funnel A (search-max 9, sims 64) ====="
fa0=$(date +%s)
uv run python scripts/ckpt_tournament.py funnel --tag A_s9 "${COMMON[@]}" \
  --search-turn-max 9 --sims 64
echo "===== [$(date +%H:%M:%S)] funnel A wall=$(( $(date +%s) - fa0 ))s ====="

echo "===== [$(date +%H:%M:%S)] PHASE 2 funnel B (search-max 8, sims 128) ====="
fb0=$(date +%s)
uv run python scripts/ckpt_tournament.py funnel --tag B_sims128 "${COMMON[@]}" \
  --search-turn-max 8 --sims 128
echo "===== [$(date +%H:%M:%S)] funnel B wall=$(( $(date +%s) - fb0 ))s ====="

# --- PHASE 3: rate (A + B + RC64 流用) ---
echo "===== [$(date +%H:%M:%S)] PHASE 3 rate (A_s9 + B_sims128 + RC64) random+crit ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/RC64_finalists.json" "$T/A_s9_finalists.json" "$T/B_sims128_finalists.json"

echo "===== [$(date +%H:%M:%S)] DONE ====="
