#!/usr/bin/env bash
# search-turn-max 7 vs 8 vs 9 の 3way (random+crit, sims64)。3アームとも新規 funnel。
# search-turn-max は rollout 最大深さ ply。学習教師の質だけに効く (funnel eval は policy-only)。
# 前回 3way (20260630_2010) では 8(baseline) vs 9 で 9 がコヒーレントに弱かった。
# 今回は 7 を加え、かつ run 間ノイズを避けるため 8/9 も作り直して同時期 3アーム比較。
# 各 funnel は --resume 可能。差分は --search-turn-max のみ (min は 4 据え置き)。
set -euo pipefail
cd "$(dirname "$0")/.."   # poke-ai3-python

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
SCR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/2c45769d-b60f-4b86-906a-3c8c7d88b007/scratchpad
PROBE_DIR="$SCR/probe_smax"
mkdir -p "$PROBE_DIR"

GEN=(--num-games 64 --sim-concurrency 16 --search-turn-min 4 --sims 64 --depth-skew 2.0 \
  --random --crit --stage 3b --max-batch-size 512 --trajectories-threshold 128 \
  --minibatch-size 256)
COMMON=(--start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --sims 64 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --epochs-per-step 5 --train-block-epochs 50 --max-added-epochs 1000 --warmup-steps 10 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit)

echo "===== [$(date +%H:%M:%S)] make build ====="
make build

# --- PHASE 1: タイミングプローブ (各 100ep, 逐次) ---
probe() {  # $1=label $2=search_max
  local label=$1 smax=$2
  local ck="$PROBE_DIR/${label}.pt" t0 t1
  rm -f "$PROBE_DIR/${label}"*.pt
  echo "##### [$(date +%H:%M:%S)] PROBE $label (search-max $smax) start #####"
  t0=$(date +%s)
  uv run train-loop "${GEN[@]}" --search-turn-max "$smax" \
    --max-epochs 100 --checkpoint-path "$ck" --snapshot-every 1000
  t1=$(date +%s)
  echo "##### [$(date +%H:%M:%S)] PROBE $label done: wall=$((t1 - t0))s (100ep) #####"
}
echo "===== [$(date +%H:%M:%S)] PHASE 1 timing probes ====="
probe SMAX7 7
probe SMAX8 8
probe SMAX9 9

# --- PHASE 2: 強さ funnel (3アーム) ---
for smax in 7 8 9; do
  tag="SMAX${smax}"
  echo "===== [$(date +%H:%M:%S)] PHASE 2 funnel $tag (search-max $smax) ====="
  f0=$(date +%s)
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" "${COMMON[@]}" \
    --search-turn-max "$smax"
  echo "===== [$(date +%H:%M:%S)] funnel $tag wall=$(( $(date +%s) - f0 ))s ====="
done

# --- PHASE 3: rate (7 + 8 + 9) ---
echo "===== [$(date +%H:%M:%S)] PHASE 3 rate (SMAX7 + SMAX8 + SMAX9) random+crit ====="
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/SMAX7_finalists.json" "$T/SMAX8_finalists.json" "$T/SMAX9_finalists.json"

echo "===== [$(date +%H:%M:%S)] DONE ====="
