#!/usr/bin/env bash
# 敵混合学習 enemy-window K 掃引ドライバ (K2→K3→rate)。
# CLI から完全に切り離して実行できるよう自己完結。既存 finalists.json がある K は skip。
# 使い方 (SSH/Windows 切断後も継続):
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_k_sweep.sh > "$LOGDIR/k_sweep_driver.log" 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
TDIR=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/ce031612-7617-405e-b16e-c0374298d32c/scratchpad
mkdir -p "$LOGDIR"

echo "[driver] start $(date -Is)"

# 既存の途中 K2/K3 funnel があれば止める (同名ファイル二重書き防止)。
pkill -f "ckpt_tournament.py funnel --tag K2" 2>/dev/null
pkill -f "ckpt_tournament.py funnel --tag K3" 2>/dev/null
sleep 2

echo "[driver] make build"
make build > "$LOGDIR/k_sweep_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

run_funnel() {
  local tag="$1" win="$2"
  if [ -f "$TDIR/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"
    return 0
  fi
  echo "[driver] === $tag (enemy-window $win) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window "$win" \
    --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 \
    --sims 64 --sim-concurrency 16 --train-num-games 64 --train-max-batch-size 512 \
    --train-trajectories-threshold 128 --train-minibatch-size 256 \
    --epochs-per-step 5 --train-block-epochs 50 --warmup-steps 10 \
    --peaks-per-rr 3 --finalists-target 3 --max-added-epochs 1000 \
    --n-per-side 512 --num-games 256 --stage 3b --random --crit \
    > "$LOGDIR/${tag}.log" 2>&1
  local ec=$?
  echo "[driver] === $tag done exit=$ec $(date -Is) ==="
  return $ec
}

run_funnel K2 2 || { echo "[driver] K2 failed"; exit 1; }
run_funnel K3 3 || { echo "[driver] K3 failed"; exit 1; }

echo "[driver] === rate (RC64+K1+K2+K3) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$TDIR/RC64_finalists.json" "$TDIR/K1_finalists.json" \
                "$TDIR/K2_finalists.json" "$TDIR/K3_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/k_sweep_rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
