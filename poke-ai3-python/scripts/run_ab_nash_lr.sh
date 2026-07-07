#!/usr/bin/env bash
# nash_learning_rate A/B ドライバ (K1 r0.5 固定, NLR20 vs NLR15 funnel → rate)。
# 両アームは shared_init.pt を共有し、--nash-learning-rate だけを変える。
# CLI から切り離して実行できるよう自己完結。既存 finalists.json があれば skip。
# 使い方 (SSH/Windows 切断後も継続):
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_ab_nash_lr.sh > "$LOGDIR/ab_nash_lr_driver.log" 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
TDIR="$(cd .. && pwd)/data/poke-ai3/tournament"
LOGDIR="${LOGDIR:-/tmp/ab_nash_lr}"
mkdir -p "$LOGDIR" "$TDIR"

echo "[driver] start $(date -Is)  TDIR=$TDIR  LOGDIR=$LOGDIR"

if [ ! -f "$TDIR/shared_init.pt" ]; then
  echo "[driver] missing $TDIR/shared_init.pt -> abort"; exit 1
fi

echo "[driver] make build"
make build > "$LOGDIR/build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

run_funnel() {
  local tag="$1" nlr="$2"
  if [ -f "$TDIR/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"
    return 0
  fi
  echo "[driver] === $tag (nash-learning-rate $nlr) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window 1 --self-play-ratio 0.5 \
    --nash-learning-rate "$nlr" \
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

# 実験一つずつ (AGENTS: benchmarks one at a time)。
run_funnel NLR20 2.0 || { echo "[driver] NLR20 failed"; exit 1; }
run_funnel NLR15 1.5 || { echo "[driver] NLR15 failed"; exit 1; }

echo "[driver] === rate (NLR20 vs NLR15) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$TDIR/NLR20_finalists.json" "$TDIR/NLR15_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
