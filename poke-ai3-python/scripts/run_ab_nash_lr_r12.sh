#!/usr/bin/env bash
# nash_learning_rate 追試: NLR12 (1.25) を1本回し、既存 NLR20(2.0)/NLR15(1.5) と
# 3手法総当たり rate。NLR20/NLR15 の finalists は path 修正前に生成されたため旧 private
# にある点に注意 (PRIV)。NLR12 は修正後なので public TDIR (PUB) に出る。
# 使い方:
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_ab_nash_lr_r12.sh > "$LOGDIR/r12_driver.log" 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PUB="$(cd .. && pwd)/data/poke-ai3/tournament"
PRIV=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR="${LOGDIR:-/tmp/ab_nash_lr}"
mkdir -p "$LOGDIR" "$PUB"

echo "[driver] start $(date -Is)  PUB=$PUB"

[ -f "$PUB/shared_init.pt" ] || { echo "[driver] missing $PUB/shared_init.pt"; exit 1; }

echo "[driver] make build"
make build > "$LOGDIR/r12_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

run_funnel() {
  local tag="$1" nlr="$2"
  if [ -f "$PUB/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"; return 0
  fi
  echo "[driver] === $tag (nash-learning-rate $nlr) start $(date -Is) ==="
  rm -f "$PUB/${tag}_"* "$PUB/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$PUB/shared_init.pt" --enemy-window 1 --self-play-ratio 0.5 \
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

run_funnel NLR12 1.25 || { echo "[driver] NLR12 failed"; exit 1; }

echo "[driver] === rate (NLR20 vs NLR15 vs NLR12) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$PRIV/NLR20_finalists.json" "$PRIV/NLR15_finalists.json" \
                "$PUB/NLR12_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/r12_rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
