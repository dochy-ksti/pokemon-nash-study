#!/usr/bin/env bash
# 敵も先読み (--enemy-lookahead) の効果測定: K1b5L (K=1, r=0.5, block=5, 敵先読み) を回し、
# 既存 K1b5 (敵 policy-only baseline) と rate 比較。tmux 内で実行 (切断耐性)。
set -u
cd "$(dirname "$0")/.." || exit 1
TDIR=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/1c773ca7-11e5-4e30-aa12-aabf59b748a9/scratchpad
mkdir -p "$LOGDIR"
echo "[driver] start $(date -Is)"

echo "[driver] make build"
make build > "$LOGDIR/enemy_lookahead_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

tag=K1b5L
if [ -f "$TDIR/${tag}_finalists.json" ]; then
  echo "[driver] $tag already has finalists.json -> skip"
else
  echo "[driver] === $tag (K=1 r=0.5 block=5 敵先読み) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window 1 --self-play-ratio 0.5 --enemy-lookahead \
    --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 \
    --sims 64 --sim-concurrency 16 --train-num-games 64 --train-max-batch-size 512 \
    --train-trajectories-threshold 128 --train-minibatch-size 256 \
    --epochs-per-step 5 --warmup-steps 10 \
    --peaks-per-rr 3 --finalists-target 3 --max-added-epochs 1000 \
    --n-per-side 512 --num-games 256 --stage 3b --random --crit \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "[driver] === $tag done exit=$? $(date -Is) ==="
fi

echo "[driver] === rate (K1b5 policy-only vs K1b5L 敵先読み) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$TDIR/K1b5_finalists.json" "$TDIR/K1b5L_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/enemy_lookahead_rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
