#!/usr/bin/env bash
# 即時 head-to-head baseline (K1 r=0.5, --train-block-epochs 省略 => block=epochs_per_step=5)。
# per-enemy ログ (enemy_by[...]) を確認しつつ、今後の block=5 実験の比較基準を作る。切断分離実行。
set -u
cd "$(dirname "$0")/.." || exit 1
TDIR=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/1c773ca7-11e5-4e30-aa12-aabf59b748a9/scratchpad
mkdir -p "$LOGDIR"
echo "[driver] start $(date -Is)"
pkill -f "ckpt_tournament.py funnel --tag K1b5" 2>/dev/null
sleep 2
echo "[driver] make build"
make build > "$LOGDIR/baseline_block5_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

tag=K1b5
if [ -f "$TDIR/${tag}_finalists.json" ]; then
  echo "[driver] $tag already has finalists.json -> skip"
else
  echo "[driver] === $tag (block 省略=5, r=0.5, 即時HtH) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  # --train-block-epochs を渡さない => block_epochs = epochs_per_step = 5 (即時 head-to-head)。
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window 1 --self-play-ratio 0.5 \
    --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 \
    --sims 64 --sim-concurrency 16 --train-num-games 64 --train-max-batch-size 512 \
    --train-trajectories-threshold 128 --train-minibatch-size 256 \
    --epochs-per-step 5 --warmup-steps 10 \
    --peaks-per-rr 3 --finalists-target 3 --max-added-epochs 1000 \
    --n-per-side 512 --num-games 256 --stage 3b --random --crit \
    > "$LOGDIR/${tag}.log" 2>&1
  echo "[driver] === $tag done exit=$? $(date -Is) ==="
fi
echo "[driver] ALL DONE $(date -Is)"
