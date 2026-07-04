#!/usr/bin/env bash
# 即時 head-to-head (block=epochs_per_step=5) で K=1/2/3 を r=0.5 固定で逐次実行 → rate。
# per-enemy ログ (enemy_by[1個前/2個前/3個前]) を各ランで取得する。
# tmux セッション内で実行する想定 (Windows/SSH 切断後も tmux サーバが継続)。
set -u
cd "$(dirname "$0")/.." || exit 1
TDIR=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/1c773ca7-11e5-4e30-aa12-aabf59b748a9/scratchpad
mkdir -p "$LOGDIR"
echo "[driver] start $(date -Is)"

echo "[driver] make build"
make build > "$LOGDIR/ksweep_block5_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

run_funnel() {
  local tag="$1" win="$2"
  if [ -f "$TDIR/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"; return 0
  fi
  echo "[driver] === $tag (enemy-window $win, r=0.5, block=5 即時HtH) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  # --train-block-epochs 省略 => block_epochs = epochs_per_step = 5 (即時 head-to-head)。
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window "$win" --self-play-ratio 0.5 \
    --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 \
    --sims 64 --sim-concurrency 16 --train-num-games 64 --train-max-batch-size 512 \
    --train-trajectories-threshold 128 --train-minibatch-size 256 \
    --epochs-per-step 5 --warmup-steps 10 \
    --peaks-per-rr 3 --finalists-target 3 --max-added-epochs 1000 \
    --n-per-side 512 --num-games 256 --stage 3b --random --crit \
    > "$LOGDIR/${tag}.log" 2>&1
  local ec=$?
  echo "[driver] === $tag done exit=$ec $(date -Is) ==="
  return $ec
}

# AGENTS.md: 実験は一度に1つ。逐次実行。
run_funnel K1b5 1 || { echo "[driver] K1b5 failed"; exit 1; }
run_funnel K2b5 2 || { echo "[driver] K2b5 failed"; exit 1; }
run_funnel K3b5 3 || { echo "[driver] K3b5 failed"; exit 1; }

echo "[driver] === rate (K1b5+K2b5+K3b5) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$TDIR/K1b5_finalists.json" "$TDIR/K2b5_finalists.json" \
                "$TDIR/K3b5_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/ksweep_block5_rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
