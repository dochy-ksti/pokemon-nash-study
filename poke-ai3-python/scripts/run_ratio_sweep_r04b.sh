#!/usr/bin/env bash
# 敵混合 自己対戦比率 r=0.4 再現ラン (K1r04b funnel → rate)。切断分離実行。
# shared_init は同一だが battle_seed はプロセス毎に新規 → 独立な複製標本。
set -u
cd "$(dirname "$0")/.." || exit 1
TDIR=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
LOGDIR=/tmp/claude-1000/-home-dochy-pokemon-ai-proj/ce031612-7617-405e-b16e-c0374298d32c/scratchpad
mkdir -p "$LOGDIR"
echo "[driver] start $(date -Is)"
pkill -f "ckpt_tournament.py funnel --tag K1r04b" 2>/dev/null
sleep 2
echo "[driver] make build"
make build > "$LOGDIR/ratio_r04b_build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

run_funnel() {
  local tag="$1" ratio="$2"
  if [ -f "$TDIR/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"; return 0
  fi
  echo "[driver] === $tag (self-play-ratio $ratio) start $(date -Is) ==="
  rm -f "$TDIR/${tag}_"* "$TDIR/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$TDIR/shared_init.pt" --enemy-window 1 --self-play-ratio "$ratio" \
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

run_funnel K1r04b 0.4 || { echo "[driver] K1r04b failed"; exit 1; }

echo "[driver] === rate (RC64+K1+K1r04+K1r04b+他) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "$TDIR/RC64_finalists.json" "$TDIR/K1_finalists.json" \
                "$TDIR/K1r06_finalists.json" "$TDIR/K1r04_finalists.json" \
                "$TDIR/K1r03_finalists.json" "$TDIR/K1r045_finalists.json" \
                "$TDIR/K1r035_finalists.json" "$TDIR/K1r04b_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/ratio_r04b_rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
