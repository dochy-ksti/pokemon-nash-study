#!/usr/bin/env bash
# nash_learning_rate 3seed 複製: 1.25 / 1.5 / 2.0 を各3本 (計9 funnel) 回して
# run 単位で独立標本化し、有意差を判定する。battle_seed は funnel が None→毎回ランダム
# なので同一コマンドの反復で独立 seed が得られる (train_loop.py: secrets.randbits(64))。
# 全 run は shared_init.pt を共有し --nash-learning-rate だけを変える。冪等: 既存
# _finalists.json はスキップ。CLI 切断後も継続できるよう自己完結。
# 使い方:
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_nash_lr_3seed.sh > /tmp/nash3/driver.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PUB="$(cd .. && pwd)/data/poke-ai3/tournament"
LOGDIR="${LOGDIR:-/tmp/nash3}"
mkdir -p "$LOGDIR" "$PUB"

echo "[driver] start $(date -Is)  PUB=$PUB  LOGDIR=$LOGDIR"
[ -f "$PUB/shared_init.pt" ] || { echo "[driver] missing $PUB/shared_init.pt"; exit 1; }

echo "[driver] make build"
make build > "$LOGDIR/build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

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

# 実験一つずつ (AGENTS: benchmarks one at a time)。アーム内 seed を順に。
for s in 1 2 3; do
  run_funnel "NLR12_s${s}" 1.25 || { echo "[driver] NLR12_s${s} failed"; exit 1; }
  run_funnel "NLR15_s${s}" 1.5  || { echo "[driver] NLR15_s${s} failed"; exit 1; }
  run_funnel "NLR20_s${s}" 2.0  || { echo "[driver] NLR20_s${s} failed"; exit 1; }
done

echo "[driver] === rate (9 finalists 総当たり) $(date -Is) ==="
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json \
    "$PUB/NLR12_s1_finalists.json" "$PUB/NLR12_s2_finalists.json" "$PUB/NLR12_s3_finalists.json" \
    "$PUB/NLR15_s1_finalists.json" "$PUB/NLR15_s2_finalists.json" "$PUB/NLR15_s3_finalists.json" \
    "$PUB/NLR20_s1_finalists.json" "$PUB/NLR20_s2_finalists.json" "$PUB/NLR20_s3_finalists.json" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
