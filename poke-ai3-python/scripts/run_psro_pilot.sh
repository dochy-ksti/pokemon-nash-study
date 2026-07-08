#!/usr/bin/env bash
# PSRO パイロット: 中心学習者 1 本を育てつつ、毎 iter その時点の中心を凍結 target に
# した専用 best-response (exploiter) を作り、敵プール (最新 N) に積む。exploiter の対
# target 勝率 (exploitability) が iter を追って 50% へ下がれば穴が塞がった証拠。
#
# 規模 (grill-me で確定): 中心 50ep/iter・exploiter +50ep(eval ep25/50)・N=4・
# self_play_ratio 0.5・6 iter・shared_init 発・value 教師 expected。
# 学習/探索は A/B と同一 (nash_lr 1.5, depth_skew 2.0, search-turn 4-8, sims 64,
# sim_concurrency 16, train_num_games 64, stage 3b, random, crit)。
# 使い方:
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_psro_pilot.sh > /tmp/psro/driver.log 2>&1 &
# 冪等: <tag>_psro.json があれば skip、途中 state があれば --resume で継続。
set -u

cd "$(dirname "$0")/.." || exit 1
PUB="$(cd .. && pwd)/data/poke-ai3/tournament"
LOGDIR="${LOGDIR:-/tmp/psro}"
mkdir -p "$LOGDIR" "$PUB"

echo "[driver] start $(date -Is)  PUB=$PUB  LOGDIR=$LOGDIR"
[ -f "$PUB/shared_init.pt" ] || { echo "[driver] missing $PUB/shared_init.pt"; exit 1; }

echo "[driver] make build"
make build > "$LOGDIR/build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

TAG="${TAG:-PSRO_p1}"

run_psro() {
  local tag="$1"
  if [ -f "$PUB/${tag}_psro.json" ]; then
    echo "[driver] $tag already has psro.json -> skip"; return 0
  fi
  local resume=()
  if [ -f "$PUB/${tag}_psro_state.json" ]; then
    echo "[driver] $tag state.json あり -> --resume で継続"
    resume=(--resume)
  fi
  echo "[driver] === $tag start $(date -Is) ==="
  uv run python scripts/ckpt_tournament.py psro --tag "$tag" \
    --shared-init "$PUB/shared_init.pt" "${resume[@]}" \
    --max-iters 6 --central-epochs 50 \
    --exploiter-epochs 50 --exploiter-eval-every 25 \
    --pool-size 4 --self-play-ratio 0.5 \
    --value-target expected --nash-learning-rate 1.5 \
    --exploiter-battle-seed 20260711 \
    --depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 \
    --sims 64 --sim-concurrency 16 --train-num-games 64 --train-max-batch-size 512 \
    --train-trajectories-threshold 128 --train-minibatch-size 256 \
    --n-per-side 512 --num-games 256 --stage 3b --random --crit \
    > "$LOGDIR/${tag}.log" 2>&1
  local ec=$?
  echo "[driver] === $tag done exit=$ec $(date -Is) ==="
  return $ec
}

run_psro "$TAG" || { echo "[driver] $TAG failed"; exit 1; }

echo "[driver] === summary $(date -Is) ==="
if [ -f "$PUB/${TAG}_psro.json" ]; then
  uv run python -c "import json; d=json.load(open('$PUB/${TAG}_psro.json')); print('[driver] %s: iters=%s exploitability推移(iter,wr)=%s'%(d['tag'], d['iters'], [[c[0],c[2]] for c in d['curve']]))"
fi
echo "[driver] ALL DONE $(date -Is)"
