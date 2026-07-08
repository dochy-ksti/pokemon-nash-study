#!/usr/bin/env bash
# value 教師 A/B: max (現行=手ごと最大勝率) vs expected (均衡混合 training_pi の期待勝率)。
# max はゼロサム同時手番で構造的に均衡値以上へ出る (自分の勝率を過大評価する) ため、
# expected で均衡値へ較正しても強さが落ちないかを funnel→rate の Bradley-Terry で比較する。
#
# 対応比較 (Q5): 各 seed で両アームが (a) shared_init.pt を共有し (b) 同一の
# --train-battle-seed を使う。差は value 教師の式だけ。まず 1 seed (計2 funnel)、
# 差が微妙で時間があれば SEEDS に追加してペアを増やす。
# 冪等: 既存 _finalists.json はスキップ。CLI 切断後も継続できるよう自己完結。
# 使い方:
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_value_target_ab.sh > /tmp/vtab/driver.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1
PUB="$(cd .. && pwd)/data/poke-ai3/tournament"
LOGDIR="${LOGDIR:-/tmp/vtab}"
mkdir -p "$LOGDIR" "$PUB"

echo "[driver] start $(date -Is)  PUB=$PUB  LOGDIR=$LOGDIR"
[ -f "$PUB/shared_init.pt" ] || { echo "[driver] missing $PUB/shared_init.pt"; exit 1; }

echo "[driver] make build"
make build > "$LOGDIR/build.log" 2>&1 || { echo "[driver] build failed"; exit 1; }

# seed s に対応する固定 battle_seed (両アーム共通)。追加 seed はここに足す。
declare -A BSEED=( [1]=20260708 [2]=20260709 [3]=20260710 )
SEEDS=(1)

run_funnel() {
  local tag="$1" vt="$2" bseed="$3"
  if [ -f "$PUB/${tag}_finalists.json" ]; then
    echo "[driver] $tag already has finalists.json -> skip"; return 0
  fi
  echo "[driver] === $tag (value-target $vt battle-seed $bseed) start $(date -Is) ==="
  rm -f "$PUB/${tag}_"* "$PUB/${tag}.pt"
  uv run python scripts/ckpt_tournament.py funnel --tag "$tag" \
    --start "$PUB/shared_init.pt" --enemy-window 1 --self-play-ratio 0.5 \
    --nash-learning-rate 1.5 --value-target "$vt" --train-battle-seed "$bseed" \
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

# 実験一つずつ (AGENTS: benchmarks one at a time)。seed ごとに max→expected をペアで。
for s in "${SEEDS[@]}"; do
  b="${BSEED[$s]}"
  run_funnel "VMAX_s${s}" max      "$b" || { echo "[driver] VMAX_s${s} failed"; exit 1; }
  run_funnel "VEXP_s${s}" expected "$b" || { echo "[driver] VEXP_s${s} failed"; exit 1; }
done

echo "[driver] === rate (max vs expected 総当たり) $(date -Is) ==="
JSONS=()
for s in "${SEEDS[@]}"; do
  JSONS+=( "$PUB/VMAX_s${s}_finalists.json" "$PUB/VEXP_s${s}_finalists.json" )
done
uv run python scripts/ckpt_tournament.py rate \
  --funnel-json "${JSONS[@]}" \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit \
  > "$LOGDIR/rate.log" 2>&1
echo "[driver] rate exit=$? $(date -Is)"
echo "[driver] ALL DONE $(date -Is)"
