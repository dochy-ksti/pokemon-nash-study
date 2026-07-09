#!/usr/bin/env bash
# メタ Nash PSRO パイロット (double-oracle)。集団 Π = 中心スナップショット列。中心学習者
# 1 本を warm-start 継続で育て、毎 iter「旧 Π の σ 混合」を相手に central_epochs 学習した
# 中心をスナップショットして Π に積む。σ = Π 総当り勝率行列の対称ゼロサム Nash。
# exploitability = 新中心 c_k が旧 σ 混合に取る勝率 (= Σ σ_prev·winrate(c_k vs Π_i))。
# c_k は旧混合への best-response なので、これが 0.5 へ近づけば double-oracle gap が閉じて
# Nash へ収束した証拠。解 = Π 上の σ 混合 (単一ネットでなく混合戦略が Nash 解)。
#
# 敵 (旧 Π の相手取る部分) の各ゲームは EnemySampler が (game_id, game_index) 単位で σ 比に
# 厳密配分する (旧: game_id スロット固定 → 実効シェアが σ÷平均ゲーム長 に歪むバグを解消)。
#
# 規模: 中心 warmup 200ep → 以降 50ep/iter、行列 matrix-n-per-side=256戦/ペア、6 iter、
# shared_init 発、value 教師 expected。学習/探索は A/B と同一 (nash_lr 1.5, depth_skew 2.0,
# search-turn 4-8, sims 64, sim_concurrency 16, train_num_games 64, stage 3b, random, crit)。
# 使い方:
#   cd poke-ai3-python
#   setsid nohup bash scripts/run_psro_pilot.sh > /tmp/psro/driver.log 2>&1 &
# 冪等: <tag>_psro.json があれば skip、途中 state があれば --resume で継続。
# env: TAG=tag, MAX_ITERS=n, META=nash|latest (既定 nash), SPR=self-play-ratio (既定 0=教科書),
#   ELA=1 で敵探索あり (敵も lookahead + σ 行列も探索込み。学習と評価の探索有無を揃える。
#   コスト ~2 倍)、MNPS=matrix-n-per-side (既定 256。探索込み行列のコスト調整用に下げられる)。
#   nash=σ サポートを σ 比で敵に。latest=直近 pool_size 個一様 (忘却ありのベースライン)。
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
  # ELA=1 で学習中の敵探索あり (敵も lookahead)。MLA=1 で σ 行列も探索込み (高コスト、iter² で
  # 効く)。既定は行列 policy-only (最終 σ 混合の探索込み exploitability は別途 1 回測る)。
  local ela=()
  [ "${ELA:-0}" = "1" ] && ela+=(--enemy-lookahead)
  [ "${MLA:-0}" = "1" ] && ela+=(--matrix-lookahead)
  echo "[driver] === $tag start $(date -Is) (meta=${META:-nash} ela=${ELA:-0} mla=${MLA:-0}) ==="
  uv run python scripts/ckpt_tournament.py psro --tag "$tag" \
    --shared-init "$PUB/shared_init.pt" "${resume[@]}" "${ela[@]}" \
    --meta-strategy "${META:-nash}" --nash-eps 0.02 --matrix-n-per-side "${MNPS:-256}" \
    --max-iters "${MAX_ITERS:-6}" --warmup-epochs 200 --central-epochs 50 \
    --pool-size 4 --self-play-ratio "${SPR:-0}" \
    --value-target expected --nash-learning-rate 1.5 \
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
