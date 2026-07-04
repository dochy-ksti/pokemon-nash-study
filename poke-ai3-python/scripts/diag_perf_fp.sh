#!/usr/bin/env bash
# フレームポインタ強制ビルド + --call-graph fp で RUN1/RUN2 を取り直す。
# dwarf 復元の破綻(__FRAME_END__ 丸め)を避け、両 run を同条件で関数名へ確実解決する。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=30; OUT=/tmp/diag_fp.pt

# フレームポインタ付きで再ビルド(release+debuginfo)。
export RUSTFLAGS="-C force-frame-pointers=yes"
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
echo ">>> rebuild with frame pointers $(date +%H:%M:%S)"
make build

rec() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] record $1"
  t0=$(date +%s)
  perf record -F 499 -g --call-graph fp -o "$T/fp_${1}.data" -- \
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
}

rec RUN1
rec RUN2

for L in RUN1 RUN2; do
  echo "===== $L flat self top25 (-n) ====="
  perf report -i "$T/fp_${L}.data" --stdio -g none -n 2>/dev/null \
    | grep -vE '^#|^$' | grep _native | head -25
  echo
done
echo "===== DONE ====="
