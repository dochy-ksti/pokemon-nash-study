#!/usr/bin/env bash
# perf で 2連続 run を別々に record し、関数別 self CPU 時間を比較する。
# release に line-tables デバッグ情報を付けて Rust シンボルを解決する。
# 事前に: sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=45; OUT=/tmp/diag_perf.pt

export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
make build

run() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] $1 perf record 開始"
  t0=$(date +%s)
  perf record -F 999 -o "$T/perf_${1}.data" -- \
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1 終了: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
  perf report -i "$T/perf_${1}.data" --stdio -g none --sort=symbol --percent-limit 0.5 2>/dev/null \
    | grep -vE '^#|^$' | sed -E 's/ +\[\./ \[./; s/[[:space:]]+/ /g' > "$T/perf_${1}.flat"
}
run RUN1
run RUN2
echo "===== DONE ====="
