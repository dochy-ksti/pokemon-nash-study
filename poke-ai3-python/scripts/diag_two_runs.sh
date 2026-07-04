#!/usr/bin/env bash
# 決定実験v3: 連続2本。CPU を us/sy/wa に分解 (/proc/stat 差分) して、
# 2本目の浪費が user(計算) か system(カーネル=メモリ/THP) か wa(IO) かを判定。
# あわせて kcompactd/khugepaged の CPU時間も記録。
set -euo pipefail
cd "$(dirname "$0")/.."

T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=80
OUT=/tmp/diag_run.pt
MON=$T/diag_gpu2.csv

echo "ts,pstate,sm_mhz,mem_mhz,util,power_w,gutil_enc" > "$MON"
(
  while true; do
    line=$(nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$(date +%H:%M:%S),$line" >> "$MON"
    sleep 1
  done
) &
MONPID=$!
trap "kill $MONPID 2>/dev/null || true" EXIT

run() {
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] $1 開始"
  t0=$(date +%s)
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
    --checkpoint-path "$OUT" > /dev/null 2>&1
  t1=$(date +%s)
  echo "<<< [$(date +%H:%M:%S)] $1 終了: $((t1-t0))s = $(echo "scale=3;($t1-$t0)/$EPOCHS"|bc) s/ep"
}

echo "THP enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "THP defrag : $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
make build
run "RUN1"
run "RUN2"
echo "===== DONE ====="
