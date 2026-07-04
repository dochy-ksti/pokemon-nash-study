#!/usr/bin/env bash
# perf stat で 2連続 run の IPC とキャッシュ/TLB ミスを比較。
# 同一命令数なら IPC 崩壊 (cycles/instr 増) とミス増がメモリ要因の証拠。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; EPOCHS=40; OUT=/tmp/diag_ipc.pt
EV=instructions,cycles,cache-references,cache-misses,LLC-load-misses,dTLB-load-misses,L1-dcache-load-misses
make build
run() {  # $1=label
  cp "$T/shared_init.pt" "$OUT"
  echo ">>> [$(date +%H:%M:%S)] $1"
  perf stat -e "$EV" -o "$T/ipc_${1}.txt" -- \
    uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
      --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
      --no-random --no-crit --stage 3b --max-epochs "$EPOCHS" \
      --checkpoint-path "$OUT" > /dev/null 2>&1
  echo "--- $1 perf stat ---"; grep -E 'instructions|cycles|cache-misses|LLC|dTLB|L1-dcache|insn per' "$T/ipc_${1}.txt"
}
run RUN1
run RUN2
echo "===== DONE ====="
