#!/usr/bin/env bash
# 捨て1本目 → 2本目を call-graph(dwarf) 付きで短く record し、ハッシュの呼び出し元を特定。
set -uo pipefail
cd "$(dirname "$0")/.."
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament
MAX=7; OUT=/tmp/diag_cg.pt
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
make build
echo ">>> 捨て1本目 (warmup, perfなし) $(date +%H:%M:%S)"
cp "$T/shared_init.pt" "$OUT"
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
  --no-random --no-crit --stage 3b --max-epochs 18 \
  --checkpoint-path "$OUT" > /dev/null 2>&1
echo ">>> 2本目 perf record (dwarf) $(date +%H:%M:%S)"
cp "$T/shared_init.pt" "$OUT"
perf record -F 499 -g --call-graph dwarf,16384 -o "$T/perf_cg.data" -- \
  uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
    --search-turn-min 4 --search-turn-max "$MAX" --depth-skew 2.0 \
    --no-random --no-crit --stage 3b --max-epochs 18 \
    --checkpoint-path "$OUT" > /dev/null 2>&1
echo ">>> DONE $(date +%H:%M:%S)"
