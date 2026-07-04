#!/usr/bin/env bash
# B history-funnel + 最終 rate を tmux 内で実行する (A は完了済み)。
set -u
cd /home/dochy/pokemon_ai_proj/poke-ai3-python
T=../data/poke-ai3/tournament
COMMON="--peaks-per-primary 3 --secondaries-target 2 --epochs-per-step 20 --back-offsets 20,40,80 --max-added-epochs 2000 --n-per-side 512 --num-games 64 --stage 3b"
{
  echo "==== B history-funnel start $(TZ=Asia/Tokyo date) ===="
  POKE_AI3_SKIP_FRESH_CHECK=1 uv run python scripts/ckpt_tournament.py funnel \
    --tag B --start "$T/shared_init.pt" $COMMON --depth-skew 1.0 --search-turn-min 6 --search-turn-max 12
  echo "==== rate start $(TZ=Asia/Tokyo date) ===="
  POKE_AI3_SKIP_FRESH_CHECK=1 uv run python scripts/ckpt_tournament.py rate \
    --n-per-side 512 --num-games 64 --stage 3b \
    --funnel-json "$T/A_secondaries.json" "$T/B_secondaries.json"
  echo "==== ALL DONE $(TZ=Asia/Tokyo date) ===="
} > /home/dochy/pokemon_ai_proj/data/poke-ai3/tournament/ab_history.log 2>&1
