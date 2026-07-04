# Phase 2 Stage 2b-det — lookahead 学習の初回検証

## 目的

PPO+GAE を lookahead Monte-Carlo + Nash accumulation の教師あり学習へ全面置換した新方式が、
これまで PPO の `--random --crit` で soft-uniform に陥った **stage 2b (ratio 1.57x, 最狭)** で
学習できるかを検証する。まず決定論モードでクリーンな信号を取り、その後 2b-rand に進む。

新方式の要点:
- 各局面で全候補手を起点に policy に沿って終局までロールアウト、平均勝率を算出 (sims=64)。
- 最初の一手のみ候補手固定、以降は両者 policy net サンプル。終局未到達のみ value net 末端評価。
- Nash accumulation で training_pi / selection_pi、V=手ごと最大勝率。loss = CE(π)+MSE(V)。
- 自己対戦の両側を同一 net で推論、P1/P2 両方を学習サンプル化。

## コマンド

```bash
cd poke-ai3-python
uv run phase2-loop \
  --num-games 64 --trajectories-threshold 64 \
  --max-epochs 300 \
  --no-random --no-crit --stage 2b \
  --sims 64 --search-turn-min 4 --search-turn-max 8 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2b_det.pt
```

## 卒業条件 (目安)

- correct_action_rate → 0.99 付近で安定 (vs Cloyster で Dark Pulse、vs Goodra で Crunch)。
- entropy が ln(2)=0.693 から十分低下、raw_logits_std が育つ (soft-uniform でない)。

## 結果

(実行中 — 追記予定)
