# sim_concurrency 並列化で 2a + random/crit のスクラッチ収束確認 (100 epoch)

## 目的

前ラン (experiments/poke-ai3/20260603_1815) は det 収束済みモデルからの継続で安定性を
確認した。本ランは事前学習なし (ゼロ初期化) で、最初から確率的設定 (16 段ダメージ乱数 +
急所) のもとで並列方式 (sim_concurrency=8) が収束するかを確認する。

## コマンド

```bash
# 新規 (存在しない) チェックポイントパス = ゼロ初期化で開始
cd poke-ai3-python
uv run phase2-loop \
  --num-games 16 --sim-concurrency 8 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --random --crit --stage 2a \
  --max-epochs 100 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2a_rand_scratch_par.pt
```

- 環境: RTX 5090, CUDA。事前学習なし。

## 結果

exit 0、100 epoch 完走、エラーなし。

| epoch | correct_action_rate | vs_cloyster_special | vs_goodra_special | value_loss | entropy | raw_logits_std |
|------:|--------------------:|--------------------:|------------------:|-----------:|--------:|---------------:|
| 1   | 0.688 | 1.000 | 1.000 | 0.113 | 0.636 | -    |
| 10  | 0.556 | 0.000 | 0.000 | 0.044 | 0.692 | -    |
| 20  | 1.000 | 1.000 | 0.000 | 0.053 | 0.453 | -    |
| 30  | 1.000 | 1.000 | 0.000 | 0.012 | 0.205 | -    |
| 50  | 1.000 | 1.000 | 0.000 | 0.011 | 0.186 | -    |
| 100 | 1.000 | 1.000 | 0.000 | 0.015 | 0.183 | 1.557 |

- 序盤 (epoch 1〜10) は探索的に振れる (epoch 10 では両診断とも 0.0 の過渡状態) が、
  **epoch 20〜26 で correct_action_rate=1.0 に到達し、以降 100 epoch まで完全に安定**。
- 撃ち分け (Cloyster=特殊 / Goodra=非特殊) を正しく獲得。`entropy` ~0.18、
  `raw_logits_std` ~1.6 と確信的。
- 収束速度は決定論スクラッチ (約 25 epoch) とほぼ同等。乱数は収束を阻害しなかった。

## 評価

ポジティブ。並列方式 (sim_concurrency=8) は事前学習なしでも確率的設定 (--random --crit)
のもとで 2a の最適技選択をゼロから学習・収束させられる。provisional 0.5 / 非決定的完了順
を許容した設計でも、スクラッチ学習の収束性に問題なし。

## 次の候補

- stage 2b (比率 1.57x、信号が弱い) + random/crit でのスクラッチ収束。
- sim_concurrency を 16/32 に上げた際の収束速度・安定性・スループット。
- num_games を 1〜4 に絞り sim_concurrency を上げた「1 試合での GPU 飽和」実測。
