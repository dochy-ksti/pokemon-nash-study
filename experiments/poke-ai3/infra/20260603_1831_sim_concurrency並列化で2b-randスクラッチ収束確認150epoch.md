# sim_concurrency 並列化で 2b + random/crit のスクラッチ収束確認 (150 epoch)

## 目的

stage 2b は比率 1.57x で Phase 2 中最も信号が弱い。これに確率的設定 (16 段ダメージ乱数 +
急所) を重ね、事前学習なし (ゼロ初期化) で並列方式 (sim_concurrency=8) が収束するかを
確認する。最も難しい条件での頑健性テスト。

## コマンド

```bash
# 新規 (存在しない) チェックポイントパス = ゼロ初期化で開始
cd poke-ai3-python
uv run phase2-loop \
  --num-games 16 --sim-concurrency 8 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --random --crit --stage 2b \
  --max-epochs 150 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2b_rand_scratch_par.pt
```

- 環境: RTX 5090, CUDA。事前学習なし。

## 結果

exit 0、150 epoch 完走、エラーなし。

| epoch | correct_action_rate | vs_cloyster_special | vs_goodra_special | value_loss | entropy | raw_logits_std |
|------:|--------------------:|--------------------:|------------------:|-----------:|--------:|---------------:|
| 1   | 0.375 | 0.000 | 0.000 | 0.091 | 0.675 | -    |
| 15  | 0.713 | 0.000 | 0.000 | 0.036 | 0.690 | -    |
| 30  | 1.000 | 1.000 | 0.000 | 0.032 | 0.457 | -    |
| 60  | 1.000 | 1.000 | 0.000 | 0.023 | 0.255 | -    |
| 90  | 0.963 | 0.925 | 0.000 | 0.016 | 0.290 | -    |
| 150 | 1.000 | 1.000 | 0.000 | 0.005 | 0.277 | 1.398 |

- **epoch 27〜30 で correct_action_rate=1.0 に到達** (2a スクラッチの epoch 20 より
  やや遅いが大差なし)。
- 終盤 30 epoch のうち 20 回が 1.0、残りは 0.90〜0.99 とわずかに揺れる。2b の弱い信号
  (1.57x) + 乱数によるノイズで、撃ち分け方向 (Cloyster=特殊 / Goodra=非特殊) は終始正しい。
- entropy は ~0.28 で、2a スクラッチ (~0.18) より高め。信号が弱い分だけ方策の鋭さが
  控えめで、比率に対して妥当。
- value_loss は 0.005 付近まで低下。

## 評価

ポジティブ。Phase 2 中最も信号の弱い 2b に確率的設定 + スクラッチを重ねた最難条件でも、
並列方式 (sim_concurrency=8) は epoch 30 前後で収束し以降安定 (僅かなノイズあり)。
provisional 0.5 / 非決定的完了順を許容した設計の頑健性を確認。

## 並列方式の検証総括 (本セッション)

- 2a-det: epoch 25 前後で収束 (300 epoch)
- 2a + random/crit (det 継続): 全 50 epoch で 1.0 維持・安定
- 2a + random/crit (スクラッチ): epoch 20 前後で収束 (100 epoch)
- 2b + random/crit (スクラッチ): epoch 30 前後で収束 (150 epoch、僅かなノイズ)
- 前回 1 epoch も回らなかったパフォーマンス壁は解消

## 次の候補

- sim_concurrency を 16/32 に上げた際の収束速度・安定性・スループット。
- num_games を 1〜4 に絞り sim_concurrency を上げた「1 試合での GPU 飽和」実測。
