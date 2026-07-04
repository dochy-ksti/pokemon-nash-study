# sim_concurrency 並列化の検証ラン (2a-det)

## 目的

lookahead の rollout を `--sim-concurrency` でスライディングウィンドウ並列化した
新方式 (commit d60ad6d) で、前回パフォーマンス問題により 1 epoch も完了できなかった
学習ループが、実用的な速度で進行するかを検証する。学習の収束確認ではなく、
「epoch が回るか・速度が実用的か」を見る短いラン。

## コマンド

```bash
cd poke-ai3-python
uv run phase2-loop \
  --num-games 16 \
  --sim-concurrency 8 \
  --sims 64 \
  --search-turn-min 4 --search-turn-max 8 \
  --no-random --no-crit --stage 2a \
  --max-epochs 5 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2a_det_par.pt
```

- 環境: RTX 5090, CUDA。
- in-flight 最大 = `num_games * sim_concurrency` = 16 * 8 = 128。
  `chunk_threshold` も自動連動で 128 (`num_games * sim_concurrency`)。

## 結果

exit 0、5 epoch 完走。**前回 1 epoch も完了できなかったのに対し、スムーズに完走**。
並列化の主目的 (1 試合あたりの並列度を上げて GPU を飽和させ、パフォーマンス壁を突破)
は達成。

| epoch | examples | win_rate | correct_action_rate | value_loss | entropy | raw_logits_std |
|------:|---------:|---------:|--------------------:|-----------:|--------:|---------------:|
| 1 | 64 | 1.00  | 0.625 | 0.119 | 0.684 | 0.132 |
| 2 | 68 | 0.625 | 0.618 | 0.059 | 0.682 | 0.141 |
| 3 | 64 | 1.00  | 0.563 | 0.019 | 0.626 | 0.380 |
| 4 | 72 | 0.50  | 0.569 | 0.073 | 0.677 | 0.151 |
| 5 | 64 | 0.75  | 0.625 | 0.060 | 0.688 | 0.100 |

## 所見

- `value_loss` は下降傾向で値ネットは学習している。
- `correct_action_rate` は 0.56〜0.63 で横ばい、`entropy` も 0.68 付近で高いまま。
  5 epoch では方策はまだ収束しておらず、これは想定どおり (検証目的のため正常)。
- `vs_cloyster/goodra_special_rate` が epoch ごとに 0↔1 で振れるのはサンプル少
  (64〜72/epoch) によるノイズ。

## 評価

ポジティブ (仕組みの検証として)。並列方式で学習ループが実用速度で回ることを確認。
方策が正しい手へ収束するかは、この後の 300 epoch 本番ランで評価する。

## 次

同一チェックポイントから継続で 2a-det / num_games 16 / sim_concurrency 8 /
max-epochs 300 の本番ランを実行する。
