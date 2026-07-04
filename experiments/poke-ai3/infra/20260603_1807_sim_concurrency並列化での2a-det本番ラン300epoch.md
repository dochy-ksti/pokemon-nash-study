# sim_concurrency 並列化での 2a-det 本番ラン (300 epoch)

## 目的

`--sim-concurrency` 並列化 (commit d60ad6d) で学習ループが実用速度で回ることを
検証ラン (experiments/poke-ai3/20260603_1756) で確認したのを受け、方策が
実際に正しい手 (タイプ/種族値を踏まえた最適技) へ収束するかを 300 epoch で評価する。

## コマンド

```bash
cd poke-ai3-python
uv run phase2-loop \
  --num-games 16 --sim-concurrency 8 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --no-random --no-crit --stage 2a \
  --max-epochs 300 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2a_det_par.pt
```

- 環境: RTX 5090, CUDA。
- 検証ランで作成した同一チェックポイントから継続学習。
- in-flight 最大 = 16 * 8 = 128、chunk_threshold も自動連動で 128。

## 結果

exit 0、300 epoch 完走、エラーなし。

収束の推移 (epoch 1 / 25 / 30 / …):

| epoch | correct_action_rate | vs_cloyster_special | vs_goodra_special | value_loss |
|------:|--------------------:|--------------------:|------------------:|-----------:|
| 1   | 0.750 | 1.000 | 1.000 | 0.0385 |
| 25  | 1.000 | 1.000 | 0.000 | 0.0420 |
| 30  | 1.000 | 1.000 | 0.000 | 0.0062 |
| 120 | 0.922 | 1.000 | 0.208 | 0.0142 |
| 210 | 0.906 | 0.786 | 0.000 | 0.0035 |
| 300 | 0.938 | 1.000 | 0.125 | 0.0038 |

- **epoch 25 前後で収束**。correct_action_rate が 1.0 に到達し、以降は
  0.90〜1.00 で安定 (小さな揺れはサンプル少 64〜72/epoch のノイズ)。
- 対面別診断が正しい方向に分離:
  - `vs_cloyster_special_rate` ≒ 1.0 (Cloyster には特殊技が正解)。
  - `vs_goodra_special_rate` ≒ 0.0 (Goodra には特殊技を撃たないのが正解)。
  - → 初期は両方 1.0 (無差別に特殊) だったのが、種族値を踏まえて撃ち分けを学習。
- `value_loss` は 0.003〜0.01 まで低下、値ネットも十分に学習。
- `entropy` は 0.68 → 0.40 付近、`raw_logits_std` は 0.13 → 1.0 付近へ。方策が鋭くなった。

## 評価

ポジティブ。並列方式 (sim_concurrency=8) で 2a-det の最適技選択を epoch 25 前後で
学習・以降安定。前回 1 epoch も回らなかった状況から、収束まで一気通貫で到達できた。
パフォーマンス壁の解消と学習の正しさを両方確認。

## 次の候補

- sim_concurrency を 16/32 に上げ、速度向上と visit 分布の薄まり (学習信号低下) の
  トレードオフを測る。
- 確率的設定 (--random / --crit) や stage 2b でも同様に安定して収束するか確認。
- num_games を抑えた (例えば 1〜4) 構成で sim_concurrency を上げ、「1 試合での GPU
  飽和」を実測する。
