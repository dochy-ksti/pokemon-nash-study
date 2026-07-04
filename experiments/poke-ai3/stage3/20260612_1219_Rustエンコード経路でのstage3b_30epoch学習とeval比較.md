# Rust エンコード経路での stage 3b 30 epoch 学習と eval-vs-rule 比較

実施: 2026-06-12 12:19 JST

## 目的

Rust エンコード化 (bbe5ec2) 後の新経路で stage 3b を 30 epoch 学習し
(`stage3b_fast30.pt`)、eval-vs-rule で現行 `stage3b_long_xteam.pt` (0.564) と比較する
(設計判断 docs/poke-ai3/20260611_2359 の完了条件 3)。

## コマンド

学習 (3:53 で完走):

```bash
cd poke-ai3-python
uv run train-loop --num-games 32 --sim-concurrency 256 --sims 512 \
  --search-turn-min 4 --search-turn-max 8 --no-random --no-crit --stage 3b \
  --max-epochs 30 --max-batch-size 8192 \
  --checkpoint-path ../data/poke-ai3/stage3b_fast30.pt
```

評価 (long_xteam の評価と同条件, n=512):

```bash
uv run eval-vs-rule --checkpoint-path ../data/poke-ai3/stage3b_fast30.pt \
  --stage 3b --num-games 16 --num-eval-games 512 --sims 64 --sim-concurrency 16 --no-random --no-crit
```

ログ: `logs_stage3b_fast30.log` / `logs_eval_stage3b_fast30.log`

## 結果

| モデル | 学習量 | 全体 | 有利(SE可) | 不利/等倍 |
|---|---|---|---|---|
| stage3b_long_xteam (基準) | 300 ep × 32 games, sims 64, turn 6-12 | **0.564** | 0.576 | 0.553 |
| stage3b_fast30 | 30 ep × 32 games, sims 512, turn 4-8 | **0.174** | 0.224 | 0.125 |
| rule vs rule 基準 | - | 0.500 | 0.502 | 0.498 |

- 30 epoch では固定ルールに大敗。学習中の self-play 指標
  (correct_action_rate ≈ 0.97-0.99, switch_prob は teacher 追従) は健全で、
  経路の退行ではなく**単純に学習量不足** (long_xteam の 1/10 のゲーム数) とみられる。
- 学習条件も基準と異なる (sims 512/turn 4-8 vs sims 64/turn 6-12) 点に注意。

## 結論 / 次の候補

- 新経路の学習自体は正常 (stage 3a は即収束、3b も指標健全)。30 epoch は
  rule 比較に届く水準ではない。
- 速度は 30 ep ≈ 4 分なので、300 epoch 相当 (約 40 分) を同条件で回して
  long_xteam と対等比較するのが次の確認として妥当。
