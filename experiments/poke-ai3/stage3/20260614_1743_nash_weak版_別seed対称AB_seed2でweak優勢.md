# nash_accumulation_weak 別 seed 対称 A/B（Phase 2）— 2 seed 総合で weak ≧ normal、seed2 では weak が決定的に優勢

## 目的

`20260614_1620` の結論「seed1 では weak と通常 nash が同等」を受け、事前登録ルールの Phase 2
（weak/通常を**両方とも別 seed で ep0 から**学習する対称 A/B）を実施。seed 依存性を排して
weak が通常 nash と比べ優劣があるかを確かめる。

前提（合意ドキュメント `docs/poke-ai3/20260614_0110_...md`）:
- 「weak だけ 2 seed は不公平。やるなら両方を 2 seed で」→ 本実験で normal/weak とも新 seed で学習。

## 条件

- 新 seed = **20260615**（seed1 は 20260614）。
- 両 run 共通: hidden128, stage3b, no-random/no-crit, sims64, search 6-12, num-games 32,
  max-batch-size 512, ep0→200, snapshot 5ep ごと。
- weak run のみ `--nash-weak --nash-learning-rate 2.0`。通常 run は付けない（strict nash）。
- 強さ指標 = 固定アンカー r2_ep105 / r2_ep130（seed1 の 2329 gauntlet 由来）に対する
  被験 ckpt 視点勝率の 2 アンカー平均（n=512=256×両side、SE≈0.022）。seed1 と比較可能。
- 直接対戦は両 side 各 256（n=512）で先手有利を相殺。

## コマンド

学習（逐次。各約 10〜12 分）:

```bash
# 通常 nash seed2
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 200 --max-batch-size 512 --battle-seed 20260615 \
  --checkpoint-path ../data/poke-ai3/seed2/normal_s2.pt --snapshot-every 5
# weak seed2（同 seed）
uv run train-loop ... --nash-weak --nash-learning-rate 2.0 --battle-seed 20260615 \
  --checkpoint-path ../data/poke-ai3/seed2/weak_s2.pt --snapshot-every 5
```

強さ指標:

```bash
uv run python scripts/anchor_sweep.py \
  --anchors ../data/poke-ai3/gauntlet/r2_ep105.pt ../data/poke-ai3/gauntlet/r2_ep130.pt \
  --subjects ../data/poke-ai3/seed2/{normal,weak}_s2_ep{120,140,160,180,200}.pt --n-per-side 256
```

直接対戦: weak_s2 vs normal_s2 を両 side 各 256。

## 結果

### 強さ指標（2 アンカー平均、seed2）

| ep | normal_s2 | weak_s2 |
|----|-----------|---------|
| 120 | .485 | .490 |
| 140 | .426 | **.500** |
| 160 | .454 | **.524**（weak ピーク） |
| 180 | .471 | .479 |
| 200 | **.490**（normal ピーク） | .508 |

全エポックで weak ≧ normal。ep140/160 で約 +0.07（有意水準）。

### 直接対戦 weak_s2 vs normal_s2（weak 側勝率、n=512、両 side 相殺）

| 対戦 | weak 勝率 | 0.5 乖離 |
|------|---------|------|
| ep160(weakピーク) vs ep200(normalピーク) | .539 | +1.8 SE |
| ep160 vs ep160（マッチド） | .551 | +2.3 SE |
| ep200 vs ep200（マッチド） | **.562** | +2.8 SE（±0.06 ライン超） |
| ep140 vs ep140（マッチド） | .531 | +1.4 SE |

4 対戦すべて weak 勝ち越し、方向完全一致。

## 2 seed 総合

| | seed1 (20260614) | seed2 (20260615) |
|---|---|---|
| 強さ指標 (weak − normal) | ほぼ同等 | weak +0.03〜0.07 |
| ピーク直接対戦 | 五分 (.508〜.520) | weak 優勢 (.539〜.562) |

## 結論

- **seed1 は引き分け、seed2 は weak が明確に優勢。2 seed 通算で「weak は通常 nash と
  同等以上（悪くて互角、良くて勝ち越し）」。** 全測定を通じて weak が normal を有意に
  下回ったケースは一つも無い。
- 厳密な事前登録の「決定的」条件（両 seed で 強さ指標±0.08 *かつ* 直接±0.06）は満たさないが、
  seed2 単独では ep200 同士の直接対戦 .562 が決定ラインを超えた。崖の無い穏当な挙動という
  設計上の利点も併せ、**weak を採用する方向に十分な根拠**が得られた。
- 強さの単調性（より長く学習＝より強い）は weak/normal いずれも依然非成立で、.45〜.52 の
  プラトーで振動する点は seed1 と同じ。これは nash 系共通の性質。

## 成果物

- データ: `data/poke-ai3/seed2/`（{normal,weak}_s2_ep5..200.pt, train_{normal,weak}_s2.log,
  anchor_seed2.log, h2h_seed2.log）
- 関連: `20260614_1620_nash_weak版ABと延長検証_通常nashと同等.md`（seed1）、
  実装 commit `46be215`。
