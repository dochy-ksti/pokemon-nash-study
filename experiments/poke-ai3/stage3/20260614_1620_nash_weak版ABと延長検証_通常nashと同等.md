# nash_accumulation_weak の A/B と ep260 までの延長検証 — 通常 nash と同等、優劣つかず

## 目的

nash_accumulation を穏当化した `nash_accumulation_weak`（崖を撤去し、最低ライン
`nash_avg/2` 以下の選択肢も training_pi=0 に落とさず係数 `1/nash_learning_rate` を与える版。
詳細設計は `docs/poke-ai3/20260614_0110_nash_accumulation_weak設計とABプロトコル合意.md`）を、
通常 nash（`20260613_2329` の run r2）と A/B し、

1. weak は ep0 から強くなるか／強さの上がり方は安定か
2. weak のピーク checkpoint は通常 nash のピーク（r2_ep140）を超えるか
3. ep150 で頭打ちか、より先にピークがあるか（本セッションで追加検証）

を確かめる。

## 共通条件

- hidden128, stage3b, no-random/no-crit, sims64, search 6-12, num-games 32。
- weak run: `--nash-weak --nash-learning-rate 2.0 --battle-seed 20260614`、ep0→150 を学習後、
  本セッションで ep150→260 へ延長（同一 battle_seed で checkpoint 再開）。
- 強さ指標 = 固定アンカー r2_ep105 / r2_ep130 に対する被験 ckpt 視点勝率を 2 アンカー平均
  （`scripts/anchor_sweep.py`、各 n=512 = 256×両side、SE≈0.022）。
- 直接対戦は `eval_ckpt_vs_ckpt` を両 side（各 256、計 n=512）で先手有利を相殺。

## コマンド

延長学習（ep150→260、約 5 分）:

```bash
cd poke-ai3-python
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 260 --max-batch-size 512 \
  --nash-weak --nash-learning-rate 2.0 --battle-seed 20260614 \
  --checkpoint-path ../data/poke-ai3/weak/weak_s1.pt --snapshot-every 5
```

強さ指標（ep150→260、10ep ごと）:

```bash
uv run python scripts/anchor_sweep.py \
  --anchors ../data/poke-ai3/gauntlet/r2_ep105.pt ../data/poke-ai3/gauntlet/r2_ep130.pt \
  --subjects ../data/poke-ai3/weak/weak_s1_ep{150,160,...,260}.pt --n-per-side 256
```

ピーク直接対戦:

```bash
# weak_s1_ep220 / ep240 を r2_ep140 と両 side 各 256 で対戦
```

## 結果

### 強さ指標 ep150→260（2 アンカー平均）

| ep | vs ep105 | vs ep130 | 指標 |
|----|---------|---------|------|
| 150 | .547 | .492 | **.520** |
| 160 | .480 | .496 | .488 |
| 170 | .484 | .477 | .481 |
| 180 | .504 | .457 | .481 |
| 190 | .484 | .402 | .443（谷） |
| 200 | .559 | .434 | .497 |
| 210 | .555 | .465 | .510 |
| 220 | .562 | .488 | **.525** |
| 230 | .543 | .484 | .514 |
| 240 | .574 | .473 | **.524** |
| 250 | .566 | .426 | .496 |
| 260 | .539 | .484 | **.512** |

参考: ep0→150 区間の weak ピークは ep140=.512 / ep150=.520。通常 nash(2329) ピークは .559。

### ピーク直接対戦（n=512、両 side 相殺）

| weak | vs r2_ep140 | 0.5 からの乖離 |
|------|------------|------|
| ep220 | .520 (266-246) | +0.9 SE |
| ep240 | .508 (260-252) | +0.4 SE |

### ep140/ep150 vs 通常ピーク群（ep0→150 区間、参考）

| weak | vs r2_ep140 | vs r2_ep130 | vs r2_ep120 |
|------|------|------|------|
| ep140 | .488 | .488 | .488（3 者一致＝測定アーティファクト疑い） |
| ep150 | .574 | .504 | .551 |

## 結論

- **weak は通常 nash と同等。明確な優劣は出なかった。** 強さ指標は ep150 以降も
  .44〜.525 を振動し、通常の .559 ピークには延長しても届かない。weak ピーク
  （ep220/ep240）を通常ピーク r2_ep140 に直接当てても .520/.508 と**五分**（事前登録の
  決定ライン勝率 ±0.06 に未到達、強さ指標も ±0.08 マージン内）。
- **「より長く学習＝より強い」の単調性は weak でも成立しない。** 単峰ピークは無く、
  .50〜.52 プラトーで横ばい＋振動。これは通常 nash（2329, ep120 前後頭打ち）と同じ性質。
- アンカー別に傾向が割れる（vs ep105 は後半 .55〜.57 へ上昇、vs ep130 は .40〜.49 で
  伸びない）。**非推移性（じゃんけん的）**の兆候があり、単一指標でのピーク特定は不安定。

## 事前登録ルールに対する扱い

ルール上は「微妙→Phase 2（weak/通常を両方 +1 seed）」に該当するが、ピーク直接対戦が
五分である以上、追加 seed で「決定的な差」が出る見込みは薄いと判断し、Phase 2 は保留。
weak を採用しても実害も実益も無い（崖が無いぶん挙動は穏当）。採否は今回データでは
根拠を持って決められない。

## 成果物

- 実装: commit `46be215`「nash: 穏当化版 nash_accumulation_weak を追加 (崖なし) + nash/lr の CLI 化」
- データ: `data/poke-ai3/weak/`（weak_s1_ep5..260.pt, train_weak_s1.log,
  train_weak_s1_ext.log, anchor_weak_s1.log, anchor_weak_s1_ext.log,
  h2h_weak_s1.log, h2h_weak_peak.log）
