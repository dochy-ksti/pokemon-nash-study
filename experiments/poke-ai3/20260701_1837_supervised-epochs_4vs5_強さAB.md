# 教師あり学習パス回数 supervised-epochs 4 vs 5 — 強さ A/B

実施: 2026-07-01 09:04–09:37 JST (E5 funnel 新規 + rate, random+crit)

同日の [4 vs 3](20260701_1735_supervised-epochs_4vs3_強さAB.md) の続き。今度は増やす方向
(4→5)を測り、3/4/5 の3点で既定 4 の妥当性を確認する。

## 目的・仮説

`--train-supervised-epochs` = 1 バッチの生成データを教師あり学習で何パスなめるか(毎パス
シャッフルし直す)。学習軌跡そのものを変える真のノブ。4→5 でパス 25% 増。フィットが
深まり強くなるか、それとも過適合/飽和かを測る。生成は不変(learn step だけ変わる)。

| 条件 | supervised-epochs | 状態 |
|---|---:|---|
| baseline | 4 | RC64 流用 (RC64_ep110/170/295) |
| E5 | **5** | 新規 funnel (E5_ep110/170/240) |

- E5 は RC64 と epochs 以外完全一致(shared_init / depth-skew2.0 / search-min4 /
  search-max8 / sims64 / g64 b512 th128 / minibatch256 / eps5 / block50 / warmup10 /
  finalists3 / max-added-epochs1000 / stage3b / random+crit)。
- baseline は [RC64](20260626_1615_学習教師sims_32vs64_random+crit環境_強さAB.md) 流用。
- 段階的方針 (c): 1 ラン。差が閾値 (|Δ|≳8〜10) 超なら再現ラン。
- driver: `scripts/run_epochs5_ab.sh`。E5 funnel 実 wall = 1951s。

## 結果

### 強さ (random+crit, rate 6個1プール, n-per-side 512)

| ckpt | 手法 | レート |
|---|---|---:|
| RC64#1 (ep170) | epochs4 | +3.6 |
| RC64#0 (ep110) | epochs4 | +2.6 |
| RC64#2 (ep295) | epochs4 | -0.2 |
| E5#2 (ep240) | epochs5 | -0.2 |
| E5#1 (ep170) | epochs5 | -0.5 |
| E5#0 (ep110) | epochs5 | -5.3 |

**手法平均: epochs4 (RC64) = +2.0 / epochs5 (E5) = -2.0**(差 4.0 Elo)

## 結論

- **supervised-epochs 5 は採用しない(既定 4 維持)**。手法平均差 4.0 Elo は
  単発ノイズ帯(±5 Elo 規模)**内**で、信頼できる強さ差なし。E5 個体
  (-5.3/-0.5/-0.2)は RC64 群(+2.6/+3.6/-0.2)とほぼ重なり、コヒーレントな優劣なし。
- パス 25% 増は学習時間を増やすだけで強さを伸ばさない(むしろ僅かに下振れ)= **飽和**。

### 3点まとめ(3/4/5、いずれも baseline=RC64=epochs4 流用と比較)

| supervised-epochs | 手法平均 (vs RC64) | 差 |
|---:|---:|---:|
| 3 | -3.1 | 6.2 |
| **4** | **基準** | — |
| 5 | -2.0 | 4.0 |

- **4 が中庸で最良**。3(パス減)も5(パス増)もノイズ帯前後で強さを伸ばさず、
  どちらも僅かに下振れ。**既定 supervised-epochs=4 を据え置くのが妥当**と確認。
- 減らす(時短)側も増やす(精度)側も現状の強さには効かない = このノブは
  現行設定 4 付近で飽和/平坦。

### 留意

- 各 1 ラン・random+crit 単一、baseline は RC64 流用。差がノイズ帯内〜近傍のため
  再現ランは省略(Q5 の判断基準どおり)。「4 維持」の安全側結論は頑健。

## 成果物

- E5 finalists: `data/poke-ai3/tournament/E5_finalists.json` (E5_ep110/170/240.pt)
- baseline: `data/poke-ai3/tournament/RC64_finalists.json`(流用)
- 実行ログ: `scratchpad/epochs5_ab.log`
- driver: `poke-ai3-python/scripts/run_epochs5_ab.sh`
