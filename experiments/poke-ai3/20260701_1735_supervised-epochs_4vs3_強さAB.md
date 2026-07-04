# 教師あり学習パス回数 supervised-epochs 4 vs 3 — 強さ A/B

実施: 2026-07-01 08:04–08:35 JST (E3 funnel 新規 + rate, random+crit)

## 目的・仮説

`SupervisedConfig.epochs`(= `--train-supervised-epochs`)は **1 バッチの生成データを
教師あり学習で何パスなめるか**(毎パスでシャッフルし直す)。既定 4。これは
`epochs-per-step`(= `--snapshot-every`、選抜の刻みだけ変え学習不変)とは別物で、
**学習の軌跡そのものを変える真のノブ**。4→3 に減らすとパス 25% 減で学習が速くなるが、
強さが落ちないかを測る。生成は不変(learn step だけ変わる)。

| 条件 | supervised-epochs | 状態 |
|---|---:|---|
| baseline | 4 | RC64 流用 (RC64_ep110/170/295) |
| E3 | **3** | 新規 funnel (E3_ep65/225/250) |

- E3 は RC64 と **epochs 以外完全一致**: start=shared_init / depth-skew2.0 / search-min4 /
  search-max8 / sims64 / g64 b512 th128 / minibatch256 / eps5 / block50 / warmup10 /
  finalists3 / max-added-epochs1000 / stage3b / random+crit。
- baseline は新規実行せず [RC64](20260626_1615_学習教師sims_32vs64_random+crit環境_強さAB.md) 流用。
- 時間プローブは省略(learn 側ノブのため生成不変、wall 差は小さく埋もれる)。
- 段階的方針 (c): まず 1 ラン。差が閾値 (|Δ|≳8〜10 Elo) 超なら再現ラン。
- driver: `scripts/run_epochs_ab.sh`。E3 funnel 実 wall = 1820s。

## 結果

### 強さ (random+crit, rate 6個1プール, n-per-side 512)

| ckpt | 手法 | レート |
|---|---|---:|
| RC64#2 (ep295) | epochs4 | +7.7 |
| E3#0 (ep65) | epochs3 | +1.5 |
| RC64#1 (ep170) | epochs4 | +1.4 |
| RC64#0 (ep110) | epochs4 | +0.0 |
| E3#1 (ep225) | epochs3 | -1.2 |
| E3#2 (ep250) | epochs3 | -9.4 |

**手法平均: epochs4 (RC64) = +3.1 / epochs3 (E3) = -3.1**(差 6.2 Elo)

## 結論

- **supervised-epochs 3 は採用しない(既定 4 維持)**。手法平均差 6.2 Elo は
  単発ランのノイズ帯(±5 Elo 規模、finalists 個体差込み)をわずかに超える程度で、
  再現ラン発動閾値(|Δ|≳8〜10)には届かない = **信頼できる強さ差なし**。
- E3 の劣位は個体 **E3#2 (-9.4) の外れ値**が主因。残る E3#0/#1 は +1.5 / -1.2 で
  RC64 個体群(+0.0 / +1.4 / +7.7)と同水準に混在。「epochs3 がコヒーレントに弱い」
  パターンではない(前回 search9 の教訓どおり、単発の手法平均差・個体順位は鵜呑みにしない)。
- パス 25% 削減の学習時短メリットはあるが、**強さを伸ばさず・むしろ僅かに下振れ傾向**
  なので、コスト削減目的で 3 に落とす積極的理由も薄い。既定 4 を据え置く。

### 留意

- 1 ラン・random+crit 単一、baseline は RC64 流用(別シード既存ラン)。差がノイズ帯
  近傍なので「有意差なし・4 維持」の安全側結論には影響しない。
- 差 6.2 Elo は閾値未満のため再現ランは省略(Q5 の判断基準どおり)。もし epochs3 の
  時短を本採用検討する場合のみ、両アーム新規で再現ランを追加すべき。

## 成果物

- E3 finalists: `data/poke-ai3/tournament/E3_finalists.json` (E3_ep65/225/250.pt)
- baseline: `data/poke-ai3/tournament/RC64_finalists.json`(流用)
- 実行ログ: `scratchpad/epochs_ab.log`
- driver: `poke-ai3-python/scripts/run_epochs_ab.sh`
