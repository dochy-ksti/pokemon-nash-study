# minibatch-size スループット + 強さ非劣化 A/B (改善案 step3)

実施: 2026-06-25 23:21 JST (step3b run 13:05–14:21、約1.3時間)

## 目的

[step2](20260625_2154_threshold_batch_強さ非劣化AB_step2.md) で採用した
`g64/b512/th128` を固定し、学習の `--minibatch-size` を上げて学習フェーズの GPU 使用率を
上げる。examples/s 改善 (step3a) と強さ非劣化 (step3b) を確認する。

## step3a: スループット計測

`scratchpad/measure_minibatch.py`。採用設定 g64/b512/th128 固定、search A / sims 32 /
stage 3b / 固定seed / shared_init コピーから max-epochs 14、warmup 4更新を捨てた定常窓で
examples/s 算出。各3run中央値。

| minibatch | examples/s 中央値 | mbs64比 |
|---|---:|---:|
| 64 (既定) | 1025.4 | 1.000x |
| 256 | 1734.6 | **1.692x** |
| 512 | 2039.8 | 1.989x |

頭打ちせず 512 まで上昇。ただし th128 は1更新≈1500 examples で、epoch4 固定だと勾配 step 数が
mbs64≈92 → mbs256≈24 → mbs512≈12 と激減し、mbs512 は強さ劣化リスク最大。256→512 の上積みは
+18% のみ。リスク/リターンで **mbs256 を step3b 候補に採用** (512 は見送り)。

## step3b: 強さ非劣化 A/B

driver: `scripts/run_ab_minibatch.sh`。両手法とも g64/b512/th128 + search A + epochs-per-step 5
で共通、差分は `--train-minibatch-size` のみ (A=64 / B=256)。funnel で各3 finalists →
Bradley-Terry。head-to-head は eval num-games 256 / 先後 512+512=1024。
`ckpt_tournament.py` は本セッションで `--train-minibatch-size` をプラム追加。

最終生存: A = {A_ep55, A_ep135, A_ep215}、B = {B_ep105, B_ep165, B_ep330}。

| ckpt | 手法 | レート |
|---|---|---:|
| A_ep215 | A | +8.3 |
| B_ep105 | B | +5.5 |
| A_ep135 | A | +0.9 |
| B_ep330 | B | -2.6 |
| A_ep55 | A | -5.3 |
| B_ep165 | B | -6.8 |

手法平均: **A = +1.3 / B = -1.3**。全クロス対戦 (A手法 vs B手法) は 470〜554/1024 で全て
50% 近傍・勝敗混在。手法間差 < ckpt 間ばらつき (step2 と同じノイズ帯 ±1.3〜1.5)。

## 結論

- B (mbs256) は -1.3 とわずかに負け側だが step2 で互角と判定したのと同じノイズ圏内であり
  「明確に負け」ではない。合否ゲートを満たす → **mbs256 採用**。
- 学習フェーズ GPU 使用率向上で **examples/s 1.69x を追加獲得** (強さ非劣化)。

## 累積成果 (step1→3)

baseline `g32/b256/th32/mbs64` から最終採用 `g64/b512/th128/mbs256` へ:

- examples/s: step1で 1.405x (g64/b512/th128) → step3で ×1.69 = **概算 2.4x** (生成全体)。
  ※step1とstep3aは測定条件 (epoch数) が違うため厳密な積ではなく方向値。
- いずれも強さ非劣化 (手法平均レート差はすべてノイズ圏内 ±1.5 以内)。

## 推奨既定設定

`--num-games 64 --max-batch-size 512 --trajectories-threshold 128 --minibatch-size 256`
(sim-concurrency 16, sims 32, stage3b 系の学習)。

## 補足 / 留保

- step3b の B は僅かに負け側 (-1.3)。より厳密にするなら多シード funnel か n-per-side 拡大で
  信頼区間付き判定をする価値はある (step2 同様の留保)。
- mbs512 (1.99x) は未検証。GPU をさらに埋めたい場合は mbs256 採用後に単独 A/B で確認可能。
