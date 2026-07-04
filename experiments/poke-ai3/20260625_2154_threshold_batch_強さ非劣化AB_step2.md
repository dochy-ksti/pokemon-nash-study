# threshold + batch充填 強さ非劣化 A/B (改善案step2)

実施: 2026-06-25 21:54 JST (run 11:53–12:54、約1時間)

## 目的

[step1 スループット計測](20260625_2050_threshold_batch_スループット計測_step1.md) で確定した
新設定 `g64/b512/th128` (examples/s 1.405x) が、baseline `g32/b256/th32` に対して
強さで劣化しないかを 0624_1752 手法 (全履歴比較 funnel + Bradley-Terry) で検証する。

## 手法

driver: `poke-ai3-python/scripts/run_ab_threshold.sh`。A→B→rate を直列実行。
両手法とも search A (depth-skew 2.0 / search 4-8 / sims 32 / sim-concurrency 16 /
stage 3b) で共通。差分は学習スループット設定のみ:

| 手法 | num-games | max-batch | threshold | epochs-per-step |
|---|---:|---:|---:|---:|
| A (baseline) | 32 | 256 | 32 | 20 |
| B (new) | 64 | 512 | 128 | 5 |

epochs-per-step を threshold に反比例 (20→5) させ snapshot 粒度を examples 等価に正規化。
warmup-steps=10 は step 単位 → A=200ep / B=50ep = 両者 6400 traj で examples 等価。
共通 shared_init.pt から funnel で各3 finalists 選抜 → 6 ckpt 総当たり → Bradley-Terry。
head-to-head は policy-only / 先後 512+512=1024 試合 / eval num-games=256・max-batch=256
(Phase3 で確定した 256/256 高速設定)。

`ckpt_tournament.py` は commit 9324696 で train 側 `--train-max-batch-size` /
`--train-trajectories-threshold` をプラム済み。

## 結果

最終生存: A = {A_ep360, A_ep540, A_ep680}、B = {B_ep55, B_ep135, B_ep215}。

Bradley-Terry レーティング (平均0):

| ckpt | 手法 | レート |
|---|---|---:|
| A_ep540 | A | +6.8 |
| B_ep135 | B | +6.8 |
| B_ep215 | B | -1.1 |
| B_ep55 | B | -1.2 |
| A_ep680 | A | -1.8 |
| A_ep360 | A | -9.5 |

手法平均: **B = +1.5 / A = -1.5 → 勝者 B**

全 head-to-head が 476〜548 / 1024 (ほぼ50%) で A/B 実質互角。差 ±1.5 Elo はノイズ圏内
(0624 と同様「手法間差 < ckpt 間ばらつき」)。

## 結論

- 合否ゲート「B が A に明確に負けていない」を満たす (むしろ僅かに上)。
- **新設定 `g64/b512/th128` を採用**。スループット 1.405x (examples/s) を強さ非劣化で獲得。
- policy lag 4倍 (1更新あたり examples が baseline の4倍) の懸念は実害なし。

## 次 (step3)

新設定 (g64/b512/th128) を固定し、学習バッチ `--minibatch-size` を上げた版 vs 上げない版を
同じ funnel 手法で A/B。学習フェーズの GPU 使用率を上げつつ強さ非劣化を確認する。

## 成果物・後片付け

- finalists と state: `data/poke-ai3/tournament/{A,B}_finalists.json`, `{A,B}_state.json`
- 旧 depth_skew A/B の B_ep{320,420,560}.pt が残存 (本実験と無関係、別実験成果物)。
