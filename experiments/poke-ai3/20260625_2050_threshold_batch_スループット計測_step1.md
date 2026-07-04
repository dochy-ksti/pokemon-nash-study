# threshold + batch充填 スループット計測 (改善案step1)

実施: 2026-06-25 20:50 JST

## 目的

[20260625_1728 改善案](20260625_1728_sims32_GPU使用率乱高下_原因調査と改善案.md) の方針
(grill-me で確定) に基づき、生成スループット (examples/s) を上げる学習設定を確定する。
本ファイルは step1 = スループット確認。強さ非劣化の funnel A/B は step2 で別途実施。

確定済み方針:

- 主目的(B): examples/s 最大化。強さは step2 の funnel A/B (0624_1752手法) で別途担保。
- 効きの源泉は直交する2軸: threshold = learn/save 割り込み頻度 (固定費)、
  games64+b512 = 本物の大バッチ充填による GPU 効率・round 半減。
- num-games 単独軸は不採用 (固定費頻度が真の効きで高コスト)。提案優先度2/3/4・補助は不採用。

## 手法

`scratchpad/measure_throughput.py`: train-loop を subprocess 起動し、
`lookahead_update examples=N` 行ごとに monotonic 時刻を記録。warmup 4更新を捨てた
定常窓で `Σexamples / Δwall` = examples/s を算出。各構成3run、中央値で比較。

共通条件: `--sims 32 --sim-concurrency 16 --depth-skew 2.0 --search-turn-min 4
--search-turn-max 8 --no-random --no-crit --stage 3b --battle-seed 12345`、
shared_init.pt のコピーから開始。総trajectory数を揃えるため epochs は threshold に反比例
(各≈1280 traj)。

| tag | num-games | max-batch | threshold | max-epochs |
|---|---:|---:|---:|---:|
| baseline | 32 | 256 | 32 | 40 |
| new-th64 | 64 | 512 | 64 | 20 |
| new-th128 | 64 | 512 | 128 | 10 |

事前 `make build` で .so 鮮度を保証。

## 結果

| 構成 | examples/s 中央値 | baseline比 | 各run |
|---|---:|---:|---|
| baseline g32/b256/th32 | 769.5 | 1.000x | 731.6 / 769.5 / 773.4 |
| new g64/b512/th64 | 943.9 | 1.227x | 943.9 / 1007.0 / 938.5 |
| new g64/b512/th128 | 1081.2 | **1.405x** | 1062.9 / 1165.6 / 1081.2 |

## 判定

- th128 vs th64 = 1081.2/943.9 = **+14.5%**。事前ゲート「th128 が th64 比 +10%以上なら
  th128 採用」を満たす。
- **step2 の funnel A/B では B手法 = g64/b512/th128 (epochs-per-step=5) を採用**。
- b512 (num-games64で in-flight 上限 2048 内、実バッチが本当に太る) の効果が、提案の
  短時間プローブ th64=1.33x を上回り 1.405x を実現。padding 由来でない本物のスループット改善。

## 次 (step2)

`ckpt_tournament.py funnel` (commit 9324696 で train側 batch/threshold をプラム済み) で:

- 手法A (baseline): `--train-num-games 32 --train-max-batch-size 256
  --train-trajectories-threshold 32 --epochs-per-step 20`
- 手法B (new): `--train-num-games 64 --train-max-batch-size 512
  --train-trajectories-threshold 128 --epochs-per-step 5`
- 共通: depth-skew 2.0 / search 4-8 / sims 32 / stage 3b / shared_init から、
  finalists 3 ずつ → rate (Bradley-Terry)。
- 合否: 手法平均レートで B が A に明確に負けていなければ強さ非劣化 = 採用。

epochs-per-step を threshold に反比例 (20→5) させ、snapshot 粒度を examples 等価に正規化。
