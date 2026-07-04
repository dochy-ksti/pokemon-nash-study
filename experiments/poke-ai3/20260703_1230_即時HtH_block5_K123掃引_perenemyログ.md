# 即時 head-to-head (block=5) で K=1/2/3 掃引 + per-enemy ログ

実行完了: 2026-07-03 (JST) / r=0.5 固定・enemy-window 1/2/3・block=epochs_per_step=5

## 目的

1. funnel を **即時 head-to-head** (--train-block-epochs 省略 = block=5) で回す新 baseline を作る。
2. **per-enemy ログ** (enemy_by[...]) を実装し、K>=2 で「1個前/2個前/3個前」の学習中勝率を取得。
3. r=0.5 固定で K=1/2/3 の手法平均レートを block=5 環境で再確認。

tmux セッション ksweep5 で K1b5→K2b5→K3b5 逐次 → rate。切断耐性 (Windows/SSH 切断後も継続) 確認。
ドライバ: [run_k_sweep_block5.sh](../../poke-ai3-python/scripts/run_k_sweep_block5.sh)。

## 結果1: 手法平均レート (K1b5+K2b5+K3b5 の全9 finalist 1プール総当たり, random+crit, n-per-side 512)

| 手法 | enemy-window | 手法平均 | 個体レート |
|---|---:|---:|---|
| **K1b5** | 1 | **+6.7** | +7.3 / +10.8 / +2.1 |
| K2b5 | 2 | +0.5 | −1.1 / +6.1 / −3.4 |
| K3b5 | 3 | −7.3 | −6.9 / −13.1 / −1.7 |

**K=1 が最良**。即時 head-to-head でも block=50 の K 掃引 (experiments 20260701_2246) と同結論
(K=1 最良、K を増やすほど劣化)。K=3 の劣化 (−7.3) が顕著。

## 結果2: per-enemy 学習側勝率 (ブロック内相対距離ごと・全ブロック加重平均)

| | 1個前 | 2個前 | 3個前 |
|---|---:|---:|---:|
| K2b5 | 0.553 (n=6058) | 0.530 (n=4808) | — |
| K3b5 | 0.558 (n=4777) | 0.555 (n=3570) | 0.551 (n=2595) |

- 学習側は window 内のどの敵 (1〜3個前) にも **ほぼ一様に ~0.53〜0.56 で勝ち越し**。距離による
  明確な勾配なし = 学習者は直近プール全体に薄く先行し続ける健全な状態。
- 注意: これは全学習期間の平均 (学習途中 vs その時点の window)。事後eval (experiments 20260701_2246
  末尾: 最終モデル vs 直近 peak = K1 で 0.503) とは別測定。

## 実装 (本セッション, commit 3d75903)

- print_stats に enemy_by[ラベル=勝率(n) ...] を追加 (_format_enemy_by)。
- configure_enemies が game_id -> 敵 checkpoint stem マップ (sess.enemy_labels) を設定。
- 敵ラベルは enemy_pool[-K:] (古→新) 順。K3 なら最古=3個前 / 最新=1個前。

## メモ

- block=5 即時化で 3ラン+rate が短時間で完走 (finalist 到達が早い: K1b5 ep510 / K2b5 ep400 / K3b5 ep415)。
- 本 block=5 baseline (K1b5/K2b5/K3b5 finalists) は今後の即時HtH実験の比較基準に使える。
