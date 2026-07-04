# depth_skew A/B を現手法(全履歴比較 funnel)で再実験

実施: 2026-06-24 JST

## 目的

[20260622_1524 の depth_skew A/B](20260622_1524_depth_skew_AB_自動選抜トーナメント.md) を、
刷新した checkpoint 選抜手法で再実行し、結論(B がわずかに優れる)が再現するかを確認する。

- A: `--search-turn-min 4 --search-turn-max 8 --depth-skew 2.0`
- B: `--search-turn-min 6 --search-turn-max 12 --depth-skew 1.0`(従来設定)

両者を共通のランダム初期 `shared_init.pt` から学習し、各々 funnel で最終生存3個を選抜
→ 6 ckpt で総当たり + Bradley-Terry → 手法平均レート比較。

## 手法の変更点(前回 → 今回)

`ckpt_tournament.py` を刷新(commit `ce61794`)。grill-me セッションでの設計合意に基づく:

- **0回戦を全履歴比較化**: 前回の「最新 vs 20/40/80 epoch 前」固定をやめ、リセット以降の
  全履歴と対戦。最新を負かした古 snapshot 群を総当たりして最高勝率を1回戦突破に。
  (20/40/80 は恣意的で、一巡が80を超えると永久に終わらない懸念があったため)
- **段構造の統一**: 「1回戦突破を3つ集めて総当たり → 2回戦突破」を3つ集めて終了。
  前回の隣接 PeakDetector 二次段を廃止。最終生存数 2 → **3**。
- **warmup**: 開始から10ステップ(~ep200)は単調増加期とみなし head-to-head 省略。
- **レジューム**: ステップ境界で `<tag>_state.json` を保存、`--resume` で継続可能。
- 判定閾値は 0.5 のまま(マージン無し)。

## コマンド

driver: `poke-ai3-python/scripts/run_ab_depth_skew.sh`(commit `bc0f7d6`)が A→B→rate を直列実行。

```bash
uv run python scripts/ckpt_tournament.py funnel --tag {A,B} \
  --start data/poke-ai3/tournament/shared_init.pt \
  --peaks-per-rr 3 --finalists-target 3 --warmup-steps 10 \
  --epochs-per-step 20 --max-added-epochs 4000 \
  --n-per-side 512 --num-games 64 --stage 3b \
  {--depth-skew 2.0 --search-turn-min 4 --search-turn-max 8 |
   --depth-skew 1.0 --search-turn-min 6 --search-turn-max 12}

uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 64 --stage 3b \
  --funnel-json data/poke-ai3/tournament/A_finalists.json data/poke-ai3/tournament/B_finalists.json
```

head-to-head はすべて policy-only(確率着手)/ 先後 512+512=1024 試合。

## 結果

最終生存(各手法とも 1回戦突破を 9 個放出 → 3 個に絞り込み):

- A = {A_ep220, A_ep560, A_ep780}
- B = {B_ep320, B_ep420, B_ep560}

Bradley-Terry レーティング(平均0):

| ckpt | 手法 | レート |
|---|---|---|
| A_ep220 | A | +6.8 |
| B_ep560 | B | +4.1 |
| B_ep320 | B | +2.5 |
| A_ep560 | A | -4.1 |
| A_ep780 | A | -4.3 |
| B_ep420 | B | -5.0 |

手法平均: **B = +0.5 / A = -0.5 → 判定「B が優れた手法」**

## 結論

- 勝者は **B(search 6-12 / depth_skew 1.0、従来設定)**。前回と**同じ方向**で、depth_skew=2.0
  への変更が改善にならないという結論が再現した。
- ただし差は前回 ~6 Elo → 今回 **~1 Elo** に縮小し、完全にノイズ圏内。最強(A_ep220 +6.8)と
  最弱(B_ep420 -5.0)が手法をまたいで混在しており、**手法間差 < checkpoint 間ばらつき**という
  前回の知見も再現。実用的には「両者ほぼ互角、depth_skew=2.0 採用の根拠なし」。

## 副産物・知見

- 全履歴比較 funnel は **A/B とも予算内(max 4000ep)で finalists 3 個を完走**。前回は隣接方式で
  B のピークが立たず履歴比較へ切替えた経緯があったが、今回は最初から安定して回った。
- 両手法とも 1回戦突破を 9 個出して 3 個に絞った(`peaks_emitted=9`)。warmup~ep200 を挟んでも
  選抜が成立。
- 「手法の優劣を 3 対 3 の点推定の符号だけで断じるのは危険」が改めて裏付けられた。次段として
  プール拡張版(多シード funnel / beaten-newer 同梱)や信頼区間付き判定で有意性を確認する価値あり。

## 成果物

- 最終生存と state: `data/poke-ai3/tournament/{A,B}_finalists.json`, `{A,B}_state.json`
- 実行ログ: `data/poke-ai3/tournament/ab_run.log`
- 前回成果物は `data/poke-ai3/tournament/archive_20260622/` に退避済み。
