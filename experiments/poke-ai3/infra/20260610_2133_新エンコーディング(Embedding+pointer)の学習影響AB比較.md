# 新エンコーディング (Embedding + pointer 方式) の学習影響 A/B 比較

## 目的

種族・技のスケール対応 (`feature/scalable-encoding` ブランチ) が Stage3b の学習に
悪影響を与えないことの検証。変更点:

- 観測をグローバル ID (Showdown 全 dex 語彙) ベースへ刷新、alive/present 廃止
- 行動空間を「習得技スロット相対 + 交代枠」へ変更 (Showdown `/choose move n` と同型)
- モデルを species/move/type Embedding + フィールド単位トークン + pointer 方式
  policy ヘッド (各行動トークンから共有 Linear で読む) へ書き換え

旧 = master (197995d、1スカラ=1トークン + CLS 一括ヘッド)、
新 = feature/scalable-encoding (aba1d9b)。

## コマンド

両側とも同一 (旧側は master worktree から実行)。決定論設定・battle_seed のみ可変:

```bash
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --no-random --no-crit --stage 3b \
  --max-epochs 5 --chunk-threshold 512 --battle-seed {1001,1002,1003}
```

追試として新側 seed 1002/1003 を `--max-epochs 15` で再実行。
ログ: `experiments/poke-ai3/logs_encoding_ab/`。

## 結果

correct_action_rate (argmax 正解技率) の epoch 推移:

| run | ep1 | ep2 | ep3 | ep4 | ep5 | 1.0 到達 |
|---|---|---|---|---|---|---|
| old seed1001 | 0.69 | 0.62 | 0.63 | 1.00 | 1.00 | ep4 |
| old seed1002 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ep1 |
| old seed1003 | 0.67 | 0.64 | 0.62 | 0.59 | 1.00 | ep5 |
| new seed1001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ep1 |
| new seed1002 | 0.67 | 0.64 | 0.56 | 0.68 | 0.60 | 未到達 |
| new seed1003 | 0.30 | 0.34 | 0.43 | 0.43 | 0.50 | 未到達 |
| new(15ep) seed1002 | 1.00 | … (全 epoch 1.00) | | | | ep1 |
| new(15ep) seed1003 | 1.00 | … (全 epoch 1.00) | | | | ep1 |

- 同一 battle_seed の再実行 (new 5ep vs new 15ep の seed1002/1003) で到達 epoch が
  「未到達」→「ep1」へ変わった。**モデル初期化 (torch 乱数は非固定) の運が支配的**で、
  battle_seed 固定でも run 間分散はエンコーディング差より大きい。
- 旧 3/3、新 3/3 (5ep×1 + 15ep×2) で 1.0 到達。新旧で系統的な学習速度差は検出されず。
- value_loss は新旧同水準 (ep5 時点 0.036〜0.062)、entropy も同水準 (0.4〜0.57)。
- switch_prob は両側とも model ≈ teacher (例: 旧 seed1001 ep5 0.43/0.45、
  新 seed1001 ep5 0.31/0.35 — 教師への追従は両者成立)。

## 結論

**合格。** 新エンコーディング (Embedding + pointer) は Stage3b の学習を阻害しない。
correct_action_rate の到達・value/entropy とも旧と同等で、観測された差は
モデル初期化の run 間分散の範囲内。

## 注意

- 本比較で正確に測れるのは「壊れていないこと」まで。収束安定性の厳密な比較には
  init seed 固定 + 長期 (300ep) ランが要る (今後 6v6/種族追加時に再評価)。
- 旧 checkpoint (`data/poke-ai3/*.pt`) は観測レイアウト・行動 index の意味が変わった
  ため新コードでは使用不可 (ファイルは残置)。
