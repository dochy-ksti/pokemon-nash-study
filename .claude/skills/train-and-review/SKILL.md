---
name: train-and-review
description: Train for N iterations then dump & review trajectory, repeated for multiple cycles
user-invocable: true
argument-hint: [--iterations N] [--cycles N] [--full-model]
allowed-tools: [Bash, Read, Glob, Grep]
---

# Train and Review

学習を N iteration 実行 → 1試合ダンプ → 簡易レビュー を cycles 回繰り返すスキル。
学習の進捗を定期的に確認しながら、異常を早期発見する。

## Arguments

オプション引数: $ARGUMENTS

- `--iterations N`: 1サイクルあたりの学習イテレーション数（デフォルト: 10）
- `--cycles N`: サイクルの繰り返し回数（デフォルト: 10）
- `--full-model`: フルサイズモデルを使用（デフォルトは小型モデル）

## Instructions

### Step 0: 引数パース

$ARGUMENTS から以下をパースする:
- `--iterations N` → ITER_PER_CYCLE (デフォルト 10)
- `--cycles N` → TOTAL_CYCLES (デフォルト 10)
- `--full-model` → FULL_MODEL フラグ

### Step 1: Showdownサーバーの確認

```bash
curl -s http://localhost:8000 > /dev/null 2>&1 && echo "OK" || echo "NOT RUNNING"
```

起動していない場合はユーザーに以下を伝えて中断:
> Showdownサーバーが起動していません。別ターミナルで以下を実行してください:
> `cd pokemon-showdown && node pokemon-showdown start --no-security`

### Step 2: 既存チェックポイントの確認

```bash
ls -t /home/dochy/pokemon_ai_proj/poke_ai2/checkpoints/*.pt 2>/dev/null | head -5
```

もし既存チェックポイントがあれば、ユーザーに「既存チェックポイントがあります。新規学習を開始すると上書きされる可能性があります。」と警告する。
ただしスキルの実行は中断しない（そのまま続行する）。

### Step 3: サイクルループ

以下を TOTAL_CYCLES 回繰り返す:

#### 3a. 学習実行

1サイクル目は新規学習、2サイクル目以降は前のチェックポイントから再開する。

FULL_MODEL_FLAG = `--full-model` (FULL_MODEL が true の場合) or 空文字

1サイクル目:
```bash
cd /home/dochy/pokemon_ai_proj/poke_ai2 && uv run python -m src.poke_ai2.orchestrator \
  $FULL_MODEL_FLAG \
  --iterations $ITER_PER_CYCLE \
  --checkpoint-every $ITER_PER_CYCLE \
  --eval-every 0 \
  2>&1
```

2サイクル目以降 (PREV_ITER = 前サイクルまでの累計イテレーション数):
```bash
cd /home/dochy/pokemon_ai_proj/poke_ai2 && uv run python -m src.poke_ai2.orchestrator \
  $FULL_MODEL_FLAG \
  --iterations $CURRENT_TOTAL_ITER \
  --checkpoint-every $ITER_PER_CYCLE \
  --eval-every 0 \
  --resume checkpoints/iter_$(printf '%04d' $PREV_ITER).pt \
  2>&1
```

ここで CURRENT_TOTAL_ITER = ITER_PER_CYCLE * 現在のサイクル番号。
例: iterations=10, cycle 2 なら --iterations 20 --resume checkpoints/iter_0010.pt

タイムアウトは600秒。
**重要**: 実行中は10秒に1回程度、進行状況を確認すること。出力が止まっていたら原因を調査する。

学習が失敗した場合はエラーを報告して中断する。

#### 3b. ダンプ実行

学習完了後のチェックポイントを使って1試合ダンプ。
CKPT_PATH = `checkpoints/iter_$(printf '%04d' $CURRENT_TOTAL_ITER).pt`
もしそのファイルがなければ `checkpoints/final.pt` を使う。

```bash
cd /home/dochy/pokemon_ai_proj/poke_ai2 && PYTHONPATH=. uv run python scripts/dump_trajectory.py \
  -o /tmp/train_review_cycle_N.txt \
  $FULL_MODEL_FLAG \
  --checkpoint $CKPT_PATH \
  2>&1
```

N はサイクル番号（1始まり）。タイムアウトは300秒。

#### 3c. 簡易レビュー

ダンプファイルから value と probs の行だけを抽出して分析する:

```bash
grep -E "^-- T|probs:" /tmp/train_review_cycle_N.txt
```

以下を確認して **簡潔に** レポートする（各サイクル5行以内）:

1. **value の範囲と変動**: 両プレイヤーの value の min/max/変動幅。前サイクルとの比較。
2. **確率分布の偏り**: 最も確率が高い手と低い手の差。均一分布からどれだけ離れたか。
3. **勝敗と value の相関**: 勝った側の value が高いか。

異常を検知した場合（学習が発散、NaN、value が突然 0 に固定、など）は即座にユーザーに報告する。

### Step 4: 最終レポート

全サイクル完了後、以下の形式でまとめレポートを出力する:

```
## Train & Review 結果サマリー

**設定**: iterations/cycle=$ITER_PER_CYCLE, cycles=$TOTAL_CYCLES, model=$MODEL_TYPE

| Cycle | Total Iter | Winner | P1 value (min→max) | P2 value (min→max) | Prob spread |
|-------|-----------|--------|---------------------|---------------------|-------------|
| 1     | 10        | P1     | 0.07→0.13           | 0.08→0.14           | 5.1%        |
| ...   | ...       | ...    | ...                 | ...                 | ...         |

### 学習の傾向
- [value の変化傾向]
- [確率分布の変化傾向]
- [異常の有無]

### 総合評価
[2-3行のまとめ]
```

Prob spread = 各ターンの「最大確率 - 最小確率」の平均値。
均一分布からどれだけ離れたかの指標。大きいほど学習が進んでいる。
