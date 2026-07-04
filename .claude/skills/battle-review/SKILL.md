---
name: battle-review
description: Run a self-play battle, dump the trajectory, and review the dump for anomalies
user-invocable: true
argument-hint: [--full-model]
allowed-tools: [Bash, Read, Glob, Grep]
---

# Battle Review

1試合self-playを実行し、trajectoryダンプを出力して、AIがその内容を分析し異常がないか調べるスキル。

## Arguments

オプション引数: $ARGUMENTS

- `--full-model`: フルサイズモデルを使用（デフォルトは小型モデル）

## Instructions

### Step 1: Showdownサーバーの確認

Showdownサーバーが起動しているか確認する。

```bash
curl -s http://localhost:8000 > /dev/null 2>&1 && echo "OK" || echo "NOT RUNNING"
```

サーバーが起動していない場合は、ユーザーに以下を伝えて中断する:
> Showdownサーバーが起動していません。別ターミナルで以下を実行してください:
> `cd pokemon-showdown && node pokemon-showdown start --no-security`

### Step 2: バトル実行とダンプ出力

引数に `--full-model` が含まれていれば `--full-model` フラグを付ける。出力ファイルは固定で `/tmp/battle_review_dump.txt` を使う。

```bash
cd /home/dochy/pokemon_ai_proj/poke_ai2 && PYTHONPATH=. uv run python scripts/dump_trajectory.py -o /tmp/battle_review_dump.txt [--full-model]
```

タイムアウトは300秒。失敗した場合はエラーメッセージをユーザーに報告して中断する。

### Step 3: ダンプファイルの読み込み

Read toolで `/tmp/battle_review_dump.txt` を読み込む。

### Step 4: 分析とレビュー

ダンプの内容を読み、以下の観点で異常がないか調べる。各項目について問題があれば具体的に指摘し、なければ「問題なし」と簡潔に報告する。

#### チェック項目

1. **行動選択の妥当性**: 明らかに不合理な行動を選んでいないか
   - 有利な技があるのに不利な技を高確率で選んでいる
   - 瀕死寸前の味方を交代せず続行している（交代先がいるのに）
   - テラスタルの使い方が不合理

2. **確率分布の異常**: probsの分布がおかしくないか
   - 全ての合法手にほぼ均等な確率（学習が進んでいない兆候）
   - 1つの手に99%以上集中しすぎ（過学習の兆候）
   - 不正な確率（合計が100%から大きくずれている等）

3. **value推定の一貫性**: 価値推定が試合展開と矛盾していないか
   - 明らかに有利な局面でvalueが低い、またはその逆
   - 勝った側のvalueが最初から最後まで低いまま（学習不足）
   - valueが急激に変動しすぎている

4. **状態表現の正しさ**: tokenize/embedのバグを示唆する異常がないか
   - `?`で始まるtoken（未知のtoken ID）が多い
   - HPが0%のポケモンがactiveにいる
   - 同じポケモンがチームに重複している
   - ally_teamに存在しないポケモンがactiveに出ている
   - PADだらけで情報がほとんどない

5. **forceSwitch処理**: force_switchの扱いが正しいか
   - force_switchターンで技選択をしていないか
   - force_switchでない通常ターンで交代のみが合法になっていないか

6. **action_maskの整合性**: 合法手マスクが正しいか
   - PP切れの技が合法になっている
   - 場にいるポケモンへのswitchが合法になっている

### Step 5: 結果報告

日本語で、以下の形式でレポートする:

```
## バトルレビュー結果

**試合結果**: Player1 [勝敗] / Player2 [勝敗] (Xターン)

### 発見された問題
- [問題があればここに列挙]

### 注目すべき点
- [異常ではないが気になる点]

### 総合評価
[1-2行の総合コメント]
```
