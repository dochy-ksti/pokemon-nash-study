# head-to-head GPU効率と集計精度の改善案

作成: 2026-06-25 17:33 JST
改訂: 2026-06-25 18:05 JST (実コード確認とgrillを経て方針確定)

## 1. 目的

checkpoint同士のhead-to-headは、funnelとBradley-Terry rateで大量に実行される。
現在はGPU使用率が低く、同時に集計上の標本数を実際より大きく数える問題もある。

本書では次を両立する改善案を整理する。

1. 勝率比較の意味と公平性を維持する
2. 実試合数を正しく数える
3. GPUへ連続的に仕事を供給する
4. 1推論batch当たりの無駄な計算を減らす
5. funnel/rateの総wall時間を短縮する

対象:

- `poke-ai3-python/python/poke_ai3_train/eval_ckpt_vs_ckpt.py`
- `poke-ai3/src/root_task.rs` (`Trajectory` 構造体に `player` を追加)
- `poke-ai3/src/game_task.rs` (trajectory生成時に `player` を設定)
- `poke-ai3-python/scripts/ckpt_tournament.py`
- 必要に応じてRust executorのbatch flush制御

## 2. 現状 (実コードで確認済み)

funnel/rateは概ね次の形でhead-to-headを呼ぶ。

```bash
uv run python -m poke_ai3_train.eval_ckpt_vs_ckpt \
  --checkpoint-a A.pt \
  --checkpoint-b B.pt \
  --num-games 64 \
  --num-eval-games 1024 \
  --policy-only \
  --no-random --no-crit --stage 3b
```

主な既定値:

| 項目 | 値 |
|---|---:|
| num-games | funnel/rateから64 |
| sim-concurrency | 1 |
| policy-only | true |
| max-batch-size | 未指定 |
| 実際の既定max-batch-size | `64 * 1 / 2 = 32` |
| idle時sleep | 50ms (`--sleep-seconds` 既定 0.05) |

policy-onlyではlookahead rolloutを行わず、各プレイヤーの着手ごとにpolicy/value推論を
1回だけ行う。1ゲーム当たり最大でP1/P2の2要求が同時に存在し得る
(policy-only・sim-concurrency=1での同時in-flight上限は `num-games * 2`)。
それに対しbatch上限は32である。

### 2.1 関連コードの実測 (改訂時の確認)

- `poke-ai3/src/game_task.rs:236-253`: 1バトル終了時、`p1_traj` と `p2_traj` の
  **2つ**を送出する。両者は同じ `game_id` と**同じ `winner`** を持つ。
  `p2_traj` は `!eval_rule_opponent` のときだけ送出される。
- `eval_ckpt_vs_ckpt.py` の `main()` は `eval_rule_opponent=False` (両側NNのself-play経路)
  なので、**1実バトルにつき必ず2 trajectoryが届く**。
- `poke-ai3/src/root_task.rs:47` の `Trajectory` は **トップレベルに `game_id` と
  `winner` を持つが `player` を持たない**。player は各 `items` の中だけにある。
- `game_id` は並列スロット番号で、iterごとに再利用される。よって `game_id` 単独では
  バトルを一意に識別できない。

これらの確認により、初版が提案していた `items[0].player` フィルタ
(items空で破綻) や `battle_id=(game_id, iter)` 重複排除 (Trajectoryにiterが無い)
は、いずれも採らない。代わりに `Trajectory` へ `player` フィールドを足す
(後述 優先度0)。

## 3. 問題点

### 3.1 集計が1試合を2回数えている

`collect_results()` は trajectoryを1本読むたびに `winner` を集計し `games += 1` する。
上記2.1のとおり1実バトルがP1/P2の2 trajectoryを生み、両方に同じ勝者が入るため、
1実試合を2試合として集計する。

影響:

- 勝率の点推定は変わらない (両側を同じ倍率で重複計上するため)
- `num-eval-games=1024`で実際に行われるバトルは概ね512試合
- 出力の`n=1024`は実試合数を2倍に表示
- 標準誤差と信頼区間を実際より狭く見積もる
- 計画書の「先後1024+1024=2048試合/ペア」も、実バトル数としては概ね半分

これは性能改善より先に直すべき正しさの問題である。

### 3.2 A/B両モデルを全行へ推論している

現在の`infer_step_split()`は、batch全体B行に対して:

1. agent AをB行すべてにforward
2. agent BをB行すべてにforward
3. player maskで必要な側だけ採用

する。

P1/P2が概ね半数ずつなら、必要な有効推論量は合計B行だが、実際には合計2B行を計算する。
GPU演算の約半分が捨てられている。

### 3.3 batchが小さい

`num-games=64`, `sim-concurrency=1`の既定max-batch-sizeは32。
P1/P2分割後の有効行は各モデル約16行相当である。小型ModernBERTをRTX 5090で
動かすには細すぎる。

head-to-headは**学習ステップを持たない**ため、学習ループのように「更新頻度・固定費を
抑えるためbatchを小さく保つ」理由が一切ない。GPUを埋める唯一のレバーは同時in-flight
推論数であり、これは `num-games * 2` で決まる。したがって本書ではbatchの微小スイープでは
なく、**num-games自体を引き上げてbatchを太らせる**方針を採る (優先度3)。

### 3.4 50ms sleepが長い

観測batchもtrajectoryも準備できていない場合、head-to-headループは50ms sleepする。
推論要求が完成した直後でも最大50ms拾われず、全ゲームが推論待ちになる可能性がある。

学習ループでは同種の問題を避けるためsleep既定値を0にしているが、head-to-headは
古い50ms既定のままである。

### 3.5 進捗printが多い

trajectory batchを受信するたびに`games so far`をprintする (`collect_results` 内)。
性能の律速ではないが、rateのように多数プロセスを逐次起動する用途ではログがノイズになる。

## 4. 改善案

### 優先度0: 実試合数を正しく数える (必須)

**Rust側**: `Trajectory` (`root_task.rs:47`) に `player: Player` を1フィールド追加し、
`Serialize` 対象にする。`game_task.rs` で `p1_traj` には `Player::P1`、`p2_traj` には
`Player::P2` を設定する。

**Python側**: `collect_results()` はP1 trajectoryだけを実試合として数える。

```python
for trajectory in payload.get("vec", []):
    if str(trajectory.get("player")) != "P1":
        continue
    winner = str(trajectory.get("winner"))
    idx = 0 if winner == "P1" else (1 if winner == "P2" else 2)
    tally[idx] += 1
    games += 1
```

この方式を採る理由 (初版の2案を棄却):

- `items[0].player` 参照は items空 trajectory で `IndexError` になる。
- `(game_id, iter)` 重複排除は `Trajectory` に iter が無く、`game_id` は並列スロットで
  iterごとに再利用されるため成立しない。
- `tally // 2` は eval_rule_opponent の有無 (片側のみ送出される経路) に依存し脆い。
- トップレベル `player` フィルタは items空でも壊れず、rule対戦時 (P1のみ送出) も
  そのP1を素通しで正しく1試合と数える。モード非依存。

修正後の互換性:

- `--num-eval-games`を「実バトル数」と定義し直す
- 既存と同じwall時間にしたい場合は一時的に値を半分にする
- 既存と同じ統計精度にしたい場合は値を維持し、実行時間が約2倍になることを受け入れる
- 過去結果と比較するときは、過去の表示nを実試合数として半分に読み替える

### 優先度1: idle sleep既定を0へ変更

`eval_ckpt_vs_ckpt.py` の `--sleep-seconds` 既定を `0.05` → `0` にする
(`else: time.sleep(0)`)。

期待:

- batch完成後の最大50ms待ちを除去
- ロックステップの谷を短縮
- 勝率の意味を変更しない

注意:

- PythonメインスレッドのCPU使用率は上がる (完全スピン)。
- 1台で複数評価を並列実行する用途には向かないが、現在は実験を1本ずつ実行する運用なので
  問題にならない。将来並列化するときは `--sleep-seconds 0.05` 等で戻せる。

最小変更であり最初に入れる。

### 優先度2: player別に必要行だけ推論する (確定タスク)

batch全体をA/B両方へ通すのをやめ、P1行→agent A、P2行→agent B だけ推論する。

```text
obs B行
  ├─ P1 index → agent Aへ約B/2行
  └─ P2 index → agent Bへ約B/2行
結果を元の行順へscatter
```

期待:

- GPUで処理する有効行数を合計2B→Bへ削減
- モデルが大きくなっても効果が維持される
- 勝率比較の意味を変更しない

実装上の注意:

- packed numpy配列をplayer indexで切り出すヘルパーを用意する
- `game_id/player/request_id`は元batchの順番でexecutorへ返す
- policy/valueだけ元のB行へscatterする
- empty batchは従来どおりechoする
- P1/P2の片側が0行のbatchも扱う

CUDA Graphの2冪bucketでは、分割により各モデルのreplay行数が概ね半減する。
ただしper-row効率を保つには分割後も各モデルの行数を確保したい。よって優先度3の
num-games拡大と併せて設計する (分割後 各モデル≒num-games相当行を狙う)。

### 優先度3: num-gamesを拡大しbatchを太らせる

head-to-headは学習しないので並列度を上げてbatchを埋めるのが本筋 (3.3参照)。

#### batch上限の見方 (2/3 ≡ 1/3 の整理)

AGENTS.md の表記:

- 同時in-flight推論の理論上限 = `num-games * sim-concurrency * 2`
- 推奨 max-batch-size = `num-games * sim-concurrency` の **2/3 以下**

この2つは同じ値を別の分母で言っているだけである。
すなわち **「productの2/3」= 「理論上限 (×2した値) の1/3」**。混同しないこと。

#### 既定とスイープ

policy-only・sim-concurrency=1 では in-flight上限 = `num-games * 2`。

既定として **num-games=256 / max-batch-size=256** を据える。

- in-flight上限 = 512、max-batch 256 はその範囲内 (= 理論上限の1/2、productの1.0倍だが
  in-flightに十分な余裕があり pending揺らぎでも埋まる)。
- CUDA Graph bucket 256 にちょうど乗る。
- player分割後でも各モデル約128行を確保できる。

スイープは **`{128, 256, 384}`** を1回実施し、勝者を既定化する。
num-gamesを上げるほどメモリ (1ゲーム = simulator + 2 runner + router) と
tokioタスクが増えるため、384より上は実測でメモリ余裕を確認してから検討する。

KPI:

- 実バトル/s
- 推論rows/s
- 実batch平均 / batch間隔
- GPU util p10/median/p90 (参考値)
- 1ペア1024実試合のwall

注意:

RootTaskは閾値に達するまでpartial flushしない。policy-onlyではゲーム局面や強制交代により
同時pending数が揺れるため、閾値を高くしすぎるとデッドロックし得る。各runへ無進捗watchdogを
付け、長期的にはtimeout付きpartial flush導入後に上限を広げる。

### 優先度4: 進捗printをnum-games試合ごとに間引く

`collect_results()` のprintを、trajectory batch毎ではなく **num-games実試合ごと** (= 1波
ごとに1行) に変更する。コストは実質ゼロでログが1波1行になる。rate用に途中printを完全に
止める `--quiet-progress` も足してよい。

## 5. 不採用・将来案

### 推論優先のイベント処理 (不採用)

「`is_ready()` を `trajectories_ready()` より先に処理する」案は、**head-to-headでは
ほぼ無意味**なため採らない。学習ループではtrajectory到達時に
「JSON展開 + 4 epoch SGD + 診断 + torch.save」を同期実行するため順序が効く
(これは別文書 `..._sims32_GPU使用率乱高下...` の主題)。一方 head-to-headの
trajectory処理は「recv + winner集計 + print」だけでマイクロ秒級なので、
順序を入れ替えても供給の谷はほぼ生まれない。backlogヒステリシス等の複雑なコードを
足す価値がない。実測でGPU供給に明確な谷が観測された場合のみ再検討する。

### 2モデルを1回のforwardへ統合 (将来案)

player分割後もA/Bで2回graph replayが必要。さらに進めるなら functional call で
A/Bを batched parameter 化し1 replay内で両modelを実行する案があるが、実装複雑度が高く、
player分割だけで計算量が半減するため今回スコープ外。

## 6. 推奨実装順

### Phase 1: 正しさと設定

1. `Trajectory` に `player` 追加 (Rust) → P1のみ計上 (Python) で実試合数を修正
2. `--sleep-seconds` 既定 0
3. 進捗printをnum-games試合ごとに間引く (+ `--quiet-progress`)

コード変更が小さく、結果の意味を変えない (3は表示のみ)。

### Phase 2: 無駄なGPU計算を除去

4. player別packed batch分割と元行順へのscatter
5. eager/graph両経路でA/B (旧実装との勝率一致を確認)

### Phase 3: 並列度の確定

6. num-games拡大 (既定 256/256) と `{128,256,384}` スイープで既定を決定
7. 必要ならtimeout付きpartial flush

## 7. 検証計画

### 7.1 正しさ

固定checkpoint A/B、固定battle seedで:

- 旧実装と新実装の実バトル勝敗列を比較
- P1/P2入替えの両方向を確認
- `num-eval-games=N`でP1 trajectoryが厳密にN件になることを確認
- 出力`a_win+b_win+draw == N`

batch分割で浮動小数点の微差が出る可能性があるためビット一致は必須にしない。
大標本の勝率が統計的に一致し、player routingの単体テストが通ることを採用条件とする。

### 7.2 性能

代表checkpoint 1ペア、実バトル4096、先後入替えで交互5組。

| 構成 | sleep | split | num-games | max-batch |
|---|---:|---:|---:|---:|
| baseline | 50ms | なし | 64 | 32 |
| A | 0 | なし | 64 | 32 |
| B | 0 | あり | 64 | 32 |
| C | 0 | あり | 128 | 128 |
| D | 0 | あり | 256 | 256 |
| E | 0 | あり | 384 | 384 |

採用KPI:

1. 実バトル/s
2. 1ペアのwall
3. GPU使用率ではなくGPU計算量当たりの実バトル数
4. deadlock/無進捗がないこと

### 7.3 採用基準

- 集計修正: 必須
- sleep=0: wallが悪化しなければ採用
- player別推論: 10%以上高速化で採用
- num-games/batch拡大: スイープ勝者を採用 (無進捗リスクが無いこと)

## 8. 期待値

最も確実なのはplayer別推論で、不要なforward行を約半分にできる。
さらにnum-games拡大でbatchが太り、小batch由来のGPU過小利用を緩和できる。

現実的な期待:

- sleep除去: 数%〜大幅改善 (現在の50ms谷の頻度次第)
- player別推論: 15〜40%改善
- num-games/batch拡大: 0〜30%改善 (現状32→256でGPU充填が改善する余地)

組み合わせでhead-to-head wallを30〜50%削減できる余地がある。最終的には実測で判断する。

## 9. 過去のrate結果を読む際の注意

現在までの`eval_ckpt_vs_ckpt`結果は勝率の点推定には利用できるが、表示された試合数は
実バトル数として概ね2倍である。

例:

```text
先後1024+1024=2048試合/ペア
```

と記録した実験は、実際には概ね:

```text
先後512+512=1024実バトル/ペア
```

として信頼区間を読む必要がある。Bradley-Terryの点推定順位は、全勝敗を同じ倍率で
重複計上している限り変わらないが、不確実性は過小評価されている。
</content>
