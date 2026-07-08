# value 教師 max vs expected A/B

## 目的

盤面の value 教師を、現行の「手ごと最大勝率」から「均衡混合の期待勝率」に変えて
強さが落ちないかを比較する。

- 現行 (max): `value = maxᵢ win_rates[i]`
- 新式 (expected): `value = Σᵢ training_pi[i] × win_rates[i]`

`training_pi` は floor なしの Nash 均衡混合 (legal で正規化済み・illegal は 0) なので、
そのまま重み付き和で正しい期待値になる。

### 動機

ゼロサム同時手番ゲームの均衡では、盤面価値は「自分の最善手を確定で取った時の勝率」
ではなく「均衡混合の下での期待勝率」。max は相手も混合することを無視して自分だけ最善
応答する前提なので、**構造的に均衡値以上へ出る = 自分の勝率を過大評価する**。これは
web 対戦で以前指摘された「このAIは自分から見た勝率を過剰に高く見積もる」現象の一因
そのものと考えられる。expected 版はそれを均衡値へ較正する狙い。

判定は **強さのみ** (funnel→rate の Bradley-Terry)。「均衡値へ直しても強さは落ちない」
が確認できれば、過大評価を潰した上で採用できる。

## 実装

value 教師だけを触る変更。rollout・win_rates・nash_accumulation・training_pi/
selection_pi は不変。単一合法手の短絡 (value=net 予測) も training_pi が one-hot なので
両式一致で影響なし。

- `LookaheadConfig.value_target_expected: bool` (既定 false=max) を追加
  ([poke-env-rust/src/lookahead.rs])。既定が現行式なので既存の実験・checkpoint の
  再現性は保持。
- 期待値は nash_accumulation で training_pi を得た後に計算し直す (rollouts は従来通り
  max も返し、フラグで差し替え)。
- pyo3 署名 (`poke-ai3-python/src/lib.rs`) → TrainSession/run_train_loop
  (`train_loop.py`, CLI `--value-target max|expected`) → funnel
  (`ckpt_tournament.py`, `--value-target`) まで nash_learning_rate と同じ経路で配線。
  finalists.json / funnel 起動ログに `value_target` を記録。
- 対応比較 (下記) 用に funnel へ `--train-battle-seed` を追加 (既定 None=毎 run ランダム
  独立)。両アームを同一 battle_seed で回せるようにした。

### 単体テスト

- `expected_value_target_is_at_most_max` (lookahead.rs): 同一局面・同一 seed で
  max/expected を切替え、expected ≤ max を確認。win_rates は seed 共有で不変 =
  value 教師だけが変わることも同時に検証。
- 既存 lookahead テスト 5 件も回帰なし。

## 実験計画

nash_lr 3-seed ドライバの構成を流用。両アーム同条件 (nash_lr=1.5, stage 3b, 同フラグ)
で違いは value 式のみ。

- **対応比較**: 各 seed で両アームが (a) shared_init.pt を共有し (b) 同一
  `--train-battle-seed` を使う。差は value 教師の式だけになり、1 seed でも式の効果を
  分離しやすい。
- まず各1 seed (計2 funnel: `VMAX_s1` / `VEXP_s1`)。差が微妙で時間があれば SEEDS に
  追加してペア (VMAX_sN, VEXP_sN) を増やす。
- funnel→rate の Bradley-Terry で強さ比較。

### コマンド

```bash
cd poke-ai3-python
setsid nohup bash scripts/run_value_target_ab.sh > /tmp/vtab/driver.log 2>&1 &
```

ドライバ (`scripts/run_value_target_ab.sh`) は冪等 (既存 _finalists.json はスキップ)。
seed→battle_seed 対応は BSEED 連想配列、比較する seed は SEEDS 配列で管理。

## 結果 (2026-07-08, 各1 seed)

VMAX_s1 (max) と VEXP_s1 (expected) を同一 shared_init.pt・同一 train-battle-seed
20260708 で回し (対応比較)、6 finalists を eval_seed ランダム (base=4539352510365600688)
で総当たり (各ペア n_per_side=512 → 1024 試合)。

### Bradley-Terry レーティング (平均 0)

| finalist | 手法 | レート | checkpoint |
|---|---|---|---|
| VEXP_s1#2 | expected | +12.8 | ep400 |
| VMAX_s1#1 | max | +2.1 | ep370 |
| VMAX_s1#0 | max | +2.1 | ep195 |
| VEXP_s1#0 | expected | -4.1 | ep160 |
| VMAX_s1#2 | max | -6.4 | ep545 |
| VEXP_s1#1 | expected | -6.5 | ep335 |

### 手法ごと平均レート

- **VEXP (expected): +0.7** (n=3, [-4.1, -6.5, +12.8])
- **VMAX (max): -0.7** (n=3, [+2.1, +2.1, -6.4])

### 解釈

- 手法差は **1.4 Elo**。前回把握した run 間 SD ~6.7 Elo・セル単位 SE ~1.56点 に対し
  完全にノイズ帯。finalist 個体は両手法とも -6.5〜+12.8 に散らばり、**1 seed では
  優劣判定不能** (ドライバの "優れた手法: VEXP" は平均の僅差を機械的に拾っただけ)。
- クロス対戦 (VMAX vs VEXP の 9 ペア) はいずれも 460〜564 勝 = ±50% 近辺で拮抗。
  value 式は着手分布 (training_pi/selection_pi) を変えないため、強さがほぼ動かないのは
  設計通りの挙動。
- **目的 (判定は強さのみ) の観点では理想的**: expected へ均衡値較正しても強さは
  落ちていない (むしろ僅かに上)。「勝率過大評価を潰しても強さを損なわない」仮説は
  少なくとも棄却されず。ただし白黒つけるには seed 2・3 を追加する必要がある
  (ドライバ SEEDS=(1)→(1 2 3)、battle_seed は BSEED に用意済み)。

### ログ

- driver: /tmp/vtab/driver.log (VMAX 78分・VEXP 62分・rate、全 exit=0)
- finalists: data/poke-ai3/tournament/{VMAX,VEXP}_s1_finalists.json
  (`value_target` フィールドで max/expected を記録)
