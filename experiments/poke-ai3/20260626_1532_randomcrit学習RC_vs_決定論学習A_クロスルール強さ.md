# random+crit 学習 (RC) vs no-random/no-crit 学習 (A) クロスルール強さ比較

実施: 2026-06-26 15:32 JST (run 06:00頃–06:32 funnel RC → rate×2)

## 目的

「random あり crit ありで学習した AI」と「random なし crit なしで学習した AI」が、
それぞれのルール下でどちらがどれだけ強いかを直接比較する。

- A 側: 既存の no-random/no-crit 学習 finalists ([20260626_1258 step3c の A 系] の v2)。
- RC 側: 今回 random+crit で新規学習した finalists。
- 評価ルールを 2 通り (random+crit / no-random no-crit) に振り、各ルールでの強さを見る。

## 手法

両者 stage 3b・同一 shared_init.pt から開始し、A と RC の差分は学習・選抜時の
**乱数 (16段ロール=random / 急所=crit) の有無のみ**。

- 共通 COMMON: depth-skew 2.0 / search 4-8 / sims 32 / sim-concurrency 16 /
  train g64 b512 th128 / minibatch 256 / epochs-per-step 5 / train-block-epochs 50 /
  warmup 10 / finalists 3 / eval n-per-side 512・num-games 256 / stage 3b。
- A: `--no-random --no-crit` で funnel 学習・選抜 (既存流用、A_ep105/175/290)。
- RC: 上記に `--random --crit` のみ追加して funnel 学習・選抜 (RC_ep85/185/240)。
- driver: `scripts/run_rc_vs_a.sh`。
- ツール改修: `ckpt_tournament.py` の `funnel`/`rate` に `--random/--crit` を追加し
  (既定 no-rng で既存挙動不変)、`train_block_to` (学習) と `head_to_head`
  (funnel 選抜・rate 評価) の両方へ素通し。
- rate は 6 個 (A finalists 3 + RC finalists 3) を 2 ルールで実施。

## 結果

### ルール (1) random+crit

| ckpt | 手法 | レート |
|---|---|---:|
| A#1 (A_ep175) | A | +5.9 |
| RC#1 (RC_ep185) | RC | +5.9 |
| A#0 (A_ep105) | A | +3.6 |
| RC#2 (RC_ep240) | RC | -0.1 |
| RC#0 (RC_ep85) | RC | -7.1 |
| A#2 (A_ep290) | A | -8.1 |

**手法平均: A = +0.5 / RC = -0.5 → 差 1.0 Elo (互角)**

### ルール (2) no-random/no-crit

| ckpt | 手法 | レート |
|---|---|---:|
| A#1 (A_ep175) | A | +11.4 |
| RC#2 (RC_ep240) | RC | +2.8 |
| A#0 (A_ep105) | A | +2.5 |
| A#2 (A_ep290) | A | -0.3 |
| RC#1 (RC_ep185) | RC | -3.1 |
| RC#0 (RC_ep85) | RC | -13.3 |

**手法平均: A = +4.5 / RC = -4.5 → 差 9.0 Elo (A 明確に上)**

## 結論

- **random+crit ルールでは A と RC はほぼ互角** (差 1.0 Elo)。これは過去 A/B の
  ノイズ帯 (±1.3〜1.5) と同程度で、有意差なし。注目すべきは、RC が自分の
  「ホームルール」(訓練と同じ random+crit) ですら A を上回れないこと。
  no-crit/no-random で訓練した A も random+crit 環境に十分対応できている。
- **no-random/no-crit ルールでは A が RC より明確に強い** (差 9.0 Elo、ノイズの
  約 3 倍)。決定論環境では A が一貫優勢 (上位 3 個中 A が #1/#3/#4 を占有、RC 最良は
  RC#2 の +2.8 止まり、RC#0 は最下位 -13.3)。
- 総括: **どちらのルールでも RC は A に対して優位を取れず、決定論ルールでは A が大きく
  勝つ。乱数込みの訓練 (RC) は、乱数環境での強さを A 以上に引き上げる効果が見られず、
  決定論での強さはむしろ犠牲にしている**。少なくとも本設定 (stage 3b・search 4-8・
  sims 32・同 epoch 規模) では、決定論学習 (A) の方が頑健に強い。

### 補足 (解釈の注意)

- 乱数は勝率を 50% 方向に薄めるため、random+crit ルールでは実力差が Elo に出にくい。
  ルール(1)の「互角」は「実力が等しい」と即断はできないが、決定論ルール(2)で差が
  はっきり A 優位に開く以上、「RC が乱数環境で A を上回る」シナリオは支持されない。
- RC の funnel 選抜は random+crit で行ったため、決定論ルールでの RC finalists 選定は
  最適でない可能性はある (それでも手法平均で A 優位)。

## 成果物

- RC finalists: `data/poke-ai3/tournament/RC_finalists.json` (RC_ep85/185/240.pt)
- A finalists: `data/poke-ai3/tournament/A_finalists.json` (流用、A_ep105/175/290.pt)
- 退避: step3c mbs512 の旧 B_* → `data/poke-ai3/tournament/step3c_B_mbs512/`
- 実行ログ: `scratchpad/rc_vs_a.log`
- driver: `poke-ai3-python/scripts/run_rc_vs_a.sh`
- ツール改修: `poke-ai3-python/scripts/ckpt_tournament.py` (funnel/rate に --random/--crit)
