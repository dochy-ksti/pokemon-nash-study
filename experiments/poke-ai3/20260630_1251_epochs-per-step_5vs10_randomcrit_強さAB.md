# epochs-per-step 5 vs 10 (random+crit, sims64) 強さ A/B

実施: 2026-06-30 03:17–03:51 JST (funnel E10 → rate, random+crit)

## 目的・仮説

snapshot 間隔 `--epochs-per-step` を 5→10 にすると、出来上がる方策の強さが変わるか。
epochs-per-step は funnel が選抜に使う snapshot の粒度を決める (5→10 で snapshot が半分の密度)。
粒度を粗くした分、選抜候補が減るので強さが落ちるのか/変わらないのか/逆に伸びるのかを測る。

## 手法

[20260626_1615 RC64 実験](20260626_1615_学習教師sims_32vs64_random+crit環境_強さAB.md) と同条件
(sims64 / random+crit / stage3b / minibatch256 / shared_init 始点) で、**差分は
`--epochs-per-step` のみ**。

| アーム | epochs-per-step | warmup-steps | 状態 |
|---|---:|---:|---|
| RC64 (=eps5) | 5 | 10 | 既存流用 (RC64_ep110/170/295) |
| E10 | 10 | 5 | 新規 funnel (E10_ep110/260/320) |

- warmup は **epoch 基準で揃えた**: eps10 は step 間隔が倍なので warmup-steps 5 (=50ep) で
  eps5 の warmup-steps 10 (=50ep) と等価。これで epochs-per-step 純粋差分になる。
- `--train-block-epochs 50` は両方 50 (50/10=5 で倍数制約 OK)。
- 共通: depth-skew 2.0 / search 4-8 / sim-concurrency 16 / g64 b512 th128 / minibatch 256 /
  block 50 / peaks-per-rr 3 / finalists 3 / stage 3b / --random --crit / --sims 64。
- driver: `scripts/run_eps_ab.sh`。eps5 は RC64 流用なので新規実行は E10 funnel 1本 + rate 1回。
- rate: RC64 finalists 3 + E10 finalists 3 = 6 個を random+crit で、n-per-side 512。

### コマンド

driver `scripts/run_eps_ab.sh` (poke-ai3-python から) を実行。実体は以下。

```bash
cd poke-ai3-python
T=/home/dochy/pokemon_ai_proj/data/poke-ai3/tournament

make build

# eps10 アーム (新規 funnel)。RC64 と --epochs-per-step / --warmup-steps 以外完全一致。
uv run python scripts/ckpt_tournament.py funnel --tag E10 \
  --start "$T/shared_init.pt" --depth-skew 2.0 \
  --search-turn-min 4 --search-turn-max 8 --sim-concurrency 16 \
  --train-num-games 64 --train-max-batch-size 512 --train-trajectories-threshold 128 \
  --train-minibatch-size 256 \
  --train-block-epochs 50 --max-added-epochs 1000 \
  --peaks-per-rr 3 --finalists-target 3 \
  --n-per-side 512 --num-games 256 --stage 3b --random --crit --sims 64 \
  --epochs-per-step 10 --warmup-steps 5

# rate: E10 (eps10) vs RC64 (eps5, 流用) を random+crit で対戦。
uv run python scripts/ckpt_tournament.py rate --n-per-side 512 --num-games 256 --stage 3b \
  --random --crit \
  --funnel-json "$T/RC64_finalists.json" "$T/E10_finalists.json"
```

eps5 アーム (RC64) は [20260626_1615 RC64 実験](20260626_1615_学習教師sims_32vs64_random+crit環境_強さAB.md)
で生成済みの finalists を流用 (同一コマンドの `--epochs-per-step 5 --warmup-steps 10` 相当)。

## 結果 (random+crit)

| ckpt | 手法 | レート |
|---|---|---:|
| E10#2 (ep320) | E10(eps10) | +10.9 |
| E10#1 (ep260) | E10(eps10) | +4.3 |
| RC64#2 (ep295) | RC64(eps5) | +1.4 |
| RC64#0 (ep110) | RC64(eps5) | -2.6 |
| RC64#1 (ep170) | RC64(eps5) | -5.9 |
| E10#0 (ep110) | E10(eps10) | -8.0 |

**手法平均: E10(eps10) = +2.4 / RC64(eps5) = -2.4 → 差 4.8 Elo、E10 が上**

## 結論

### ⚠️ 重要な前提訂正: epochs-per-step は学習を変えない

事後にコードを確認した結果、`--epochs-per-step` は train-loop に
**`--snapshot-every` として渡るだけ** ([ckpt_tournament.py:200-218](../../poke-ai3-python/scripts/ckpt_tournament.py#L200-L218))。
学習ハイパラ (num-games / sims / minibatch / threshold / max-batch / lr 等) には一切入らない。
ブロック境界 `--train-block-epochs 50` も両アーム同一なので subprocess 再起動点も同じ。
warmup は epoch 基準で揃え済み。

→ **epochs-per-step が変えるのは「funnel が選抜に使える候補 snapshot の dump 粒度」だけ**で、
学習アルゴリズム・学習軌道そのものは不変。本 A/B は「学習の強弱」ではなく
「snapshot 粒度の選抜差 + ラン間 RNG ノイズ」を測っていたことになる。
RC64 と E10 は別プロセスの独立 funnel ランなので RNG が異なり、生成・学習データは
run-to-run で揺れる (学習は同分布)。

### 観測

- 手法平均は eps10 = +2.4 / eps5 = -2.4 (差 4.8 Elo) で E10 が上に出たが、
  **これは学習が強くなったのではなく、上記のノイズ + 選抜差の現れ**と解釈すべき。
- 原理的にはむしろ粒度が細かい eps5 のほうが選抜候補が多くピークを拾いやすい
  (弱くても同等が期待値)。今回 eps10 が平均で勝ったのは run 間ノイズで説明できる範囲。
- E10 は +10.9/+4.3/-8.0 と大ばらつき (最上位も最下位も E10)、RC64 は +1.4/-2.6/-5.9 と
  密にクラスタ。finalists 個体差 (どの ep を拾うか) が支配的で、手法優位のシグナルではない。

### 実用的含意

- epochs-per-step を 5→10 にしても **学習は何も変わらず、強さも実質変わらない (落ちない)**。
- 効果は純粋に funnel 側のコスト/粒度トレードオフ: eps10 は中間 snapshot と head-to-head
  評価の回数が減る一方、良い ep を取り逃すリスクがやや増える (E10#0 最下位がその一例の可能性)。
- 「強さを変えずに funnel の選抜コストを下げたい」目的でのみ eps10 を検討する価値がある。
  強さ向上を狙うパラメータではない。
- 解釈: epochs-per-step を 10 に粗くしても少なくとも**強さは落ちていない** (むしろ平均は上)。
  snapshot 粒度を半分にしても funnel の選抜が十分機能している。subprocess 起動回数や
  head-to-head 回数は減るので、強さを落とさず eps10 に上げる余地がある可能性。

### 留意

- 1 ラン・1 ルール (random+crit) のみ。差 4.8 Elo はノイズ帯の3倍だが E10 のばらつきが
  大きいため、別シード再現ラン / no-rng ルールでの確認が望ましい (RC64 実験は no-rng でも
  測っていた)。
- eps10 は wall-clock では funnel の固定費 (head-to-head 回数・snapshot 評価) が減る方向。
  ただし生成 epoch 数自体 (max-added-epochs 1000) は同じなので生成コストは不変。
- snapshot を粗くすると "良い ep を取り逃す" リスクがあり、E10#0 の最下位はその一例の可能性。
  finalists 個体差が大きい点はこの粒度トレードオフと整合的。

## 再現ラン (別シード独立, 2026-06-30 04:2x–05:0x JST)

前回は eps5=RC64 流用だったため、再現性確認として **両アームとも新規 funnel を別シードで**
回した (eps5=E5b / eps10=E10b)。epochs-per-step は学習を変えない (`--snapshot-every` のみ) ので、
これは実質「ラン間ノイズ + 選抜粒度差」の再測定。driver: `scripts/run_eps_ab_rep.sh`。

### 結果 (random+crit)

| ckpt | 手法 | レート |
|---|---|---:|
| E5b#2 (ep310) | eps5 | +12.2 |
| E10b#1 (ep240) | eps10 | +5.3 |
| E10b#2 (ep270) | eps10 | +0.8 |
| E10b#0 (ep130) | eps10 | -3.7 |
| E5b#1 (ep250) | eps5 | -4.4 |
| E5b#0 (ep140) | eps5 | -10.2 |

**手法平均: eps10 = +0.8 / eps5 = -0.8 → 差 1.6 Elo**

### 判定: 前回の差 4.8 Elo は再現せず

- 手法平均差は **4.8 → 1.6 Elo に縮小**し、過去ノイズ帯 (±1.3〜1.5) と同水準。
  「eps10 が上」の符号は同じだが大きさはノイズに埋もれた。
- しかも今回は **全体最上位 (E5b#2 +12.2) も最下位 (E5b#0 -10.2) も eps5 側**で、
  前回 (最上位・最下位とも eps10) と完全に逆。どちらの手法が大ばらつきになるかが
  ラン依存で入れ替わる = finalists 個体差 (どの ep を拾うか) と run 間 RNG が支配的。
- **結論: epochs-per-step 5 vs 10 に強さ差は無い**。前回の +4.8 Elo はノイズの上振れで、
  「epochs-per-step は学習を変えない (選抜 snapshot の dump 粒度のみ)」という原理と整合。
  強さは粒度に依らずほぼ不変なので、eps10 は **強さを落とさず funnel 選抜コストを下げる**
  目的でのみ採用可 (強さ目的のパラメータではない) という前回の含意が確定。

## 成果物

- E10 finalists: `data/poke-ai3/tournament/E10_finalists.json` (E10_ep110/260/320.pt)
- RC64 finalists: `data/poke-ai3/tournament/RC64_finalists.json` (流用)
- 再現ラン finalists: `data/poke-ai3/tournament/E5b_finalists.json` / `E10b_finalists.json`
- 実行ログ: `scratchpad/eps_ab.log` / `scratchpad/eps_ab_rep.log`
- driver: `poke-ai3-python/scripts/run_eps_ab.sh` / `run_eps_ab_rep.sh`
