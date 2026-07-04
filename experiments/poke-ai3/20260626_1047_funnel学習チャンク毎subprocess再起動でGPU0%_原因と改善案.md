# funnel 学習のチャンク毎 subprocess 再起動で GPU 使用率が周期的に 0% へ落ちる: 原因と改善案

実施: 2026-06-26 10:47 JST

## 背景

minibatch 256 vs 512 の A/B (step3c) を `run_ab_minibatch.sh` で実行中、nvtop で GPU 使用率が
**周期的に 0% へ落ちる**のを観測。原因を切り分けて改善案を残すため、実験を一旦停止した
(A/B 自体は未完。再開可能)。

本ファイルはコード変更前の原因調査と改善案のみ。

## 結論 (原因)

主因は **funnel が `epochs-per-step` ごとに `train-loop` を新しい subprocess として
起動・破棄していること**。

`ckpt_tournament.py` の funnel ループ ([:277-279]) は、選抜のため一定エポックごとに
snapshot を取る必要があり、`epochs-per-step` (今回の th128 examples 等価設定では **5**) 毎に
`train_to()` を呼ぶ。`train_to()` ([:158]) は毎回:

```
uv run train-loop --max-epochs <target> --checkpoint-path work.pt ...
```

を **新規 subprocess** として起動する。よって 5 エポックごとに以下が挟まる:

1. 前 train-loop プロセスの終了 (checkpoint 保存 → teardown)
2. `uv run` 解決 + 新プロセスの torch import (~1s)
3. Agent 構築 + **CUDA Graph キャプチャ** (~1-2s) + Rust executor / sim スレッド起動 +
   checkpoint ロード
4. ここでようやく学習開始 (GPU ビジー)

**手順 1→3 と 5 の間は GPU 完全アイドル = 0%**。これは head-to-head から先に除去したのと
同種のコールドスタート固定費が、学習側に残っているもの。

### なぜ今回特に目立つか

th128 の examples 等価化のため `epochs-per-step=5` と最小にしている。1 チャンクの学習自体が
短い一方、起動固定費 (~2-4s) はチャンク長によらず一定なので、相対的に 0% の谷が大きく見える。
epochs-per-step が大きい従来 A/B (例 step2 の A 手法 eps=20) では谷の頻度が 1/4 で目立ちにくい。

### 副次的な谷 (主因ではない)

- head-to-head 評価フェーズ: in-process 化済み (commit 65f5dff) かつ policy-only で軽量。
  ここの GPU は低めだが 0% 主因ではない。
- 1 回の train-loop 内の learn フェーズ・checkpoint 保存: 短い谷だが 0% ではない。

## 計測の根拠 (既存データ)

- head-to-head 調査 (experiments 20260625) で、プロセス起動床は torch import ~1.06s、
  Agent×2 ロード + CUDA Graph 構築含め ~2.83s と実測済み。train-loop も同等のコールド
  スタートを毎チャンク払う。
- A/B は結果 (強さ) には影響しない。0% は壁時計ロスのみ。

## 改善案

### 優先度1: funnel の学習を長寿命プロセス化し snapshot を内部で取る

train-loop には既に `--snapshot-every N` がある (N エポックごとに `ckpt_epN.pt` を保存)。
funnel が「5 エポック走らせて止め、また起動」する代わりに、**1 本の train-loop を起動したまま
`--snapshot-every=epochs_per_step` で連続学習**させ、detector はディスクに出てくる snapshot を
監視して消費する形にすれば、チャンク毎の再起動 (torch import + CUDA Graph + executor 起動) が
消える。

論点:

- **停止制御**: funnel は detector が finalists に達したら学習を止めたい。長寿命 train-loop に
  「いつ止めるか」を渡せない。案: (a) train-loop を background 起動し、detector が finalists 到達
  したら funnel 側から terminate する。(b) train-loop に「外部から停止ファイルを見たら終了」する
  軽い hook を足す。(c) ある程度先まで (例 +200ep) まとめて回し snapshot を出し切ってから
  detector に通し、足りなければ次バッチを継続起動 (再起動回数を 1/40 などに削減)。
- (c) が最小変更。「一度に進めるエポック数」を epochs-per-step と分離 (例 train-chunk=100,
  snapshot-every=5) すれば、再起動を 20 回に 1 回へ減らしつつ snapshot 粒度は維持できる。

### 優先度2: epochs-per-step を大きくする (超軽量 / 暫定)

単に再起動頻度を下げる。ただし:

- 選抜の snapshot 粒度が粗くなる (ピーク検出の分解能低下)。
- step2/3 で揃えた「threshold に反比例した eps で examples 等価」の条件を再設計する必要がある。
- 強さ A/B の比較条件を変えるので、確定済み step1-3 の追試には使いにくい。

恒久対策にはならないが、急ぐ場合の応急策。

### 優先度3 (補助): train-loop 自体のコールドスタート短縮

- CUDA Graph キャプチャを毎起動やり直す代わりに、warmup を省ける構成があるか検討。
- `uv run` 解決コスト (~0.2s) は `make` 経由でビルド済みなら小さい。主因は torch import と
  Graph キャプチャなのでプロセス再利用 (優先度1) が本筋。

## 推奨

優先度1の (c) 案 (train-chunk と snapshot-every の分離) が、

- 強さに影響しない (snapshot 粒度・examples 等価条件を維持できる)
- 再起動回数を桁で削減でき GPU 0% の谷をほぼ除去
- 中規模で済む (funnel ループと train_to の小改修 + train-loop の snapshot 監視)

ためバランスが良い。実装着手前に本案で合意を取る。

## 現状 / 再開メモ

- step3c (mbs256 vs mbs512) は funnel A の途中で停止。`A_state.json` から `--resume` 可能だが、
  上記改善を入れてから再開するか、改善前に素で完走させるかは未決。
- 改善は A/B 結果に影響しないため、step3c は改善前・改善後どちらで取っても比較可能。

## 関連

- head-to-head の同種固定費除去: commit 65f5dff (in-process 化, ~5.5x)。本件はその学習版。
- step1-3 経緯: experiments/poke-ai3 20260625_2050 / _2154 / _2321。
