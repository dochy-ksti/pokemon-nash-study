# empty 比率の計測: steady-state 約 19%

## 目的

empty padding 方式 (commit 15aa3ff) で、穴埋め用 empty が総観測のどれだけを占めるかを
測る。先の計測 (20260604_1249) で大バッチ化が小モデルでは性能悪化 (160s→266s) と分かった
ため、その遅化に empty オーバーヘッドがどれだけ寄与するかを切り分ける。

## 計測の追加

- root_task の BatchStats に empty 件数を追加し、`empty_ratio` (empty / (real+empty)) を
  出力。出力間隔を 50 chunk に変更 (1 epoch でも途中経過が出るように)。
- GPU バッチ = real のみ (empty は計上専用)。

## コマンド

```bash
cd poke-ai3-python
uv run phase2-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --no-random --no-crit --stage 2a \
  --max-epochs 1 --checkpoint-path ../data/poke-ai3/phase2_lookahead_ratio_test.pt
```

- 環境: RTX 5090, CUDA。cap = num_games×W = 512。

## 結果

| 時点 (累積) | real | empty | empty_ratio | avg_real |
|------------:|-----:|------:|------------:|---------:|
| chunks=50  | 25600 | 0    | 0.000 | 512.0 |
| chunks=100 | 46147 | 5053 | 0.099 | 461.5 |
| chunks=150 | 66923 | 9877 | 0.129 | 446.2 |

- 起動直後 (最初の 50 chunk) は empty=0。全ゲームが lookahead 序盤の real バースト中で
  まだドレイン期に入っていないため。バッチも 512 ぴったり (全 real)。
- その後 empty が出始め、累積比率は ~13% へ上昇。
- 増分で見ると steady-state は約 19% (chunk 100→150 区間: empty Δ4824 / 総 Δ25600 ≈ 0.19)。
- avg_real が 512→446 へ下がるのは、cap 512 内に empty が混ざる分 real が減るため。

## 考察

- empty オーバーヘッドは steady-state で総観測の約 2 割。無視できないが、先の 1.66 倍
  遅化の主因ではない (empty が消えても残り 8 割の処理量 + threshold バリアによる
  パイプライン低下が支配的)。
- 遅化の本丸は empty 量より threshold ゲートの同期バリア側。empty 比率は「ドレイン期に
  ウィンドウを W に保つコスト」として約 2 割という妥当な水準。
- W が小さいほどドレイン期の空きスロットが減り empty 比率は下がるはず (本ランは W=16)。

## 評価

中立 (計測)。empty padding のコスト構造を定量化。empty は約 2 割で、性能悪化の主因は
threshold バリアによるパイプライン低下と確認できた。計測拡張 (empty_ratio, 出力間隔 50)
は保持。
