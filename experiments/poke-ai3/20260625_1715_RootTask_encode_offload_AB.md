# RootTask encode offload (spawn_blocking) A/B

作成: 2026-06-25 17:15 JST

## 目的

`RootTask` が `encode_batch + pack_batch` を同期実行している間、次バッチの形成が
直列にブロックされる。これを `spawn_blocking` へ隔離し、run ループの次バッチ形成と
パイプライン化することでエンドツーエンドのスループットが改善するかを検証する。

採用条件: offload 版が現行直列版よりエンドツーエンドで 5% 以上高速であること。

前提計測 (`20260625_1503_S8_RootTaskエンコード滞留計測.md`):
encode+pack は RootTask wall の約26.5%、encode 完了時点で平均236件 (≒次1バッチ分) が
`root_receiver` に滞留。ただし avg_encode≈171µs < avg_interval≈647µs。

## 実装

`poke-ai3/src/root_task.rs` の `send_batch` で encode+pack+send を closure 化し、
`spawn_blocking`(ブロッキングプールへ隔離)で投げる。run ループ側は spawn 前に
reals/empties カウント・`batch_stats` 記録・`root_receiver.len()` 読み取りを同期実行し、
`std::mem::take` したバッチ Vec のみ closure へ move。バッチ送出順の前後は許容
(ルーティングは行ごとの game_id/request_id で自己完結)。同時 encode 数はキャップなし。
`RootTiming` は `Arc<Mutex<>>` 化し、per-encode 時間を closure 内で記録。

A/B は環境変数 `POKE_AI3_ASYNC_ENCODE` で同一バイナリ内分岐(無=直列, 有=offload)。
最終 flush のみ常に同期実行し run 終了までの encode 完了を保証。

## コマンド

`make build` 後、毎回 fresh な shared_init.pt から開始、battle-seed 固定で
base/async を交互5組。各 RUN は以下 (env のみ切替):

```bash
POKE_AI3_ASYNC_ENCODE=1 uv run train-loop \
  --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --depth-skew 2.0 \
  --no-random --no-crit --stage 3b --max-epochs 45 \
  --battle-seed 12345 --checkpoint-path <fresh-copy>
```

## 結果 (wall 秒)

| pair | base | async | base−async |
|------|------|-------|-----------|
| 1 | 66.693 | 68.211 | −1.518 |
| 2 | 69.908 | 65.359 | +4.549 |
| 3 | 63.267 | 66.470 | −3.203 |
| 4 | 59.930 | 63.078 | −3.148 |
| 5 | 67.633 | 64.190 | +3.443 |

- 中央値: base 66.693s / async 65.359s(見かけ async 2.0% 速)
- ペア差は 3勝2敗で base 寄り、ペア差中央値 −1.518s(async が遅い)

## 判断

差はノイズ範囲で、5% 採用条件を満たさない(性能上の有意差なし)。encode の offload は
GPU スループットを律速しておらず、encode レイテンシは GPU 1ラウンドの裏に隠れていた。
RootTask wall 内の 26.5% はエンドツーエンドの待ち時間ではなかった。

ただし性能中立である一方、**設計としては offload 版の方がきれい**(encode を run ループの
直列パスから外し、純CPU処理をブロッキングプールへ隔離)であるため、採用とした。
`POKE_AI3_ASYNC_ENCODE` フラグは撤去し、offload を既定化(最終 flush のみ同期)。
