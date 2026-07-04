# RootTask busy/idle 計測 — RootTask は天井ではない (encode busy 24.5% / recv idle 61%)

## 目的

20260613_1057 (スケーリング計測) は「スループット天井 ~270 samples/s は
RootTask の encode+pack 直列化が容疑」と結論したが、これは推論ベースの
間接推定 (旧実測 4.9µs/obs → 理論 ~204k obs/s) に依拠していた。RootTask が
本当に天井なら、RootTask::run は wall のほぼ全時間を encode+pack に張り付いて
いる (busy ≈ 100%) はず。直接計測で切り分ける。

## 方法

`RootTask::run` に計測を追加 (root_task.rs):

- `idle_ns`: `root_receiver.recv().await` でブロックしていた累積時間。
- `encode_ns`: `send_batch` 内の `encode_batch + pack_batch` の累積時間。
- `wall_ns`: run 開始からの経過。
- 50 バッチごとに `root_timing` 行を eprintln (RootTask は全 sender drop まで
  終了しないため、プロセスは run 終了前に exit する。よって定期出力にした)。

release ビルド (`uv run maturin develop --release`) で計測。

## コマンド

```bash
cd poke-ai3-python
uv run maturin develop --release
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 30 --battle-seed 20260612 2>&1 | grep root_timing
```

## 結果 (最終付近の定期出力)

```
root_timing wall_ms=37554.7 encode_ms=9217.1 (busy=24.5%) idle_ms=22915.1 (idle=61.0%) encoded_obs=16031369 per_obs_us=0.57
```

| 指標 | 値 |
|---|---|
| encode+pack busy | **24.5%** of wall |
| recv().await idle | **61.0%** of wall |
| その他 (send/分岐/集計) | ~14.5% |
| per-obs encode コスト | **0.57µs/obs** |

## 所見

- **RootTask は天井ではない**。encode+pack は wall の 24.5% にすぎず、
  61% は次の観測の到着待ち (recv idle)。容疑は棄却。
- per-obs コストは **0.57µs/obs** で、20260612_2200 当時の実測 4.9µs/obs から
  約 9 倍速くなっている。パック行列化 (21→3 配列) と release ビルドの効果。
  単一タスク直列でも理論天井は ~1.75M obs/s で、実測 ~250〜430k obs/s を
  まったく律速しない。1057 の「~204k obs/s で頭打ち」推定は前提が古かった。
- RootTask が 61% idle ということは、天井は **上流 (rollout 生産)** にある。
  game_task / シミュレータ / lookahead が観測を供給しきれていない、もしくは
  各ゲームが「GPU 応答を待ってから次の観測を出す」同期構造 (convoy) で
  in-flight 並列度がラウンドのレイテンシを埋めきれていない。これは
  20260613_1012 追記2 の「レイテンシ律速 (convoy)」結論と完全に一致し、
  1057 の RootTask 容疑説を上書きする。

## 結論

- 直列 encode は天井ではない (busy 24.5%)。RootTask の並列化
  (rayon / シャーディング / game_task 側 encode) は **やっても無駄**。
- 真のボトルネックは RootTask より上流 = rollout 生産側のレイテンシ律速。
  次に攻めるべきは:
  (1) game_task が GPU 応答待ちで idle している時間 (in-flight 並列度 vs
      GPU ラウンドのレイテンシ) の直接計測、
  (2) lookahead / シミュレータ 1 手あたりの CPU コスト、
  (3) Python↔Rust の推論往復 (oneshot ルーティング、ack タイミング) の遅延。
- 20260613_1012 追記2 の convoy 仮説が計測で裏付けられた形。num_games/W の
  さらなるスケールアップ、もしくはゲーム側の非同期度 (1 ゲームが複数推論を
  同時 in-flight にできるか) を見直すのが本命。

## 備考

計測コードは root_task.rs / async_executor.rs に残置 (定期 report のみ、
オーバーヘッドは recv ごとの Instant 2 回で無視可能)。不要なら revert する。
</content>
</invoke>
