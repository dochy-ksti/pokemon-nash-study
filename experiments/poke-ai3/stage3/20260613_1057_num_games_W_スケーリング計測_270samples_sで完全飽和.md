# num_games × W スケーリング計測 — 270 samples/s で完全飽和、RootTask 直列化が容疑

## 目的

CPU 利用率計測 (20260613_1012 追記2) で「CPU 27%・全スレッド非飽和の
レイテンシ律速」と判明。in-flight 並列度を上げればレイテンシを覆い隠せる
はずなので、num_games × sim_concurrency (W) のグリッドでスループットの
スケールを測る。

## コマンド

```bash
cd poke-ai3-python
uv run train-loop --num-games {32|64} --sim-concurrency {16|32} --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 30 --battle-seed 20260612
# max_batch_size は既定 (num_games*W/2)。num_games で 1ep の仕事量が変わるため
# 指標は学習サンプル総数 ÷ wall (samples/s)。
```

## 結果

| num_games | W | in-flight (×2P) | mbs (既定) | wall (30ep) | examples | samples/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 1024 | 256 | 70.3s | 19036 | **271** |
| 64 | 16 | 2048 | 512 | 153.5s | 41162 | **268** |
| 32 | 32 | 2048 | 512 | 70.0s | 19140 | **273** |
| 64 | 32 | 4096 | 1024 | 146.7s | 40252 | **274** |

## 所見

- **in-flight を 1024→4096 (4 倍)、バッチを 256→1024 にしてもスループットは
  ±1% で不変**。271 / 268 / 273 / 274 samples/s。レイテンシ律速仮説は棄却。
  並列度でもバッチサイズでも動かない以上、**観測 1 件あたりの直列コスト**が
  天井を作っている。
- 推論観測ベースでは ~250k obs/s (batch_stats real 19.4M / 78s) で頭打ち。
- 容疑筆頭は **RootTask の encode_batch + pack の直列実行**。20260612_2200 で
  メインスレッドから RootTask (単一 async タスク) へ offload したが、RootTask は
  1 タスクなので全観測のエンコードがそこで直列化する。当時の実測
  1.26ms/256obs ≈ 4.9µs/obs → 理論天井 ~204k obs/s で、実測 ~250k obs/s と
  オーダー一致。CPU 計測で tokio ワーカーが最大 53% だったのは、エンコード
  負荷が複数ワーカーに分散して見えるため矛盾しない (RootTask 自体の専有率は
  スレッドではなくタスク単位なので pidstat では直接見えない)。
- 対案 (未実施): RootTask のエンコードを並列化する。
  - send_batch 時に rayon / spawn_blocking でバッチ単位に並列エンコード
  - または RootTask を観測シャーディングで複数化
  - または encode を game_task 側 (観測生成元) で行い、RootTask は連結のみにする
## 追記: GPU 利用率も非飽和 (nvidia-smi, cap=256 実行中)

```
utilization.gpu: 14〜38% (平均 ~30%)
utilization.memory: 0%
power.draw: 128〜144W (RTX 5090 上限 ~575W に対し約 1/4)
```

- **GPU も飽和していない**。CPU 27% / GPU 30% / メモリ帯域 0% で、計算資源は
  どれも余っている。util が 14〜38% で振れるのは 20260613_0943 のバッチ脈動
  (谷) と整合。
- これでボトルネックは計算 (CPU/GPU) ではなく **per-observation の直列制御
  パス (RootTask の単一タスクでの encode+pack、および Python↔Rust の往復)**
  に確定。資源ではなくパイプライン構造の問題。

## 結論

スループット天井 ~270 samples/s (~250k obs/s) は並列度・バッチサイズ非依存で、
CPU(27%)・GPU(30%)・メモリ帯域(0%) すべて非飽和の直列ボトルネック。
次の一手は RootTask の encode+pack 直列化の解消:
(1) send_batch のエンコードを rayon / spawn_blocking で並列化、
(2) または RootTask を観測シャーディングで複数化、
(3) または encode を game_task 側へ戻し RootTask は連結のみにする。
まず (1) が最小変更で効果を見やすい。
