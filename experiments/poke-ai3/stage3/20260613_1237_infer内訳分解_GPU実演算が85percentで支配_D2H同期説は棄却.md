# infer 内訳分解 — GPU 実演算が 85% で支配、「D2H 同期/発行が主因」説は棄却

## 目的

20260613_1129 で pump 1 ラウンドの infer (GPU forward + `.cpu()` D2H 同期) が
75% を占めると判明。ただし nvidia-smi の GPU util が 30% だったため「367µs の
大半は GPU 演算ではなく D2H 同期待ち + カーネル発行 + Python オーバーヘッド」と
**推測**していた。この推測を CUDA events で直接検証する。

## 方法

`Agent.infer_encoded` に 3 計測を追加 (agent.py、純 Python・Rust 再ビルド不要):

- **launch**: forward 呼び出し (graph replay / model forward) を発行し終えるまでの
  CPU wall (`perf_counter`)。カーネルをキューに積み終えた時点で返る (非同期)。
- **gpu**: forward 区間に CUDA events (`ev_start`/`ev_end`) を挟み、`.cpu()` 同期後に
  `elapsed_time` を読む = **GPU が実際に演算していた時間**。
- **d2h**: `.cpu().numpy()` の wall。GPU 完了までの同期待ち + ホスト転送を含む。

launch / gpu / d2h は時間軸上で**重なる** (launch は GPU 開始と並走、d2h は GPU
末尾の演算待ちを含む) ため合計は infer wall を超える。各々が infer wall に占める
割合で「何が律速か」を読む。

## コマンド

```bash
cd poke-ai3-python
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 30 --battle-seed 20260612 2>&1 | grep -E "pump_timing|infer_split"
```

## 結果 (80000 ラウンド時点、安定値)

```
pump_timing rounds=80000 rows=18776557 avg_rows=234.7 total_s=41.98
  recv=1.35s(3.2%)  infer=31.83s(75.8%)  send=8.79s(20.9%)
  us/round[recv=16.9  infer=397.9  send=109.9]
  infer_split launch=9.20s(28.9%) gpu=27.02s(84.9%) d2h=21.64s(68.0%)
  us/round[launch=115.0  gpu=337.7  d2h=270.5]
```

| infer 内訳 | µs/round | infer wall 比 |
|---|---:|---:|
| launch (発行 CPU wall) | 115.0 | 28.9% |
| **gpu (CUDA events 実演算)** | **337.7** | **84.9%** |
| d2h (`.cpu()` wall = 同期待ち+転送) | 270.5 | 68.0% |
| infer wall | 397.9 | 100% |

## 所見 — 前回の推測は棄却

- **GPU 実演算が infer wall の 85% (338µs/round) を占める**。infer の 398µs は
  D2H 同期や発行オーバーヘッドではなく、**ModernBERT forward の GPU 演算そのもの**が
  主因だった。20260613_1129 の「GPU 演算は小さく D2H/launch が主因」という推測は
  **誤り**。
- d2h wall 270µs は GPU 演算と**重なっている**: `.cpu()` が GPU forward の完了を
  待つため、その 270µs の大半は「コピー」ではなく「GPU 末尾演算の同期待ち」。
  launch 115µs も GPU 開始と並走。つまり 3 つは直列加算でなく重畳で、律速は GPU。
- → **infer の D2H をダブルバッファで overlap しても効果は薄い**。GPU が infer
  ラウンドの 85% を実演算で埋めており、隠せる「待ち」は実質ない。20260613_1129 で
  提案した打ち手 (1)(2) のうち、(2) pump のダブルバッファは**棄却**。

## なぜ nvidia-smi GPU util は 30% なのか (矛盾の解消)

ラウンド内では GPU は 338µs / pump 525µs (recv+infer+send) = **64% 稼働**している。
にもかかわらず実測 util 30% なのは、**pump 自体が wall 全体の半分しか回っていない**
ため。残り半分は learn (教師あり学習エポック) と、rollout 生産待ちで pump が
`is_ready()==false` の間 (convoy)。GPU は「pump が回っている間は忙しいが、
回っていない時間が長い」。util 30% は瞬間稼働率ではなく時間平均の希薄化。

## 真の律速と打ち手

GPU 実演算が batch=235 で 338µs = **1.44µs/row**。これが per-round の床。打ち手は:

1. **バッチを太らせて GPU の per-row 効率を上げる**。小バッチ (235) は ModernBERT に
   とって非効率で、倍にしても GPU 時間は倍にならず µs/row が下がる。ただしバッチは
   in-flight 観測数 (≒ 同時進行ゲーム数) で頭打ち → **num_games / sim-concurrency を
   増やして in-flight を増やす**のが本筋 (20260613_1057 の convoy 結論と接続)。
2. **GPU を連続供給する (rollout 生産と GPU を overlap)**。util 30% の主因は pump が
   断続的に止まること。rollout 生産が GPU ラウンドと重なれば util が上がり throughput
   が伸びる。これは pump 内 D2H の overlap (棄却した(2)) ではなく、**pump 全体と
   rollout 生産の overlap** の話。
3. **モデルを軽くする / バッチ効率の良い形にする** (中長期)。GPU 演算自体が律速なら、
   ModernBERT の隠れ次元・層数・seq 長の見直しが直接効く。

優先は (1): num_games を増やして avg batch を 235 から引き上げ、GPU の µs/row を
下げられるか計測する。

## 備考

計測コードは agent.py の infer_encoded / _report_pump に残置 (CUDA events 2 個 +
perf_counter のみ、オーバーヘッドは無視可能)。不要なら revert する。
</content>
</invoke>
