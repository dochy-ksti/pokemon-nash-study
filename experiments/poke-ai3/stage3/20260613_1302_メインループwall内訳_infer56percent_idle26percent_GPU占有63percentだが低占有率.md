# メインループ wall 内訳 — infer 56% / idle 26% / learn 17.5%、GPU は wall の ~63% 占有だが低 SM 占有率

## 目的

20260613_1237 で infer の 85% が GPU 実演算と判明 (D2H 同期説は棄却)。だが
nvidia-smi util は 30% のまま。この矛盾を解くため、メインループ (train_loop.py の
while) の wall を learn / infer pump / idle (バッチ未到着待ち) の 3 区分に分けて、
**pump が実際に wall の何 % 回っているか**、**GPU が遊んでいる残り時間の正体**を測る。

## 方法

`run_train_loop` の while ループに 3 バケット計測を追加 (純 Python):

- **learn**: `trajectories_ready()` 真 → recv + `agent.learn` (教師あり学習エポック)。
- **infer**: `is_ready()` 真 → `agent.infer_step` (推論 pump)。
- **idle**: どちらも偽 → `time.sleep(sleep_seconds=0)` (ビジーポーリング空振り)。

各分岐直前のポーリング (drain) コストは発火した分岐に計上。エポックごとに累積を print。

## コマンド

```bash
cd poke-ai3-python
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 30 --battle-seed 20260612 2>&1 | grep loop_timing
```

## 結果 (epoch 30、安定値)

```
loop_timing epochs=30 wall=67.07s
  learn=11.75s(17.5%,n=30)
  infer=37.79s(56.3%,n=71313)
  idle=17.41s(26.0%,n=309772)
```

| 区分 | wall 比 | n | 中身 |
|---|---:|---:|---|
| infer (pump) | **56.3%** | 71313 | lookahead ロールアウト推論。うち 85% が GPU 実演算 (1237) |
| idle (待ち) | **26.0%** | 309772 | rollout 生産が間に合わず pump にバッチが無いビジーポール空振り |
| learn | 17.5% | 30 | 教師あり学習エポック (これも GPU forward+backward) |

samples/s ≈ examples/wall ≈ 19000/67 ≈ 284 で 1057 の ~270 と整合。

## 所見 — GPU util 30% の矛盾が完全に解消

- infer 56.3% のうち 85% が GPU 実演算 (1237) → **wall の ~48% が GPU 推論で占有**。
  learn 17.5% もほぼ GPU。→ **GPU は wall の ~63% を実際に占有**している。
- なのに nvidia-smi util が 30% なのは、**batch=235 の小バッチでカーネルが小さく
  SM 占有率が低い**ため。nvidia-smi の util は SM 稼働の瞬間サンプルなので「壁時計
  では忙しいが 1 カーネルで GPU を使い切れていない」状態を低く出す。**GPU は時間的
  には忙しいが効率的に使えていない**——これが util 30% の正体。CPU 27% も同様に
  「直列パスでコアを使い切れていない」希薄化。
- **idle は 26% しかない**。rollout 生産を完璧に追いつかせても消せるのは最大 26%
  (≒ throughput +35% 上限)。GPU を連続供給する打ち手 (1237 の(2)) は**副次的**。

## レバーの優先順位 (更新)

最大の塊は infer 56% = **lookahead の sims=64 ロールアウト推論**。samples/s (実意思
決定) が obs/s (~250k) よりはるかに小さいのは 1 サンプルあたり大量のロールアウト
推論を消費するため。よって:

1. **obs-per-sample 比 (sims / rollout 深さ) の削減** ← 本命。infer 56% の大半が
   ロールアウト推論なので、sims を 64 から減らせば samples/s に直接効く
   (lookahead 品質とのトレードオフ)。これが最大かつ最も直接的なレバー。
2. **GPU の SM 占有率向上 (バッチを実効的に太らせる)**。ただし 1057 で mbs を
   256→1024 にしても samples/s 不変だった = 小バッチ低占有でラウンドが延びて相殺。
   実効バッチが本当に太るか (batch_stats avg_real が cap に追随するか) の確認が要る。
3. **idle 26% の解消 (rollout 連続供給)** ← 副次的。上限 +35%。

## 結論

天井 ~270 samples/s は「GPU/CPU が遊んでいる」のではなく「**GPU が wall の ~63% を
小バッチ低占有で占有し、その大半を lookahead ロールアウト推論に費やしている**」状態。
資源の空き (util 30%) は飽和不足ではなく**低占有率による希薄化**。直接効くのは
sims/rollout 深さの削減 (obs-per-sample を下げる) で、バッチ拡大や idle 解消は副次的。

## 備考

計測コードは train_loop.py の while ループに残置 (perf_counter のみ、エポックごと
1 print)。agent.py 側 (pump 内訳 / infer 内訳) も残置。不要なら revert する。
</content>
