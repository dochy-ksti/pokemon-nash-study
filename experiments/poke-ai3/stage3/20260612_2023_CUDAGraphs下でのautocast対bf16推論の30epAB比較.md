# CUDA Graphs 下での autocast vs bf16 推論の 30ep A/B 比較

## 目的

CUDA Graphs 化 (20260612_2013) でカーネル発行コストが消えた後、推論重みを
bf16 専用コピーにする意味が残るかを確認する。autocast のキャストカーネルも
グラフに焼き込まれて発行コストゼロになるため、差が消える仮説。

実装: `GraphedInferModel(autocast_dtype=...)` を追加し、`--no-infer-bf16 --infer-graph`
で fp32 学習モデルを autocast ごとキャプチャする経路を用意 (AdamW の in-place 更新は
replay に反映される)。

## コマンド

```
uv run train-loop --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --no-random --no-crit --stage 3b \
  --max-epochs 30 --battle-seed 20260612 --infer-graph [--no-infer-bf16 | --infer-bf16]
```

## 結果

| 経路 | wall (30ep) | CPU% | switch_samples 合計 | samples/s |
|---|---|---|---|---|
| graph + autocast (fp32 重み) | 116.1s | 662% | 9954 | 85.7 |
| graph + bf16 重みコピー | 121.7s | 630% | 9754 | 80.2 |

- 差は ~5% で、方策軌道の違いによる仕事量ゆらぎ (~2%) を考えると **実質同等**。
  仮説どおり、グラフ化後は bf16 専用コピーの優位は消える。
- 参考: 10ep 時点 (18.6s) より 30ep のペースが落ちるのは、交代を学習して
  試合が長引くため (switch_samples/epoch が 130 前後 → 330 前後に増加)。

## 結論

- CUDA Graphs があれば autocast (fp32 重み) と bf16 重みはスループット同等。
- fp32+autocast 経路は学習モデルと数値が一致し重み同期コピーも不要なため、
  むしろ既定を fp32+graph に寄せる選択肢もある。当面は現状の既定
  (bf16+graph) を維持し、必要なら別途判断する。
