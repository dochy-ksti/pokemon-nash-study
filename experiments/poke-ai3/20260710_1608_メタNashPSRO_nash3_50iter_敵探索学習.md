# メタ Nash PSRO PSRO_nash3 (50 iter・敵探索学習): 前半スパイク帯 → 後半 0.55 帯へ収束。探索を train/eval で揃えると探索込みでも 0.55 頭打ち (較正ギャップ消失)

## 目的

nash2 (12 iter, 敵 policy-only 学習) は policy-only 行列で 0.55〜0.64 頭打ち、探索込みでは
0.46〜0.52 まで下がった。ただしこの低下は **学習時 policy-only / 評価時探索あり** の
train/eval ミスマッチ由来の疑いがあった。本実験で (1) iter を 12→50 に伸ばし、(2) 学習中の敵も
lookahead 着手 (ELA=1) にして探索有無を揃え、探索込みが真の exploitability かを確認する。
行列自体は policy-only のまま (探索込み head-to-head は 37x で iter² 律速のため、既定 policy-only)。

## コマンド

```bash
cd poke-ai3-python
TAG=PSRO_nash3 ELA=1 META=nash SPR=0 MAX_ITERS=50 LOGDIR=/tmp/psro \
  setsid nohup bash scripts/run_psro_pilot.sh > /tmp/psro/driver_nash3.log 2>&1 < /dev/null &
```

driver 引数: `--enemy-lookahead --meta-strategy nash --nash-eps 0.02 --matrix-n-per-side 256
--max-iters 50 --warmup-epochs 200 --central-epochs 50 --pool-size 4 --self-play-ratio 0
--value-target expected --nash-learning-rate 1.5 --depth-skew 2.0 --search-turn-min 4
--search-turn-max 8 --sims 64 --sim-concurrency 16 --train-num-games 64
--train-max-batch-size 512 --train-trajectories-threshold 128 --train-minibatch-size 256
--n-per-side 512 --num-games 256 --stage 3b --random --crit`。
実時間 ~14 時間 (14:37→04:32 UTC, 2026-07-09→10)。SSH デタッチ (setsid nohup) で完走。

## 結果 1: exploitability 推移 (= 新中心 c_k が旧 σ 混合に取る勝率、policy-only 行列由来、0.5 で収束)

| 区間 | iter | exploitability |
|---|---|---|
| 前半 (推移的追走) | 1–11 | 0.63〜**0.98** (σ 単一〜薄い混合、BR が突き放す) |
| 中盤 (混合形成) | 12–20 | 0.60〜0.74 |
| **後半 (収束帯)** | **21–49** | **0.51〜0.60**、iter26 で最小 **0.515** |

後半 10 iter (40–49) の平均 ≈ **0.553**。前半のスパイク (薄サポートを突く 0.8〜0.98) は
iter21 以降ほぼ消滅。nash2 (12 iter で 0.55〜0.64 帯・スパイク残) より明確に収束が進んだ。

最終 σ (|Π|=50 上の Nash 混合) は広く分散 (単一凍結せず): 主要 c0=0.198, c49=0.208, c40=0.136,
c21=0.082, c14=0.067, c43=0.075, c15=0.035, c42=0.037…。

## 結果 2: 最終 c49 の探索込み exploitability (σ_prev = c49 追加前 49×49 Nash 混合への勝率)

σ_prev 主要サポート 12 体 (カバー 0.97) を重み正規化して加重。各ペア 1024 戦。
スクリプト: scratchpad/verify_nash3_search_exploitability.py。

| | policy-only | 探索込み | 差 |
|---|---|---|---|
| c49 σ加重 exploitability | 0.557 | **0.552** | **+0.004** |

**nash2 との決定的な差**: nash2 は policy-only → 探索込みで **−5〜8pt** 低下したが、nash3 は
**ほぼ差なし (−0.4pt)**。

## 結論

**iter 延長で収束は進んだが、探索を揃えると探索込みでも 0.55 頭打ち。nash2 の「探索込み ≒ Nash」は
train/eval ミスマッチによる見かけの低下だった。**

1. **収束は明確に前進**: 前半スパイク帯 (0.8〜0.98) → 後半 0.55 帯へ。σ は 50 体に分散した真の
   混合を維持 (凍結・退化なし)。iter を伸ばした効果は出た。
2. **較正ギャップは消えた**: nash2 の探索込み低下 (−5〜8pt) は「学習 policy-only / 評価探索あり」の
   ミスマッチ由来だった。学習中の敵も探索させて揃えると、policy-only と探索込みが一致 (0.557 vs
   0.552)。つまり探索込み 0.552 が **真の下限 exploitability**。
3. **しかし 0.5 には未到達 (~5pt 残)**: 50 iter・敵探索学習でも double-oracle gap は完全に閉じない。
   50ep BR の強さか、Π のサポート網羅性が Nash に対してまだ不足。0.552 は **50ep BR による下限**
   (fresh BR ならさらに高い可能性) なので、真の max exploitability は 0.55 以上。

## 示唆・次アクション候補

- **BR を強める**: central-epochs 50→100、または warm-start を切って fresh BR にし、各 iter の
  best-response 品質を上げる。0.55 頭打ちが「BR が弱くて gap を測り切れていない」のか
  「Π が本当に unexploitable でない」のかを切り分ける。
- **探索込み行列 (MLA=1) で σ を組み直す**: 現行 σ は policy-only 行列の Nash。探索込みで戦うなら
  探索込み行列の Nash が正しい混合。コスト iter² だが最終 σ の質は上がりうる (小 Π で試す)。
- **深さ・sims を増やす**: 探索が強いほど混合が Nash に寄る。探索込みで 0.5 に届くかを sims 128 等で。

## 成果物・ログ

- 結果 JSON: data/poke-ai3/tournament/PSRO_nash3_psro.json (pool=c0..c49, matrix 50×50, sigma)
- state: data/poke-ai3/tournament/PSRO_nash3_psro_state.json (--resume で延長可)
- 中心スナップショット: data/poke-ai3/tournament/PSRO_nash3_c{0..49}.pt (= 集団 Π)
- ログ: /tmp/psro/PSRO_nash3.log、探索込み測定: /tmp/psro/nash3_search_expl.log
