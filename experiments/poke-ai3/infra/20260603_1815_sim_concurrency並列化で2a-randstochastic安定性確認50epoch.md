# sim_concurrency 並列化で 2a + random/crit の学習安定性確認 (50 epoch)

## 目的

AGENTS.md のフェーズ計画 (決定論フェーズが動いた後、accuracy/急所/16段ダメージ乱数を
加えて学習が安定するか試す) に沿い、2a-det で収束済みのモデルに 16 段ダメージ乱数 +
急所を加えて、並列方式 (sim_concurrency=8) でも学習が安定して維持されるかを確認する。

## コマンド

```bash
# 収束済み det チェックポイントを温存するためコピーしてから継続学習
cp data/poke-ai3/phase2_lookahead_2a_det_par.pt data/poke-ai3/phase2_lookahead_2a_rand_par.pt

cd poke-ai3-python
uv run phase2-loop \
  --num-games 16 --sim-concurrency 8 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --random --crit --stage 2a \
  --max-epochs 50 \
  --checkpoint-path ../data/poke-ai3/phase2_lookahead_2a_rand_par.pt
```

- 環境: RTX 5090, CUDA。
- 2a-det 300 epoch 収束済み (experiments/poke-ai3/20260603_1807) のコピーから継続。

## 結果

exit 0、50 epoch 完走、エラーなし。

| epoch | correct_action_rate | vs_cloyster_special | vs_goodra_special | value_loss | entropy | raw_logits_std |
|------:|--------------------:|--------------------:|------------------:|-----------:|--------:|---------------:|
| 1  | 1.000 | 1.000 | 0.000 | 0.0147 | -    | -    |
| 10 | 1.000 | 1.000 | 0.000 | 0.0100 | -    | -    |
| 30 | 1.000 | 1.000 | 0.000 | 0.0162 | -    | -    |
| 50 | 1.000 | 1.000 | 0.000 | 0.0066 | 0.164 | 1.646 |

- **全 50 epoch を通して correct_action_rate = 1.0 を完全維持**。対面別診断も
  vs_cloyster_special=1.0 / vs_goodra_special=0.0 で終始ブレなし。
- 乱数 (16 段ダメージ + 急所) を加えても決定論で学んだ撃ち分けは崩れず、むしろ
  `entropy` は 0.40 付近 → 0.14〜0.16 へさらに低下、`raw_logits_std` は ~1.0 → ~1.65 へ
  上昇し、方策はより鋭く確信的になった。
- `policy_loss` は epoch ごとに 0.07〜0.38 と振れるが (乱数由来のターゲット変動)、
  `value_loss` は 0.005〜0.016 で安定して低い。

## 評価

ポジティブ。並列方式 (sim_concurrency=8) は確率的設定 (--random --crit) でも学習を
不安定化させず、決定論で獲得した最適技選択を完全に維持・さらに鋭利化した。
provisional 0.5 / 非決定的完了順を許容した設計でも、2a 規模では学習安定性に問題なし。

## 次の候補

- ゼロからの確率的学習 (det 継続でなく random/crit でスクラッチ) でも収束するか。
- stage 2b (比率 1.57x、信号が弱い) + random/crit での安定性。
- sim_concurrency を 16/32 に上げた際の安定性と速度。
