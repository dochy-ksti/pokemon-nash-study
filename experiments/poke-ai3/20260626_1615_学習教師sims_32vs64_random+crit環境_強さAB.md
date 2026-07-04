# 学習教師 sims 32 vs 64 (random+crit 環境) 強さ A/B

実施: 2026-06-26 16:15 JST (run funnel RC64 → rate×2, 06:xx–07:14)

## 目的・仮説

[20260625_1643 の sims 実験](20260625_1643_S8_sims半分通常2倍_強さ比較.md) では **乱数なし**で
sims 32/64/128 が実質同等 (差 < ノイズ) だった。
仮説: **乱数 (16段ロール=random / 急所=crit) が入ると、探索が確率的な結果を平均する必要が
増えるため、学習時の探索教師 sims を増やすと方策が強くなる**のではないか。

funnel の eval は policy-only (MCTS なし) なので、sims は **学習教師の質だけ**に効く
(eval では未使用)。よって本 A/B は「乱数環境で教師 sims を倍にすると、出来上がる方策が
強くなるか」を測る。

## 手法

両アームとも shared_init.pt から random+crit (学習・選抜とも) で funnel 学習、
**差分は学習 `--sims` のみ**。

| アーム | sims | 状態 |
|---|---:|---|
| RC | 32 | 既存流用 ([20260626_1532 RC vs A](20260626_1532_randomcrit学習RC_vs_決定論学習A_クロスルール強さ.md) の RC、RC_ep85/185/240) |
| RC64 | 64 | 新規 funnel (RC64_ep110/170/295) |

- 共通: depth-skew 2.0 / search 4-8 / sim-concurrency 16 / g64 b512 th128 /
  minibatch 256 / epochs-per-step 5 / block 50 / warmup 10 / finalists 3 / stage 3b /
  --random --crit。
- driver: `scripts/run_rc64_vs_rc.sh`。コード改修は前回の `--random/--crit` 追加で対応済み。
- rate: RC finalists 3 + RC64 finalists 3 = 6 個を 2 ルールで実施、各 n-per-side 512。

## 結果

### ルール (1) random+crit

| ckpt | 手法 | レート |
|---|---|---:|
| RC#1 (RC_ep185) | RC(32) | +6.1 |
| RC64#0 (ep110) | RC64(64) | +3.4 |
| RC64#1 (ep170) | RC64(64) | +2.8 |
| RC64#2 (ep295) | RC64(64) | +2.1 |
| RC#0 (RC_ep85) | RC(32) | -7.0 |
| RC#2 (RC_ep240) | RC(32) | -7.5 |

**手法平均: RC64(64) = +2.8 / RC(32) = -2.8 → 差 5.6 Elo、RC64 が上**

### ルール (2) no-random/no-crit

| ckpt | 手法 | レート |
|---|---|---:|
| RC64#0 (ep110) | RC64(64) | +11.0 |
| RC#1 (RC_ep185) | RC(32) | +4.2 |
| RC#2 (RC_ep240) | RC(32) | +0.8 |
| RC64#2 (ep295) | RC64(64) | +0.1 |
| RC64#1 (ep170) | RC64(64) | -6.9 |
| RC#0 (RC_ep85) | RC(32) | -9.2 |

**手法平均: RC64(64) = +1.4 / RC(32) = -1.4 → 差 2.8 Elo、RC64 が僅差で上**

## 結論

- **random+crit 環境では sims64 (RC64) が sims32 (RC) より強い (差 5.6 Elo)**。
  過去 A/B のノイズ帯 (±1.3〜1.5) の約 3-4 倍。さらに **RC64 の finalists 3 個が
  +3.4/+2.8/+2.1 と密にクラスタし全て 0 上**、一方 RC は +6.1/-7.0/-7.5 と大ばらつき。
  RC64 が一貫して RC の中央値より上 = コヒーレントな優位シグナル
  ([step3c の minibatch 判定](20260626_1258_minibatch256vs512_強さAB_step3c.md) と同型の根拠)。
- **決定論環境でも RC64 が僅差で上 (差 2.8 Elo)** だが、ノイズ帯の約 2 倍程度で
  ばらつきも大きく (RC64: +11.0/-6.9/+0.1)、random+crit ほど明確ではない。
- **仮説は支持される**: 乱数なし学習では sims 32/64/128 が同等だった
  ([20260625_1643](20260625_1643_S8_sims半分通常2倍_強さ比較.md)) のに対し、
  **乱数あり学習では教師 sims を 32→64 に倍増すると方策が明確に強くなる (random+crit で +5.6 Elo)**。
  「乱数が入ると探索 sims を増やす価値が出る」という直観と整合。

### 留意

- 効果は random+crit ルールで最も大きく、決定論ルールでは縮む。これは「乱数環境での
  教師品質向上」が主因であることと整合的 (乱数下で平均化が効く)。
- sims64 は生成スループットが落ちるため funnel は wall-clock で長い。強さ向上が
  この計算コスト増に見合うかは別途要検討 (本実験は強さの有無の確認まで)。
- さらに sims128 まで伸ばして単調かどうか、再現ラン (別シード) で +5.6 Elo が安定かは
  今後の課題。

## 成果物

- RC64 finalists: `data/poke-ai3/tournament/RC64_finalists.json` (RC64_ep110/170/295.pt)
- RC finalists: `data/poke-ai3/tournament/RC_finalists.json` (流用)
- 実行ログ: `scratchpad/rc64_vs_rc.log`
- driver: `poke-ai3-python/scripts/run_rc64_vs_rc.sh`
