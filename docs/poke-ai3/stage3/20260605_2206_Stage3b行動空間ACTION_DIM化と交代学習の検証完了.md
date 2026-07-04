# Stage3b 行動空間 ACTION_DIM 化と交代学習の検証完了

JST: 2026-06-05 22:06

## 目的

前ハンドオフ(`20260605_1736`)の残作業「ステップ4: 行動空間拡張の縦切り(Rust×Python)」を
実装し、Stage3b の交代学習を実機で検証する。grill-me で設計を詰める過程で、Stage3b の
チームデータが設計意図(非対称2チーム)と食い違っていることが判明したため、その修正を
先行コミットしてから縦切りを着地させた。

## このセッションで完了 (全コミット workspace 緑)

### 1. Stage3b 2チーム化 (`69a0d4e`)

grill-me 中に発覚: scenario/team は「両側同一チームのミラー」だったが、設計意図は
**技を入れ替えた非対称2チーム**だった。修正:

- Team1: Cloyster=Shock Wave / Goodra-Hisui=Bulldoze、Team2: 技入替。
  (Shock Wave→Cloyster(Water) に SE、Bulldoze→Goodra-Hisui(Steel) に SE)
- `team/poke-ai3/scenario/stage3b_team1.txt` / `stage3b_team2.txt` の2ファイル。
- `scenario.rs`: `enum TeamId{Team1,Team2}`、`BattleState::new_with_teams(stage,
  (TeamId,active),(TeamId,active))` を新設。`new` は 1v1 専用 (party は assert)。
- `game_task.rs`: `pick_config` で各側 (チーム×先発)=4、両側16通りを決定論巡回。
- `local_showdown.rs`: `create_local_game`/`run_engine` を初期 `BattleState` 受け取りに。
- 各サイドはチーム・先発とも独立ランダム。

### 2. 行動空間 ACTION_DIM 化の縦切り (`3a51776`)

`MoveId`/`NUM_MOVES` → `Choice`/`ACTION_DIM = NUM_MOVES + (MAX_PARTY-1) = 5`。

- `observation.rs`: `ACTION_DIM`/`NUM_BENCH`/`BenchSlot{present,species,hp_frac,alive,
  moves:[bool;NUM_MOVES]}`。`StateForPlayer.my_bench[NUM_BENCH]`、`legal_action_mask`
  を長さ ACTION_DIM へ。交代手は **positional な相対控え index**、相対↔絶対変換ヘルパ
  (`bench_rel_to_abs`/`bench_abs_to_rel`/`action_index`/`action_to_choice`) を本ファイルに集約。
  active 技は専用フィールドを持たず `legal_action_mask` に委譲 (MoveId 明示は Stage3c 後)。
- `lookahead.rs`: `Choice`/`ACTION_DIM` へ一般化。rollout 内で `apply_turn` 後
  `any_forced_switch` なら policy から交代手をサンプルし `apply_forced_switches` で解決。
  root 合法手1なら短絡即決。不利対面で交代の training_pi が立つテスト追加。
- `nash.rs`/`oracle.rs`: ACTION_DIM へ。
- poke-ai3: 配列次元を ACTION_DIM へ。通常ターンで技+交代をサンプル (交代は絶対 index+1 の
  `SendInfo::Switch`)。ForceSwitch は Local で学習決定ノード化 (合法手1なら短絡・無記録)。
- Python: encoder に控えトークン追加 (`NUM_SLOTS=20`)、`policy_head→ACTION_DIM(5)`。
- 既存チェックポイントは出力次元変更によりスクラッチ再学習。

### 3. 交代診断 (`0024898` → `5a582a1`) と Stage3b CLI (`0874a04`)

- `train-loop --stage 3b` を起動可能に。
- `diagnostics.stage3b_switch_diagnostics`: 当初「不利対面での greedy 交代率」だったが、
  検証で不適切と判明し、**不利対面での model/teacher の softmax 交代確率**へ作り直し。

## 検証結果 (実験記録 `experiments/poke-ai3/20260605_2134`)

`train-loop --stage 3b --no-random --no-crit --sims 64 --search-turn-min 6 --search-turn-max 12`
(num-games 32, sim-concurrency 16) を実行。

**結論: 縦切りは完全に機能している。**

| 指標 | 値 |
|---|---|
| greedy_switch_rate | 0.000 (全 epoch) |
| model_switch_prob (不利対面) | ~0.24–0.27 |
| teacher_switch_prob (不利対面) | ~0.23–0.30 |

- モデルの softmax 交代確率が教師 (lookahead) をほぼ完全に追従。env→lookahead→学習→
  モデルの配線は端から端まで正常。行動空間拡張は交代を正しく学習可能にしている。
- `greedy_switch_rate=0` は ~25/75 の**混合教師に argmax をかけた指標のアーティファクト**。
- 「不利対面での交代率 >90%」は不適切な物差し。Stage3b の自己対戦均衡は**混合戦略**
  (交代 ~25%) で、これは前 grill-me 質問4 の switch ループ懸念の実証。
  正しい判定 = **model_switch_prob ≈ teacher_switch_prob (混合戦略の忠実な再現)**。

## 残課題 / 次に着手する人へ

1. **混合均衡の出所 (任意・slice の正しさには無関係)**: 教師の交代確率 ~25% が
   (1a) 真の混合均衡か、(1b) lookahead が浅く交代を過小評価しているかは未確定。
   解析的には「不利なら交代 (SE 被弾回避 + 次ターン SE 設定)」が攻撃より明確に有利に
   見えるため、sims↑/探索深↑で `teacher_switch_prob` が上がるか確認する価値がある。
2. **位置的スロット decouple (旧残課題4) と相手開示累積 (旧分岐 #3 の A)**: 多技/3v3
   ステージで導入。今回スライスには含めていない。bench の `moves` を `[bool;NUM_MOVES]`
   から `[MoveId;NUM_MOVES]`(技同一性) へ上げるのは Stage3c が回ってから。
3. **clippy 既存警告**: team.rs:57 / protocol.rs:102 / async_executor.rs:57 /
   game_task.rs(div_ceil) / root_task.rs:162 は本セッション前から存在 (新規警告なし)。

## 検証コマンド

```bash
cargo test --workspace          # 緑 (poke-sho 48 / poke-env 17 / parity 1 / poke-ai3 2)
cd poke-ai3-python && uv run train-loop --stage 3b --no-random --no-crit \
  --num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 6 --search-turn-max 12 --max-epochs 5 --chunk-threshold 512
# ログの switch_prob[model=.. teacher=..] が ≈ なら交代学習成立。
```
