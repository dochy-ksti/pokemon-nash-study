# Phase 1 卒業記録

## 目的

Phase 1 (1v1 同種ポケモン、Tackle 40 / Strength 80 の威力差を学ぶ最小タスク) を
卒業し、Phase 2 (物理/特殊技を相手の防御特化/特防特化に合わせて選ぶ) に移行する。

## Phase 1 で達成したこと

- `poke-sho-rust` 上の決定論バトルシミュレータ (Mew vs Mew、Tackle/Strength) の実装。
- ダメージ計算式が Showdown と一致 (16段ダメロール、急所、STAB、type effectiveness の
  4096固定小数 modify を含む)。
- `poke-env-rust` の env API (HP%, legal mask) と Showdown subprocess 駆動 (`showdown_trait`)。
- `tests/parity.rs` で Showdown とのイベント列パリティを 20 ゲーム検証。
- `poke-ai3` の async executor + game task で 2 プレイヤー分の trajectory を収集。
- `poke-ai3-python` で PPO ループを回し、Tackle vs Strength の選択を学習可能と確認
  (実験記録: `20260510_1816_phase1_ppo_20epochs.md`)。

## Phase 2 への引き継ぎ

Phase 1 専用コード (`phase1.rs`、`PHASE1_TEAM_TEXT`、`phase1_*` Python モジュール、
`team/poke-ai3/phase1/`) はこのコミットで削除する。

引き継ぐもの:
- 共通モジュール (`damage.rs`, `event.rs`, `battle_rng.rs`, `types.rs`, `moves.rs`,
  `species.rs`, `team.rs`) — 種族/技/タイプ表を Phase 2 用に拡張して再利用。
- Showdown subprocess 駆動 (`showdown_trait.rs`, `local_showdown.rs`, `protocol.rs`)
  — フォーマットは引き続き `gen9customgame`。
- 決定論モード (`MaxRoll`) と乱数モードの両対応の枠組み。

Phase 2 の設計詳細は同セッションの設計議事録参照。
