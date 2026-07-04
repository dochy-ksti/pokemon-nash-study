# Stage3b 交代機構の simulator 層実装

JST: 2026-06-05 16:33

## 目的

Stage3a (タイプ相性・4技1v1) 完了後の次増分として、**交代機構 (Stage3b)** を導入する。
本セッションは grill-me で設計を確定したうえで、合意どおり **simulator 層 (poke-sho-rust)
を先行実装**し、`cargo test --workspace` で固めるところまでを完了。env / model 層は次増分。

Stage3b の狙い: 「自分の場のポケと相手の場のポケの相性が悪いとき、有利な方に交代する」を
学習させる。

## 設計 (grill-me で確定した合意)

### チーム構成
- 両側とも **{Cloyster, Goodra-Hisui} の 2 体パーティ**。各個体は相手の弱点を突く専用技
  **1 つだけ**を持つ:
  - **Cloyster**: Shock Wave のみ (でんき特殊、相手 Cloyster の みず2倍 + 低SpD に刺さる)。
  - **Goodra-Hisui**: Bulldoze のみ (じめん物理、相手 Goodra-Hisui の はがね2倍 + 低Def に刺さる)。
- 種族値・ステータスは Stage3a の鏡像 (Cloyster B=222/D=94、Goodra-Hisui B=94/D=222) を流用。
- 両側同一チームなので、**先発だけをランダムに変えれば対称**。

### 最適戦略 (評価指標の根拠)
各個体の killer 技は「自分と同じ種族の相手」を倒す:
- Cloyster の Shock Wave は **相手 Cloyster** に刺さる (対 Goodra-Hisui は電気が等倍 + 高SpD で無力)。
- Goodra-Hisui の Bulldoze は **相手 Goodra-Hisui** に刺さる (対 Cloyster は地面が等倍 + 高Def で無力)。

→ 最適戦略は **相手の場の種族をミラーする交代** (相手 Cloyster なら自 Cloyster を、相手
Goodra なら自 Goodra を場に出す)。不利対面 (自分の場が相手をミラーできていない) での
適正交代率 >90% を定量判定する想定。

### action 空間
- `MAX_PARTY` 定数を導入し、action 次元 = `NUM_MOVES + (MAX_PARTY-1)` の固定長。
  Stage3b は `MAX_PARTY=2` → 交代スロット 1 個。3v3 拡張時はこの定数だけ上げる。
- `Choice` を `Move(MoveId) | Switch(party_index)` の enum に変更。

### 強制交代 (瀕死後)
- **Showdown 準拠の独立意思決定点**としてモデル化。技で相手が瀕死になると、ターン終了後に
  そのサイドへ `forced_switch` フラグを立て、別ステップ (`apply_forced_switches`) で控えを
  選ばせる (技スロット非合法・交代スロットのみ合法)。

### 観測 (次増分で実装)
- **Showdown 準拠 (相手控え隠匿)**。自分の控えは種族+実数HP を完全可視、相手の控えは
  reveal まで種族隠匿。固定長 `MAX_PARTY` 分の控え枠を確保する。

### BattleState 構造
- **一律パーティ化**。`BattleState.parties: [Party; 2]`。1v1 ステージ (2a/2b/3a) は
  メンバー 1 体・active=0・交代手なしの退化ケースとして同じコードパスで扱う。

### ターン解決順
- 交代は全ての技より先 (交代フェーズ → 技フェーズ)。技は交代後の新しい場のポケに当たる。
- 技フェーズは `first_player` (ChaCha8 の速度タイ) の速度順。

## 変更ファイル (poke-sho-rust)

- `src/party.rs` (新規): `PokemonState` (battle.rs から移動)、`Party{members,len,active}`、
  `MAX_PARTY=2`、`active_mon`/`has_living_bench`/`all_fainted`/`switch_targets` 等のヘルパ。
- `src/battle.rs`: `BattleState{parties, turn, forced_switch}`、`Choice::{Move,Switch}` enum、
  交代込みの `legal_choices` (強制交代中は交代手のみ)、`is_lost`/`winner` をパーティ全滅判定に。
  ターン解決を `turn` へ分離し re-export (外部 import `poke_sho_rust::battle::apply_turn` は維持)。
- `src/turn.rs` (新規): `apply_turn` (交代フェーズ→技フェーズ→強制交代要求)、
  `apply_forced_switches`、`execute_action`、`TurnResult`。300 行ルール対応のため battle.rs から分割。
- `src/scenario.rs`: `Stage::Stage3b` (short_name "3b")、`Stage::is_party()`、`build_party()`
  (1v1 は単体・3b は固定2体で先発を active に)、`SpeciesId::from_species_name`
  (Goodra-Hisui→Goodra)。`team_text` は party ステージで panic (build_party を使う)。
- `src/event.rs`: `Switch{who,species,hp,max_hp}` イベント追加 (`|switch|` 相当)。
- `src/team.rs`: `resolve_team` (全メンバーを party 順で解決) を追加。
- `src/lib.rs`: `pub mod party;` `pub mod turn;`。

### team データ
- `team/poke-ai3/scenario/stage3b_team.txt` (新規): Cloyster (Shock Wave のみ) +
  Goodra-Hisui (Bulldoze のみ) の 2 体。ステータスは Stage3a 鏡像を流用。

### poke-env-rust (Choice enum 化への最小追従のみ)
- `src/observation.rs` / `src/lookahead.rs`: `Choice` を enum 化に追従し、現状は
  **技スロットのみ抽出** (`Choice::Move` だけ取り出す)。交代の観測・行動対応は次増分。
- `src/local_showdown.rs`: `Choice { move_id }` → `Choice::Move(...)`。

> 注: 1v1 ステージは len=1 で控えがないため、`forced_switch` は常に false・交代手も
> 0 件となり、挙動は完全に従来どおり (既存テスト全通過で確認)。

## 検証

- `cargo test --workspace` 全通過 (poke-sho-rust 43 / poke-env-rust 15 / parity 1 ほか)。
- 新規テスト (poke-sho-rust):
  - `stage3b_has_two_members_and_switch_choice`: 2 体パーティ、合法手 = 技1 + 交代1。
  - `switch_brings_bench_in_before_move_hits`: 交代が相手の技より先に解決し、新しい場の
    ポケに技が当たる。
  - `faint_requests_forced_switch_then_resolves`: 瀕死 → `forced_switch` 要求 →
    `apply_forced_switches` で控えを出す。
  - `battle_lost_only_when_whole_party_faints`: パーティ全滅でのみ敗北判定。
- `cargo clippy -p poke-sho-rust`: 追加した turn.rs はクリーン (collapsible-if を let-chain に修正)。
  team.rs:57 の既存警告は本セッション対象外につき未着手。
- 全ソース非テスト 300 行以内 (battle.rs=117, turn.rs=221, party.rs=108, scenario.rs=270)。

## 残課題 (次増分: env / model 層)

1. **poke-env-rust**:
   - 観測 (`StateForPlayer` / `observation_for`) に控えの種族・HP・場の index を追加
     (Showdown 準拠で相手控えは隠匿)。
   - `local_showdown` の request / force-switch 対応 (`|switch|`・force-switch リクエスト)。
   - `lookahead` の rollout を交代手対応に (現状は技のみ抽出で素通し)。交代を含む
     合法手・方策サンプリングへ拡張。
2. **poke-ai3 / poke-ai3-python**:
   - 行動空間を `NUM_MOVES + (MAX_PARTY-1)` に拡張 (`MAX_MOVE_LEN` 同様に定数参照で
     ドリフト防止)。観測エンコーディングに控えスロットを追加。
   - `policy_head` の出力次元・fallback policy を交代込みに。
3. **学習・評価**: 両陣営ランダム先発で対戦させ、不利対面での適正交代率 >90% を定量判定。
   交代の質はトラジェクトリ定性レビュー (battle-review) でも確認。
4. **Showdown slot 整合**: Stage3b は各個体 1 技だが `MoveId::showdown_slot` は index 由来
   (Shock Wave=3, Bulldoze=4)。Showdown の request 技リストは実際の覚え技順 (slot 1) なので、
   env の parity 経路で slot 対応を見直す必要あり (次増分で扱う)。

## 次に着手する人へ

simulator 層の交代ルールはユニットテストで保証済み。次は poke-env-rust の観測・request
対応から進めるのが自然。`MAX_PARTY` / `NUM_MOVES` は定数 1 か所で、3v3 や技数変更時に
ドリフトしない設計になっている。
