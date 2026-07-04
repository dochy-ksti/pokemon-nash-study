//! エージェントの観測まわりの共有型と、真の `BattleState` からの観測構築。
//!
//! Stage3b 以降は交代を伴う。行動空間は技スロット (`MAX_MOVE_SLOTS`) + 控えへの交代
//! (`MAX_PARTY - 1`) の固定長 `ACTION_DIM`。技スロット数は技の**種類数** (`NUM_MOVES`)
//! とは独立で、ポケモンの技上限 (=4) に固定する。これにより新しい技種を追加しても
//! 行動次元・トークン数 (= checkpoint レイアウト) が変わらない。交代手は **相対控え
//! index** (active を除いたパーティ位置順) で表現し、相対↔絶対パーティ index の変換は
//! 本ファイルのヘルパに一箇所へ閉じる (lookahead / game_task が共用)。観測には自軍の控え情報
//! (`BenchSlot`) を固定長で持たせる。相手側も active の技と控え (種族・量子化 HP・技)
//! を神視点で観測に入れる (フォーマットは revealed-only と同型: `None` = 空き枠、
//! 将来は「未公開」へ意味を広げるだけで Showdown 忠実化できる)。

use serde::{Deserialize, Serialize};

pub use poke_sho_rust::battle::{BattleState, Choice, Player};
pub use poke_sho_rust::event::Event;
pub use poke_sho_rust::global_ids::{move_meta, species_meta};
pub use poke_sho_rust::party::{MAX_PARTY, Party, PokemonState};
pub use poke_sho_rust::scenario::{MoveId, NUM_MOVES, SpeciesId, Stage, TeamId};

/// 行動の技スロット数 (ポケモンが同時に持てる技の上限)。技の**種類数** `NUM_MOVES`
/// とは独立。技スロットトークン数・行動の技領域幅・encoded の move 次元はすべてこの
/// 定数で確定する。新しい技種を `MoveId` / move TSV に追加してもここは変えない。
pub const MAX_MOVE_SLOTS: usize = 4;

/// 行動空間の次元。技スロット + 控えへの交代枠 (active を除く最大控え数)。
pub const ACTION_DIM: usize = MAX_MOVE_SLOTS + (MAX_PARTY - 1);

/// 観測に含める控え枠数 (active を除く)。
pub const NUM_BENCH: usize = MAX_PARTY - 1;

/// 控え 1 体分の観測。空き枠は `Option::None` で表す (present/alive フラグは持たない:
/// 瀕死は `hp_frac == 0` と恒等で、交代の合法性は legal mask が担う)。
/// `species_gid`/`move_gids` はグローバル ID (`global_ids`)。`move_gids` はスロット順。
#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub struct BenchSlot {
    pub species_gid: u16,
    pub hp_frac: f32,
    pub move_gids: Vec<u16>,
}

/// 1 プレイヤー視点の観測。種族・技はグローバル ID (`global_ids`) で表す。
///
/// HP は割合で持つ。自分の HP は実数 HP から正確に求めるが、相手の HP は
/// Showdown が相手に見せる整数パーセント (`opp_quantized_hp_frac` は `percent / 100`) しか
/// 観測できない。`my_move_gids` は active の習得技 (スロット順) で、行動 index の
/// 技スロットと 1:1 に対応する。`legal_action_mask` は長さ `ACTION_DIM` で、
/// 技はスロット相対・交代枠は相対控え index。`my_bench` は自軍の控え。
/// 相手側は神視点: `opp_move_gids` は相手 active の真の習得技、`opp_bench` は
/// 相手の控え (HP は `opp_quantized_hp_frac` と同じ量子化)。
#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub struct StateForPlayer {
    /// 自分の active のグローバル種族 ID。
    pub my_species_gid: u16,
    /// 相手の active のグローバル種族 ID。
    pub opp_species_gid: u16,
    pub my_exact_hp_frac: f32,
    pub opp_quantized_hp_frac: f32,
    /// 自分の active の習得技 (グローバル ID、スロット順)。
    pub my_move_gids: Vec<u16>,
    /// 相手の active の習得技 (グローバル ID、スロット順)。神視点で真の技を入れる。
    pub opp_move_gids: Vec<u16>,
    /// 自軍の控え (相対控え index 順、長さ NUM_BENCH、空き枠は None)。
    pub my_bench: Vec<Option<BenchSlot>>,
    /// 相手の控え (相手 active 除く相対 index 順、長さ NUM_BENCH、空き枠は None)。
    /// HP は量子化、瀕死は hp_frac == 0。
    pub opp_bench: Vec<Option<BenchSlot>>,
    /// 長さ `ACTION_DIM` の合法手マスク。
    pub legal_action_mask: Vec<bool>,
}

/// 種族のグローバル ID。名前 (フォルム込み正式表記) で ID 表を引く。
pub fn species_gid_of(mon: &PokemonState) -> u16 {
    species_meta(mon.name)
        .unwrap_or_else(|| panic!("species '{}' not in global id table", mon.name))
        .id
}

/// 技のグローバル ID。
pub fn move_gid_of(mv: MoveId) -> u16 {
    let name = mv.data().name;
    move_meta(name)
        .unwrap_or_else(|| panic!("move '{name}' not in global id table"))
        .id
}

/// 習得技マスクをスロット順のグローバル ID 列へ。
pub fn move_gids_of(moves: &[bool; NUM_MOVES]) -> Vec<u16> {
    MoveId::ALL
        .into_iter()
        .filter(|m| moves[m.index()])
        .map(move_gid_of)
        .collect()
}

/// 相対控え index → 絶対パーティ index。active を 1 つ飛ばすだけ。
pub fn bench_rel_to_abs(active: usize, rel: usize) -> usize {
    if rel < active { rel } else { rel + 1 }
}

/// 絶対パーティ index → 相対控え index。active より後ろは 1 つ詰める。
pub fn bench_abs_to_rel(active: usize, abs: usize) -> usize {
    if abs < active { abs } else { abs - 1 }
}

/// 習得技マスクからスロット `slot` 番目 (習得順 = `MoveId::ALL` 順) の技を返す。
pub fn move_at_slot(moves: &[bool; NUM_MOVES], slot: usize) -> Option<MoveId> {
    MoveId::ALL
        .into_iter()
        .filter(|m| moves[m.index()])
        .nth(slot)
}

/// 技 `mv` が習得技マスクの何スロット目かを返す。未習得は `None`。
pub fn move_slot_of(moves: &[bool; NUM_MOVES], mv: MoveId) -> Option<usize> {
    if !moves[mv.index()] {
        return None;
    }
    Some(
        MoveId::ALL
            .into_iter()
            .take(mv.index())
            .filter(|m| moves[m.index()])
            .count(),
    )
}

/// `Choice` を行動 index (0..ACTION_DIM) へ。技は **active の習得技スロット相対**
/// (Showdown の `/choose move n` と同型)、交代は `MAX_MOVE_SLOTS +` 相対控え index。
pub fn action_index(party: &Party, choice: Choice) -> usize {
    match choice {
        Choice::Move(m) => {
            let slot = move_slot_of(&party.active_mon().moves, m)
                .expect("action_index: active does not know the move");
            debug_assert!(
                slot < MAX_MOVE_SLOTS,
                "move slot {slot} exceeds MAX_MOVE_SLOTS (a Pokemon may know at most {MAX_MOVE_SLOTS} moves)"
            );
            slot
        }
        Choice::Switch(abs) => MAX_MOVE_SLOTS + bench_abs_to_rel(party.active, abs),
    }
}

/// 行動 index (0..ACTION_DIM) を `Choice` へ。技はスロット相対から `MoveId` へ、
/// 交代は絶対パーティ index へ戻す。
pub fn action_to_choice(party: &Party, action: usize) -> Choice {
    if action < MAX_MOVE_SLOTS {
        Choice::Move(
            move_at_slot(&party.active_mon().moves, action)
                .expect("action_to_choice: empty move slot"),
        )
    } else {
        Choice::Switch(bench_rel_to_abs(party.active, action - MAX_MOVE_SLOTS))
    }
}

/// 真の `BattleState` から 1 プレイヤー視点の観測を作る。自分の HP は実数、
/// 相手の HP は Showdown が見せる整数パーセントへ量子化する (game_task と同じ)。
pub fn observation_for(state: &BattleState, player: Player) -> StateForPlayer {
    let party = state.party(player);
    let opp_party = state.party(player.opponent());
    let me = state.pokemon(player);
    let opp = state.pokemon(player.opponent());
    let my_hp_frac = if state.is_fainted(player) || me.max_hp <= 0 {
        0.0
    } else {
        exact_hp_frac(&me)
    };
    let opp_hp_frac =
        opp_quantized_hp_frac(opp.hp, opp.max_hp, state.is_fainted(player.opponent()));

    // 合法手マスク (ACTION_DIM)。技はスロット相対、交代は相対控え index へ写す。
    let mut legal = vec![false; ACTION_DIM];
    for c in state.legal_choices(player) {
        legal[action_index(party, c)] = true;
    }

    StateForPlayer {
        my_species_gid: species_gid_of(&me),
        opp_species_gid: species_gid_of(&opp),
        my_exact_hp_frac: my_hp_frac,
        opp_quantized_hp_frac: opp_hp_frac,
        my_move_gids: move_gids_of(&me.moves),
        opp_move_gids: move_gids_of(&opp.moves),
        // 自軍控えは実数 HP 比、相手控えは active と同じ量子化 HP (瀕死は 0)。
        my_bench: bench_slots(party, exact_hp_frac),
        opp_bench: bench_slots(opp_party, |mon| {
            opp_quantized_hp_frac(mon.hp, mon.max_hp, mon.hp <= 0)
        }),
        legal_action_mask: legal,
    }
}

/// 控えを相対 index 順に詰める (positional: active を除いた位置固定、空き枠は None)。
/// HP 割合の方針 (実数 / 量子化) は呼び出し側が `hp_frac_of` で注入する。
fn bench_slots(party: &Party, hp_frac_of: impl Fn(&PokemonState) -> f32) -> Vec<Option<BenchSlot>> {
    let mut bench: Vec<Option<BenchSlot>> = vec![None; NUM_BENCH];
    for (rel, slot) in bench.iter_mut().enumerate() {
        let abs = bench_rel_to_abs(party.active, rel);
        if abs >= party.len {
            continue;
        }
        let mon = &party.members[abs];
        *slot = Some(BenchSlot {
            species_gid: species_gid_of(mon),
            hp_frac: hp_frac_of(mon),
            move_gids: move_gids_of(&mon.moves),
        });
    }
    bench
}

/// 実数 HP 比 (自軍用)。max_hp 不明 (0 以下) は 0 扱い。
fn exact_hp_frac(mon: &PokemonState) -> f32 {
    if mon.max_hp <= 0 {
        0.0
    } else {
        (mon.hp as f32 / mon.max_hp as f32).clamp(0.0, 1.0)
    }
}

fn opp_quantized_hp_frac(hp: i32, max_hp: i32, fainted: bool) -> f32 {
    if fainted || max_hp <= 0 || hp <= 0 {
        return 0.0;
    }
    let pct = (100 * hp + max_hp - 1) / max_hp;
    let pct = if pct >= 100 && hp < max_hp { 99 } else { pct };
    pct as f32 / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use poke_sho_rust::scenario::{Stage, TeamId};

    #[test]
    fn move_slots_are_relative_to_known_moves() {
        // Bulldoze しか知らない個体ではスロット 0 = Bulldoze。
        let mut moves = [false; NUM_MOVES];
        moves[MoveId::Bulldoze.index()] = true;
        assert_eq!(move_at_slot(&moves, 0), Some(MoveId::Bulldoze));
        assert_eq!(move_at_slot(&moves, 1), None);
        assert_eq!(move_slot_of(&moves, MoveId::Bulldoze), Some(0));
        assert_eq!(move_slot_of(&moves, MoveId::Crunch), None);

        // DarkPulse と Bulldoze を知る個体ではスロット 0=DarkPulse, 1=Bulldoze。
        let mut moves = [false; NUM_MOVES];
        moves[MoveId::DarkPulse.index()] = true;
        moves[MoveId::Bulldoze.index()] = true;
        assert_eq!(move_at_slot(&moves, 0), Some(MoveId::DarkPulse));
        assert_eq!(move_at_slot(&moves, 1), Some(MoveId::Bulldoze));
        assert_eq!(move_slot_of(&moves, MoveId::Bulldoze), Some(1));
    }

    #[test]
    fn action_index_roundtrips_with_action_to_choice() {
        let state = BattleState::new_with_teams(Stage::Stage3b, (TeamId::Team2, 0), (TeamId::Team1, 0));
        let party = state.party(Player::P1);
        for c in state.legal_choices(Player::P1) {
            let idx = action_index(party, c);
            assert!(idx < ACTION_DIM);
            assert_eq!(action_to_choice(party, idx), c);
        }
        // Team2 Cloyster は Bulldoze のみ習得 → 技はスロット 0 だけが合法。
        let obs = observation_for(&state, Player::P1);
        assert!(obs.legal_action_mask[0]);
        assert!(!obs.legal_action_mask[1]);
        assert!(obs.legal_action_mask[MAX_MOVE_SLOTS]);
    }

    #[test]
    fn opponent_side_is_observed_god_view() {
        let state = BattleState::new_with_teams(Stage::Stage3b, (TeamId::Team2, 0), (TeamId::Team1, 0));
        let p1 = observation_for(&state, Player::P1);
        let p2 = observation_for(&state, Player::P2);

        // 相手 active の技は真の習得技 (相手視点の自分の技と一致する)。
        assert_eq!(p1.opp_move_gids, p2.my_move_gids);
        assert_eq!(p2.opp_move_gids, p1.my_move_gids);

        // 相手控えは種族・技が自分視点の my_bench と同一の並びで見える。
        assert_eq!(p1.opp_bench.len(), NUM_BENCH);
        for (mine, theirs) in p2.my_bench.iter().zip(p1.opp_bench.iter()) {
            match (mine, theirs) {
                (Some(m), Some(t)) => {
                    assert_eq!(m.species_gid, t.species_gid);
                    assert_eq!(m.move_gids, t.move_gids);
                    // 開始時は満タン: 実数比・量子化とも 1.0 で一致する。
                    assert_eq!(m.hp_frac, 1.0);
                    assert_eq!(t.hp_frac, 1.0);
                }
                (None, None) => {}
                _ => panic!("bench slot presence mismatch"),
            }
        }
    }
}
