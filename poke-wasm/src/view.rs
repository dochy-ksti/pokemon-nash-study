//! JS へ渡す表示用のシリアライズ可能ビュー。真の `BattleState` から構築する。
//! ブラウザはこのビューを描画し、team/active/HP からテーブル index を自前計算する。

use poke_sho_rust::battle::{BattleState, Player};
use poke_sho_rust::moves::MoveCategory;
use poke_sho_rust::party::{Party, PokemonState};
use poke_sho_rust::scenario::MoveId;
use poke_sho_rust::types::PokeType;
use serde::Serialize;

const ALL_TYPES: [(PokeType, &str); 18] = [
    (PokeType::Normal, "Normal"), (PokeType::Fighting, "Fighting"),
    (PokeType::Flying, "Flying"), (PokeType::Poison, "Poison"),
    (PokeType::Ground, "Ground"), (PokeType::Rock, "Rock"),
    (PokeType::Bug, "Bug"), (PokeType::Ghost, "Ghost"),
    (PokeType::Steel, "Steel"), (PokeType::Fire, "Fire"),
    (PokeType::Water, "Water"), (PokeType::Grass, "Grass"),
    (PokeType::Electric, "Electric"), (PokeType::Psychic, "Psychic"),
    (PokeType::Ice, "Ice"), (PokeType::Dragon, "Dragon"),
    (PokeType::Dark, "Dark"), (PokeType::Fairy, "Fairy"),
];

pub fn poke_type_name(t: PokeType) -> &'static str {
    ALL_TYPES.iter().find(|(p, _)| *p == t).map(|(_, n)| *n).unwrap_or("?")
}

fn category_name(c: MoveCategory) -> &'static str {
    match c {
        MoveCategory::Physical => "Physical",
        MoveCategory::Special => "Special",
        MoveCategory::Status => "Status",
    }
}

#[derive(Serialize)]
pub struct StatsView {
    pub hp: u16, pub atk: u16, pub def: u16,
    pub spa: u16, pub spd: u16, pub spe: u16,
}

#[derive(Serialize)]
pub struct MoveView {
    /// Showdown 正規化 id (でんげきは等の locale 引き / 内部キー)。
    pub id: String,
    pub name: String,
    /// スロット index (`MoveId::index`)。ダメージ計算・行動 index の基準。
    pub slot: u8,
    pub move_type: String,
    pub power: u16,
    pub category: String,
}

#[derive(Serialize)]
pub struct MonView {
    /// 種族名 (Showdown 表記。locale 引きの内部キー)。
    pub species: String,
    /// 種族 enum index (0=Cloyster,1=Goodra-Hisui,2=Goodra)。
    pub species_idx: u8,
    pub types: Vec<String>,
    pub stats: StatsView,
    pub hp: i32,
    pub max_hp: i32,
    pub moves: Vec<MoveView>,
}

#[derive(Serialize)]
pub struct SideView {
    pub active: u8,
    pub members: Vec<MonView>,
}

#[derive(Serialize)]
pub struct StateView {
    pub p1: SideView,
    pub p2: SideView,
    pub turn: u32,
    pub done: bool,
    /// 0=P1 勝ち, 1=P2 勝ち, null=未決 or 引き分け。
    pub winner: Option<u8>,
    pub forced_switch: [bool; 2],
}

fn mon_view(mon: &PokemonState) -> MonView {
    let types = ALL_TYPES
        .iter()
        .filter(|(t, _)| mon.types.contains(*t))
        .map(|(_, n)| n.to_string())
        .collect();
    let moves = MoveId::ALL
        .into_iter()
        .filter(|m| mon.moves[m.index()])
        .map(|m| {
            let d = m.data();
            MoveView {
                id: d.id.to_string(),
                name: d.name.to_string(),
                slot: m.index() as u8,
                move_type: poke_type_name(d.move_type).to_string(),
                power: d.power,
                category: category_name(d.category).to_string(),
            }
        })
        .collect();
    MonView {
        species: mon.species_id.name().to_string(),
        species_idx: mon.species_id.index() as u8,
        types,
        stats: StatsView {
            hp: mon.stats.hp, atk: mon.stats.atk, def: mon.stats.def,
            spa: mon.stats.spa, spd: mon.stats.spd, spe: mon.stats.spe,
        },
        hp: mon.hp,
        max_hp: mon.max_hp,
        moves,
    }
}

fn side_view(party: &Party) -> SideView {
    SideView {
        active: party.active as u8,
        members: (0..party.len).map(|i| mon_view(&party.members[i])).collect(),
    }
}

pub fn state_view(state: &BattleState) -> StateView {
    let winner = state.winner().map(|p| p.index() as u8);
    StateView {
        p1: side_view(state.party(Player::P1)),
        p2: side_view(state.party(Player::P2)),
        turn: state.turn,
        done: state.is_done(),
        winner,
        forced_switch: state.forced_switch,
    }
}
