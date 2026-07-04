//! パーティと場の個体の状態モデル。
//!
//! `BattleState` は各サイドに 1 つの [`Party`] を持つ。1v1 ステージは
//! メンバー 1 体・交代不可の退化ケースとして同じ型で表現し、交代を伴う
//! ステージ (Stage3b 以降) は複数メンバー + 場の index で表現する。
//! 戦闘の進行ロジックそのものは `battle` モジュールにある。

use crate::scenario::{MoveId, NUM_MOVES, SpeciesId};
use crate::species::Stats;
use crate::types::TypeSet;

/// パーティの最大体数 (レイアウト容量)。Showdown singles の 6 体に固定し、観測・
/// 行動次元・トークン数をこの上限で確定させる。現在のシナリオ (1v1/2v2) は `len` が
/// 小さい退化ケースとして走り、残り枠は present=0 でパディングされる。これにより
/// パーティ数を増やしても checkpoint レイアウトが変わらない。
pub const MAX_PARTY: usize = 6;

/// Current state of one Pokemon, including its real stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PokemonState {
    pub species_id: SpeciesId,
    pub hp: i32,
    pub max_hp: i32,
    pub stats: Stats,
    pub types: TypeSet,
    pub level: u16,
    /// この個体が覚えている技 (MoveId の index でアクセスする固定長マスク)。
    /// 合法手はこのマスクで決まり、シナリオごとに技セットが変わっても枠は共通。
    pub moves: [bool; NUM_MOVES],
    /// 正式名 (フォルム含む)。switch DETAILS など species 表記に使う。
    pub name: &'static str,
    /// イベント ident に出る表示名。Showdown はニックネーム無しのフォルム個体を
    /// ベース種族名で識別するため `name` と別に持つ (例 Goodra-Hisui → "Goodra")。
    pub display_name: &'static str,
}

impl PokemonState {
    pub fn is_fainted(&self) -> bool {
        self.hp <= 0
    }

    pub(crate) fn from_resolved(
        species_id: SpeciesId,
        resolved: &crate::team::ResolvedSet,
    ) -> Self {
        let max_hp = resolved.stats.hp as i32;
        let mut moves = [false; NUM_MOVES];
        for mv in &resolved.moves {
            if let Some(id) = MoveId::from_showdown_id(mv.id) {
                moves[id.index()] = true;
            }
        }
        PokemonState {
            species_id,
            hp: max_hp,
            max_hp,
            stats: resolved.stats,
            types: resolved.species.types,
            level: resolved.level,
            moves,
            name: resolved.species.name,
            display_name: resolved.species.display_name,
        }
    }
}

/// 片側のパーティ。`members[0..len]` が実体で、`active` が場に出ている index。
/// 1v1 ステージは `len == 1`・`active == 0`・交代手なしの退化ケース。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Party {
    pub members: [PokemonState; MAX_PARTY],
    pub len: usize,
    pub active: usize,
}

impl Party {
    /// 1 体だけのパーティ (1v1 ステージ用)。空きスロットは先頭の複製で埋める。
    pub fn single(mon: PokemonState) -> Self {
        Party {
            members: [mon; MAX_PARTY],
            len: 1,
            active: 0,
        }
    }

    /// 場に出ている個体。
    pub fn active_mon(&self) -> &PokemonState {
        &self.members[self.active]
    }

    pub fn active_mon_mut(&mut self) -> &mut PokemonState {
        &mut self.members[self.active]
    }

    /// `idx` が生存メンバーか (範囲内かつ HP > 0)。
    pub fn is_living(&self, idx: usize) -> bool {
        idx < self.len && self.members[idx].hp > 0
    }

    /// 場に出ていない生存メンバーが 1 体でもいるか (交代可能か)。
    pub fn has_living_bench(&self) -> bool {
        (0..self.len).any(|i| i != self.active && self.members[i].hp > 0)
    }

    /// 全メンバーが瀕死か (このサイドの敗北条件)。
    pub fn all_fainted(&self) -> bool {
        (0..self.len).all(|i| self.members[i].hp <= 0)
    }

    /// 交代先になり得るパーティ index (場以外の生存メンバー)。
    pub fn switch_targets(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| i != self.active && self.members[i].hp > 0)
    }
}
