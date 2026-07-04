//! Species base stats, natures, and real-stat computation.
//!
//! Real stats follow the standard Generation 3+ formulas used by Pokemon
//! Showdown, so a parsed Showdown set produces the same in-game numbers.

use crate::types::{PokeType, TypeSet};
use serde::{Deserialize, Serialize};

/// Showdown default IV when a set omits IVs.
pub const DEFAULT_IV: u16 = 31;
/// Showdown default EV when a set omits EVs.
pub const DEFAULT_EV: u16 = 0;

/// One value per stat. Used for base stats, EVs, IVs, and computed real stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct Stats {
    pub hp: u16,
    pub atk: u16,
    pub def: u16,
    pub spa: u16,
    pub spd: u16,
    pub spe: u16,
}

impl Stats {
    pub const fn uniform(value: u16) -> Self {
        Self {
            hp: value,
            atk: value,
            def: value,
            spa: value,
            spd: value,
            spe: value,
        }
    }

    pub const ZERO: Stats = Stats::uniform(0);
    pub const DEFAULT_IVS: Stats = Stats::uniform(DEFAULT_IV);
}

/// Identifies a single stat. HP is never affected by nature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatKind {
    Hp,
    Atk,
    Def,
    Spa,
    Spd,
    Spe,
}

/// A Pokemon nature. Each non-neutral nature raises one stat 10% and lowers
/// another 10%; the five neutral natures leave every stat unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Nature {
    Hardy,
    Lonely,
    Brave,
    Adamant,
    Naughty,
    Bold,
    Docile,
    Relaxed,
    Impish,
    Lax,
    Timid,
    Hasty,
    Serious,
    Jolly,
    Naive,
    Modest,
    Mild,
    Quiet,
    Bashful,
    Rash,
    Calm,
    Gentle,
    Sassy,
    Careful,
    Quirky,
}

impl Nature {
    /// Parses a Showdown nature name (e.g. "Serious"). Case-insensitive.
    pub fn from_name(name: &str) -> Option<Self> {
        use Nature::*;
        let n = match name.trim().to_ascii_lowercase().as_str() {
            "hardy" => Hardy,
            "lonely" => Lonely,
            "brave" => Brave,
            "adamant" => Adamant,
            "naughty" => Naughty,
            "bold" => Bold,
            "docile" => Docile,
            "relaxed" => Relaxed,
            "impish" => Impish,
            "lax" => Lax,
            "timid" => Timid,
            "hasty" => Hasty,
            "serious" => Serious,
            "jolly" => Jolly,
            "naive" => Naive,
            "modest" => Modest,
            "mild" => Mild,
            "quiet" => Quiet,
            "bashful" => Bashful,
            "rash" => Rash,
            "calm" => Calm,
            "gentle" => Gentle,
            "sassy" => Sassy,
            "careful" => Careful,
            "quirky" => Quirky,
            _ => return None,
        };
        Some(n)
    }

    /// The (raised, lowered) stats, or `None` for a neutral nature.
    fn modifiers(self) -> Option<(StatKind, StatKind)> {
        use Nature::*;
        use StatKind::*;
        let pair = match self {
            Hardy | Docile | Serious | Bashful | Quirky => return None,
            Lonely => (Atk, Def),
            Brave => (Atk, Spe),
            Adamant => (Atk, Spa),
            Naughty => (Atk, Spd),
            Bold => (Def, Atk),
            Relaxed => (Def, Spe),
            Impish => (Def, Spa),
            Lax => (Def, Spd),
            Timid => (Spe, Atk),
            Hasty => (Spe, Def),
            Jolly => (Spe, Spa),
            Naive => (Spe, Spd),
            Modest => (Spa, Atk),
            Mild => (Spa, Def),
            Quiet => (Spa, Spe),
            Rash => (Spa, Spd),
            Calm => (Spd, Atk),
            Gentle => (Spd, Def),
            Sassy => (Spd, Spe),
            Careful => (Spd, Spa),
        };
        Some(pair)
    }

    /// Nature multiplier for a stat as a (numerator, denominator) pair so the
    /// computation stays integer-only and matches Showdown's flooring.
    fn multiplier(self, stat: StatKind) -> (u32, u32) {
        match self.modifiers() {
            Some((up, _)) if up == stat => (11, 10),
            Some((_, down)) if down == stat => (9, 10),
            _ => (1, 1),
        }
    }
}

/// A Pokemon species: name, base stats, and type pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Species {
    /// フォルムまで含む正式名 (Showdown の switch DETAILS と一致。例 "Goodra-Hisui")。
    pub name: &'static str,
    /// イベント ident に出る表示名。Showdown はニックネーム無しのフォルム個体を
    /// ベース種族名で識別するため (例 Goodra-Hisui → "Goodra")、`name` と分ける。
    pub display_name: &'static str,
    pub base: Stats,
    pub types: TypeSet,
}

/// Cloyster: シナリオの物理耐久側。Water/Ice、base 50/95/180/85/45/70。
pub const CLOYSTER: Species = Species {
    name: "Cloyster",
    display_name: "Cloyster",
    base: Stats {
        hp: 50,
        atk: 95,
        def: 180,
        spa: 85,
        spd: 45,
        spe: 70,
    },
    types: TypeSet::dual(PokeType::Water, PokeType::Ice),
};

/// Goodra (原種): シナリオの特殊耐久側。Dragon 単、base 90/100/70/110/150/80。
pub const GOODRA: Species = Species {
    name: "Goodra",
    display_name: "Goodra",
    base: Stats {
        hp: 90,
        atk: 100,
        def: 70,
        spa: 110,
        spd: 150,
        spe: 80,
    },
    types: TypeSet::mono(PokeType::Dragon),
};

/// Goodra-Hisui (ヒスイのすがた): タイプ相性シナリオの特殊耐久側。Steel/Dragon、
/// base 80/100/100/110/150/60。じめん技 (Bulldoze) で弱点 (はがね 2倍) を突かれる。
pub const GOODRA_HISUI: Species = Species {
    name: "Goodra-Hisui",
    display_name: "Goodra",
    base: Stats {
        hp: 80,
        atk: 100,
        def: 100,
        spa: 110,
        spd: 150,
        spe: 60,
    },
    types: TypeSet::dual(PokeType::Steel, PokeType::Dragon),
};

/// Looks up a species by its Showdown name (case-insensitive, ignoring spaces
/// and dashes). Only the species used by the current phases are registered.
pub fn species_by_name(name: &str) -> Option<Species> {
    match name.trim().to_ascii_lowercase().replace([' ', '-'], "").as_str() {
        "cloyster" => Some(CLOYSTER),
        "goodra" => Some(GOODRA),
        "goodrahisui" => Some(GOODRA_HISUI),
        _ => None,
    }
}

fn compute_hp(base: u16, iv: u16, ev: u16, level: u16) -> u16 {
    let base = base as u32;
    let iv = iv as u32;
    let ev = ev as u32;
    let level = level as u32;
    (((2 * base + iv + ev / 4) * level) / 100 + level + 10) as u16
}

fn compute_other(base: u16, iv: u16, ev: u16, level: u16, nature: (u32, u32)) -> u16 {
    let base = base as u32;
    let iv = iv as u32;
    let ev = ev as u32;
    let level = level as u32;
    let pre = ((2 * base + iv + ev / 4) * level) / 100 + 5;
    (pre * nature.0 / nature.1) as u16
}

/// Computes a Pokemon's real stats from base stats, IVs, EVs, level, and nature.
pub fn compute_stats(base: Stats, ivs: Stats, evs: Stats, level: u16, nature: Nature) -> Stats {
    use StatKind::*;
    Stats {
        hp: compute_hp(base.hp, ivs.hp, evs.hp, level),
        atk: compute_other(base.atk, ivs.atk, evs.atk, level, nature.multiplier(Atk)),
        def: compute_other(base.def, ivs.def, evs.def, level, nature.multiplier(Def)),
        spa: compute_other(base.spa, ivs.spa, evs.spa, level, nature.multiplier(Spa)),
        spd: compute_other(base.spd, ivs.spd, evs.spd, level, nature.multiplier(Spd)),
        spe: compute_other(base.spe, ivs.spe, evs.spe, level, nature.multiplier(Spe)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloyster_level50_serious_default_spread() {
        let stats = compute_stats(
            CLOYSTER.base,
            Stats::DEFAULT_IVS,
            Stats::ZERO,
            50,
            Nature::Serious,
        );
        // 50/95/180/85/45/70 base → Lv50 neutral, IV31 EV0.
        assert_eq!(stats.hp, 125);
        assert_eq!(stats.atk, 115);
        assert_eq!(stats.def, 200);
        assert_eq!(stats.spa, 105);
        assert_eq!(stats.spd, 65);
        assert_eq!(stats.spe, 90);
    }

    #[test]
    fn nature_raises_and_lowers_by_ten_percent() {
        // Relaxed: +Def, -Spe (Cloyster に使う性格).
        let stats = compute_stats(
            CLOYSTER.base,
            Stats::DEFAULT_IVS,
            Stats::ZERO,
            50,
            Nature::Relaxed,
        );
        assert_eq!(stats.def, 220); // floor(200 * 11 / 10)
        assert_eq!(stats.spe, 81); // floor(90 * 9 / 10)
        assert_eq!(stats.atk, 115); // unaffected
    }

    #[test]
    fn nature_name_parsing() {
        assert_eq!(Nature::from_name("Serious"), Some(Nature::Serious));
        assert_eq!(Nature::from_name("adamant"), Some(Nature::Adamant));
        assert_eq!(Nature::from_name("Nonsense"), None);
    }
}
