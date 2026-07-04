//! Move data for the early phases.

use crate::types::PokeType;
use serde::{Deserialize, Serialize};

/// Damage class of a move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum MoveCategory {
    Physical,
    Special,
    Status,
}

/// Static data describing one move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveData {
    /// Showdown move name (display form).
    pub name: &'static str,
    /// Showdown move id (lowercase, no spaces).
    pub id: &'static str,
    pub move_type: PokeType,
    pub category: MoveCategory,
    pub power: u16,
    /// Accuracy percent; 100 for the always-hit early-phase moves.
    pub accuracy: u16,
}

/// Crunch: Dark physical, power 80. 現在のシナリオで物理側として使用。
/// Showdown では 20% で Def↓の追加効果があるが、おんみつマントで無効化される。
/// 決定論モードでは追加効果は未実装 (BattleRng は damage roll と急所のみ供給)。
pub const CRUNCH: MoveData = MoveData {
    name: "Crunch",
    id: "crunch",
    move_type: PokeType::Dark,
    category: MoveCategory::Physical,
    power: 80,
    accuracy: 100,
};

/// Dark Pulse: Dark special, power 80. 現在のシナリオで特殊側として使用。
/// Showdown では 20% でひるみの追加効果があるが、おんみつマントで無効化される。
pub const DARK_PULSE: MoveData = MoveData {
    name: "Dark Pulse",
    id: "darkpulse",
    move_type: PokeType::Dark,
    category: MoveCategory::Special,
    power: 80,
    accuracy: 100,
};

/// Shock Wave: Electric special, power 60. Cloyster (Water/Ice) を弱点 (でんき 2倍)
/// かつ低 SpD として突くための技。Showdown では必中だが、決定論モードでは accuracy=100
/// 扱い。でんきタイプの個体が居ないため STAB は常に乗らない。
pub const SHOCK_WAVE: MoveData = MoveData {
    name: "Shock Wave",
    id: "shockwave",
    move_type: PokeType::Electric,
    category: MoveCategory::Special,
    power: 60,
    accuracy: 100,
};

/// Bulldoze: Ground physical, power 60. ヒスイヌメルゴン (Steel/Dragon) を弱点
/// (じめん 2倍) かつ低 Def として突くための技。Showdown では相手の素早さを下げる
/// 追加効果があるが、おんみつマントで無効化され、決定論モードでは未実装。じめんタイプの
/// 個体が居ないため STAB は常に乗らない。
pub const BULLDOZE: MoveData = MoveData {
    name: "Bulldoze",
    id: "bulldoze",
    move_type: PokeType::Ground,
    category: MoveCategory::Physical,
    power: 60,
    accuracy: 100,
};

/// FightSpe60: Fighting 特殊, 威力 60。stage3c の対称対面用合成技。Cloyster (Water/Ice)
/// を弱点 (かくとう→こおり 2倍) かつ低 SpD として突く。STAB の乗る個体が居ないため
/// 常に不一致。追加効果なし。命中 100 (決定論モードで accuracy=100 扱い)。
pub const FIGHT_SPE_60: MoveData = MoveData {
    name: "FightSpe60",
    id: "fightspe60",
    move_type: PokeType::Fighting,
    category: MoveCategory::Special,
    power: 60,
    accuracy: 100,
};

/// FairyPhy60: Fairy 物理, 威力 60。stage3c の対称対面用合成技。原種 Goodra (Dragon)
/// を弱点 (フェアリー→ドラゴン 2倍) かつ低 Def として突く。FightSpe60 と物理/特殊・
/// 弱点対象が鏡像になるよう設計。STAB なし・追加効果なし。
pub const FAIRY_PHY_60: MoveData = MoveData {
    name: "FairyPhy60",
    id: "fairyphy60",
    move_type: PokeType::Fairy,
    category: MoveCategory::Physical,
    power: 60,
    accuracy: 100,
};

/// Looks up a move by its Showdown id or display name (case-insensitive).
pub fn move_by_name(name: &str) -> Option<MoveData> {
    match name.trim().to_ascii_lowercase().replace([' ', '-'], "").as_str() {
        "crunch" => Some(CRUNCH),
        "darkpulse" => Some(DARK_PULSE),
        "shockwave" => Some(SHOCK_WAVE),
        "bulldoze" => Some(BULLDOZE),
        "fightspe60" => Some(FIGHT_SPE_60),
        "fairyphy60" => Some(FAIRY_PHY_60),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_by_name_and_id() {
        assert_eq!(move_by_name("Crunch"), Some(CRUNCH));
        assert_eq!(move_by_name("Dark Pulse"), Some(DARK_PULSE));
        assert_eq!(move_by_name("darkpulse"), Some(DARK_PULSE));
        assert_eq!(move_by_name("Shock Wave"), Some(SHOCK_WAVE));
        assert_eq!(move_by_name("bulldoze"), Some(BULLDOZE));
        assert_eq!(move_by_name("FightSpe60"), Some(FIGHT_SPE_60));
        assert_eq!(move_by_name("fairyphy60"), Some(FAIRY_PHY_60));
        assert_eq!(move_by_name("Hyper Beam"), None);
    }
}
