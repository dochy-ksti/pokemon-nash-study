//! Pokemon types and type effectiveness.
//!
//! Encodes the full Gen 6+ (current) 18-type chart. Each attacking row lists
//! the defending types it is super effective against, those that resist it, and
//! those immune to it; everything else is neutral.

use serde::{Deserialize, Serialize};

/// Pokemon elemental type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum PokeType {
    Normal,
    Fighting,
    Flying,
    Poison,
    Ground,
    Rock,
    Bug,
    Ghost,
    Steel,
    Fire,
    Water,
    Grass,
    Electric,
    Psychic,
    Ice,
    Dragon,
    Dark,
    Fairy,
}

/// A Pokemon's type pairing (one or two types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct TypeSet {
    primary: PokeType,
    secondary: Option<PokeType>,
}

impl TypeSet {
    pub const fn mono(t: PokeType) -> Self {
        Self {
            primary: t,
            secondary: None,
        }
    }

    pub const fn dual(a: PokeType, b: PokeType) -> Self {
        Self {
            primary: a,
            secondary: Some(b),
        }
    }

    /// Whether the set contains `t` (used for STAB).
    pub fn contains(&self, t: PokeType) -> bool {
        self.primary == t || self.secondary == Some(t)
    }

    /// Combined type effectiveness of `attacking` against this defender,
    /// multiplying each defending type's factor.
    pub fn effectiveness(&self, attacking: PokeType) -> f64 {
        let mut factor = effectiveness(attacking, self.primary);
        if let Some(second) = self.secondary {
            factor *= effectiveness(attacking, second);
        }
        factor
    }

    /// Combined type modifier as Showdown's integer `typeMod`: the number of
    /// effectiveness steps (each `+1` doubles, each `-1` halves). Returns
    /// `None` if any defending type is immune to `attacking`.
    pub fn type_mod(&self, attacking: PokeType) -> Option<i32> {
        let mut steps = type_relation(attacking, self.primary)?;
        if let Some(second) = self.secondary {
            steps += type_relation(attacking, second)?;
        }
        Some(steps)
    }
}

/// Single-type effectiveness of `attacking` against `defending`.
///
/// Returns 2.0 (super effective), 1.0 (neutral), 0.5 (resisted), or 0.0
/// (immune), following the full Gen 6+ type chart.
pub fn effectiveness(attacking: PokeType, defending: PokeType) -> f64 {
    match type_relation(attacking, defending) {
        None => 0.0,
        Some(steps) => 2f64.powi(steps),
    }
}

/// Single-type effectiveness as a step count (Showdown's `typeMod` contribution):
/// `Some(1)` super effective, `Some(0)` neutral, `Some(-1)` resisted, `None`
/// immune. Encodes the complete Gen 6+ (current) type chart.
fn type_relation(attacking: PokeType, defending: PokeType) -> Option<i32> {
    use PokeType::*;
    // Per attacking row: list the defending types it is super effective (+1)
    // against, the ones that resist it (-1), and the ones immune (None). Every
    // other defender is neutral (0).
    match attacking {
        Normal => match defending {
            Rock | Steel => Some(-1),
            Ghost => None,
            _ => Some(0),
        },
        Fighting => match defending {
            Normal | Rock | Steel | Ice | Dark => Some(1),
            Flying | Poison | Bug | Psychic | Fairy => Some(-1),
            Ghost => None,
            _ => Some(0),
        },
        Flying => match defending {
            Fighting | Bug | Grass => Some(1),
            Rock | Steel | Electric => Some(-1),
            _ => Some(0),
        },
        Poison => match defending {
            Grass | Fairy => Some(1),
            Poison | Ground | Rock | Ghost => Some(-1),
            Steel => None,
            _ => Some(0),
        },
        Ground => match defending {
            Poison | Rock | Steel | Fire | Electric => Some(1),
            Bug | Grass => Some(-1),
            Flying => None,
            _ => Some(0),
        },
        Rock => match defending {
            Flying | Bug | Fire | Ice => Some(1),
            Fighting | Ground | Steel => Some(-1),
            _ => Some(0),
        },
        Bug => match defending {
            Grass | Psychic | Dark => Some(1),
            Fighting | Flying | Poison | Ghost | Steel | Fire | Fairy => Some(-1),
            _ => Some(0),
        },
        Ghost => match defending {
            Ghost | Psychic => Some(1),
            Dark => Some(-1),
            Normal => None,
            _ => Some(0),
        },
        Steel => match defending {
            Rock | Ice | Fairy => Some(1),
            Steel | Fire | Water | Electric => Some(-1),
            _ => Some(0),
        },
        Fire => match defending {
            Bug | Steel | Grass | Ice => Some(1),
            Rock | Fire | Water | Dragon => Some(-1),
            _ => Some(0),
        },
        Water => match defending {
            Ground | Rock | Fire => Some(1),
            Water | Grass | Dragon => Some(-1),
            _ => Some(0),
        },
        Grass => match defending {
            Ground | Rock | Water => Some(1),
            Flying | Poison | Bug | Steel | Fire | Grass | Dragon => Some(-1),
            _ => Some(0),
        },
        Electric => match defending {
            Flying | Water => Some(1),
            Grass | Electric | Dragon => Some(-1),
            Ground => None,
            _ => Some(0),
        },
        Psychic => match defending {
            Fighting | Poison => Some(1),
            Steel | Psychic => Some(-1),
            Dark => None,
            _ => Some(0),
        },
        Ice => match defending {
            Flying | Ground | Grass | Dragon => Some(1),
            Steel | Fire | Water | Ice => Some(-1),
            _ => Some(0),
        },
        Dragon => match defending {
            Dragon => Some(1),
            Steel => Some(-1),
            Fairy => None,
            _ => Some(0),
        },
        Dark => match defending {
            Ghost | Psychic => Some(1),
            Fighting | Dark | Fairy => Some(-1),
            _ => Some(0),
        },
        Fairy => match defending {
            Fighting | Dragon | Dark => Some(1),
            Poison | Steel | Fire => Some(-1),
            _ => Some(0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_is_neutral_against_psychic() {
        assert_eq!(effectiveness(PokeType::Normal, PokeType::Psychic), 1.0);
    }

    #[test]
    fn normal_is_resisted_by_rock_and_immune_to_ghost() {
        assert_eq!(effectiveness(PokeType::Normal, PokeType::Rock), 0.5);
        assert_eq!(effectiveness(PokeType::Normal, PokeType::Ghost), 0.0);
    }

    #[test]
    fn type_set_contains_and_effectiveness() {
        let psychic = TypeSet::mono(PokeType::Psychic);
        assert!(psychic.contains(PokeType::Psychic));
        assert!(!psychic.contains(PokeType::Normal));
        assert_eq!(psychic.effectiveness(PokeType::Normal), 1.0);
    }

    #[test]
    fn scenario_relations() {
        use PokeType::*;
        let cloyster = TypeSet::dual(Water, Ice);
        let goodra_hisui = TypeSet::dual(Steel, Dragon);
        // Shock Wave (Electric) hits Cloyster (Water) super effectively; Goodra-Hisui
        // resists it (Dragon halves Electric), so there is no reason to use it there.
        assert_eq!(cloyster.effectiveness(Electric), 2.0);
        assert_eq!(goodra_hisui.effectiveness(Electric), 0.5);
        // Bulldoze (Ground) hits Goodra-Hisui (Steel) super effectively, Cloyster neutrally.
        assert_eq!(goodra_hisui.effectiveness(Ground), 2.0);
        assert_eq!(cloyster.effectiveness(Ground), 1.0);
        // Dark moves (Crunch / Dark Pulse) are neutral on both.
        assert_eq!(cloyster.effectiveness(Dark), 1.0);
        assert_eq!(goodra_hisui.effectiveness(Dark), 1.0);
    }

    #[test]
    fn full_chart_spot_checks() {
        use PokeType::*;
        // Dual super effective stacks to x4.
        assert_eq!(TypeSet::dual(Grass, Ground).effectiveness(Ice), 4.0);
        // Ice vs Water/Ice (Cloyster) is doubly resisted = 0.25.
        assert_eq!(TypeSet::dual(Water, Ice).effectiveness(Ice), 0.25);
        // Immunity zeroes the whole product.
        assert_eq!(TypeSet::mono(Flying).effectiveness(Ground), 0.0);
        assert_eq!(TypeSet::mono(Fairy).effectiveness(Dragon), 0.0);
        // Fairy super effective on Dragon, Steel resists Dragon.
        assert_eq!(effectiveness(Fairy, Dragon), 2.0);
        assert_eq!(effectiveness(Dragon, Steel), 0.5);
    }

    #[test]
    fn type_mod_steps_and_immunity() {
        assert_eq!(TypeSet::mono(PokeType::Psychic).type_mod(PokeType::Normal), Some(0));
        assert_eq!(TypeSet::mono(PokeType::Rock).type_mod(PokeType::Normal), Some(-1));
        // Dual resist (e.g. Rock/Steel) stacks to -2.
        assert_eq!(
            TypeSet::dual(PokeType::Rock, PokeType::Steel).type_mod(PokeType::Normal),
            Some(-2)
        );
        assert_eq!(TypeSet::mono(PokeType::Ghost).type_mod(PokeType::Normal), None);
    }
}
