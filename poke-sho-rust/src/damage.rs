//! Damage calculation faithfully mirroring Pokemon Showdown's Gen 9
//! `getDamage` / `modifyDamage` pipeline.
//!
//! Showdown computes an integer base damage, then applies modifiers in a fixed
//! order using two truncation primitives:
//! - [`tr`]: `x >>> 0`, a 32-bit unsigned truncation (floor for our magnitudes);
//! - [`modify`]: 4096-based fixed-point multiply with round-half-up.
//!
//! Order: `base += 2`, critical hit (`tr(base * 1.5)`), damage roll, STAB
//! (`modify(base, 1.5)`), then per-step type effectiveness (`*2` or `tr(/2)`).
//! A connecting damaging move deals at least 1 HP unless the target is immune.

use crate::moves::{MoveCategory, MoveData};
use crate::species::Stats;
use crate::types::TypeSet;

/// Showdown's `Dex.trunc(num)` with no bit argument: `num >>> 0`, i.e. a 32-bit
/// unsigned truncation. For the non-negative magnitudes in the damage formula
/// this is equivalent to flooring.
fn tr(x: u64) -> u64 {
    x % (1u64 << 32)
}

/// Showdown's `Dex.trunc(num, 16)`: `(num >>> 0) % 2**16`.
fn tr16(x: u64) -> u64 {
    x % (1u64 << 16)
}

/// Showdown's `Battle.modify(value, numerator, denominator)`: a 4096-based
/// fixed-point multiply, `tr((tr(value * modifier) + 2048 - 1) / 4096)` where
/// `modifier = tr(numerator * 4096 / denominator)`.
fn modify(value: u64, numerator: u64, denominator: u64) -> u64 {
    let modifier = tr(numerator * 4096 / denominator);
    tr((tr(value * modifier) + 2048 - 1) / 4096)
}

/// Inputs describing one attacker-vs-defender damage calculation.
pub struct DamageInput<'a> {
    pub level: u16,
    pub attacker: &'a Stats,
    pub attacker_types: TypeSet,
    pub defender: &'a Stats,
    pub defender_types: TypeSet,
    pub mv: &'a MoveData,
    pub crit: bool,
    /// Damage roll percent in `85..=100`.
    pub roll: u8,
}

/// Computes the HP damage of a single move use.
pub fn calc_damage(input: &DamageInput) -> i32 {
    if input.mv.category == MoveCategory::Status || input.mv.power == 0 {
        return 0;
    }

    // Type immunity: the move never reaches `modifyDamage` in Showdown.
    let type_mod = match input.defender_types.type_mod(input.mv.move_type) {
        None => return 0,
        Some(steps) => steps.clamp(-6, 6),
    };

    let (atk, def) = match input.mv.category {
        MoveCategory::Physical => (input.attacker.atk, input.defender.def),
        MoveCategory::Special => (input.attacker.spa, input.defender.spd),
        MoveCategory::Status => unreachable!(),
    };
    let def = (def as u64).max(1);

    let level = input.level as u64;
    let level_factor = 2 * level / 5 + 2;
    // getDamage: base = tr(tr(tr(tr(2L/5+2) * power * A) / D) / 50)
    let mut dmg = tr(tr(tr(level_factor * input.mv.power as u64 * atk as u64) / def) / 50);

    // modifyDamage begins.
    dmg += 2;

    // Critical hit (Gen 6+): tr(base * 1.5). Not a 4096 modifier.
    if input.crit {
        dmg = tr(dmg * 3 / 2);
    }

    // Damage roll: tr(tr(base * (100 - r)) / 100), with `roll` = 100 - r in 85..=100.
    let roll = (input.roll as u64).clamp(85, 100);
    dmg = tr(tr(dmg * roll) / 100);

    // STAB: 4096 fixed-point x1.5.
    if input.attacker_types.contains(input.mv.move_type) {
        dmg = modify(dmg, 3, 2);
    }

    // Type effectiveness: step-by-step doubling / halving with truncation.
    if type_mod > 0 {
        for _ in 0..type_mod {
            dmg *= 2;
        }
    } else if type_mod < 0 {
        for _ in 0..(-type_mod) {
            dmg = tr(dmg / 2);
        }
    }

    // Gen != 5: minimum 1 damage applies after the final modifiers.
    if dmg == 0 {
        return 1;
    }

    // 16-bit truncation happens last (can truncate to 0, but min-1 above guards
    // the common case for our magnitudes).
    tr16(dmg) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{CRUNCH, DARK_PULSE};
    use crate::species::{CLOYSTER, Nature, compute_stats};
    use crate::types::PokeType;

    fn cloyster_stats() -> Stats {
        compute_stats(
            CLOYSTER.base,
            Stats::DEFAULT_IVS,
            Stats::ZERO,
            50,
            Nature::Serious,
        )
    }

    fn input<'a>(stats: &'a Stats, mv: &'a MoveData, crit: bool, roll: u8) -> DamageInput<'a> {
        DamageInput {
            level: 50,
            attacker: stats,
            attacker_types: CLOYSTER.types,
            defender: stats,
            defender_types: CLOYSTER.types,
            mv,
            crit,
            roll,
        }
    }

    #[test]
    fn crunch_vs_cloyster_max_roll_no_crit() {
        // Atk 115, Def 200, power 80, Dark vs Water/Ice = 1x, no STAB.
        // base = floor(floor(floor(22*80*115)/200)/50) = floor(floor(1012)/50) = 20.
        // +2 = 22, roll 100 → 22.
        let s = cloyster_stats();
        assert_eq!(calc_damage(&input(&s, &CRUNCH, false, 100)), 22);
    }

    #[test]
    fn dark_pulse_vs_cloyster_max_roll_no_crit() {
        // SpA 105, SpD 65, power 80, Dark vs Water/Ice = 1x, no STAB.
        // base = floor(floor(184800/65)/50) = floor(2843/50) = 56.
        // +2 = 58, roll 100 → 58.
        let s = cloyster_stats();
        assert_eq!(calc_damage(&input(&s, &DARK_PULSE, false, 100)), 58);
    }

    #[test]
    fn min_roll_floors_lower() {
        // Dark Pulse base 58 → 58 * 85 / 100 = floor(49.3) = 49.
        let s = cloyster_stats();
        assert_eq!(calc_damage(&input(&s, &DARK_PULSE, false, 85)), 49);
    }

    #[test]
    fn crit_multiplies_before_roll() {
        // Dark Pulse base 58 → crit floor(58 * 3 / 2) = 87 → roll 100 → 87.
        let s = cloyster_stats();
        assert_eq!(calc_damage(&input(&s, &DARK_PULSE, true, 100)), 87);
    }

    #[test]
    fn modify_matches_showdown_fixed_point() {
        // modify(v, 3, 2) == tr((tr(v*6144) + 2047) / 4096).
        assert_eq!(super::modify(58, 3, 2), 87);
        assert_eq!(super::modify(100, 3, 2), 150);
        assert_eq!(super::modify(1, 3, 2), 1);
    }

    #[test]
    fn stab_applies_4096_modifier() {
        // Attacker treated as Dark-typed → Dark Pulse base 58 with STAB = 87.
        let s = cloyster_stats();
        let mut inp = input(&s, &DARK_PULSE, false, 100);
        inp.attacker_types = TypeSet::mono(PokeType::Dark);
        assert_eq!(calc_damage(&inp), 87);
    }

    #[test]
    fn resisted_halves_with_truncation() {
        // Defender Fighting → Dark resisted, type_mod = -1, floor(58/2) = 29.
        let s = cloyster_stats();
        let mut inp = input(&s, &DARK_PULSE, false, 100);
        inp.defender_types = TypeSet::mono(PokeType::Fighting);
        assert_eq!(calc_damage(&inp), 29);
    }

    #[test]
    fn super_effective_doubles() {
        // Defender Psychic → Dark super effective, type_mod = +1, 58 * 2 = 116.
        let s = cloyster_stats();
        let mut inp = input(&s, &DARK_PULSE, false, 100);
        inp.defender_types = TypeSet::mono(PokeType::Psychic);
        assert_eq!(calc_damage(&inp), 116);
    }
}
