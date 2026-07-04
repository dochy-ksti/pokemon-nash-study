//! Randomness injected into battle resolution.
//!
//! Per the project's determinism constraint, the simulator never reads a global
//! RNG. Callers pass a [`BattleRng`] so tests and reproducible rollouts can use
//! [`MaxRoll`], while real games supply a seeded implementation.

/// Critical-hit probability denominators by crit stage (Gen 7+). Stage 0 is
/// 1/24; higher stages are guaranteed.
const CRIT_DENOM: [u32; 4] = [24, 8, 2, 1];

/// Source of per-hit randomness: the 16-step damage roll and critical hits.
pub trait BattleRng {
    /// Damage roll as a percent in `85..=100`.
    fn damage_roll(&mut self) -> u8;

    /// Whether this hit is a critical hit, given the crit `stage`.
    fn is_crit(&mut self, stage: u8) -> bool;
}

/// Returns the crit denominator for a stage (clamped to the known table).
pub fn crit_denominator(stage: u8) -> u32 {
    let idx = (stage as usize).min(CRIT_DENOM.len() - 1);
    CRIT_DENOM[idx]
}

/// Deterministic RNG: always the maximum (100%) damage roll and never a crit.
/// Equivalent to "no random damage, no critical hits".
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxRoll;

impl BattleRng for MaxRoll {
    fn damage_roll(&mut self) -> u8 {
        100
    }

    fn is_crit(&mut self, _stage: u8) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_roll_is_deterministic() {
        let mut rng = MaxRoll;
        assert_eq!(rng.damage_roll(), 100);
        assert!(!rng.is_crit(0));
    }

    #[test]
    fn crit_denominator_clamps() {
        assert_eq!(crit_denominator(0), 24);
        assert_eq!(crit_denominator(3), 1);
        assert_eq!(crit_denominator(99), 1);
    }
}
