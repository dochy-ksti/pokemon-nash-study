//! ChaCha8 ベースの戦闘用 RNG。
//!
//! 以前はローカル試合・rollout とも自前の LCG を使っていたが、LCG は下位ビットの
//! 品質が悪く (bit0 の周期は 2)、速度タイ (`& 1`) が退化して先手が固定化する不具合が
//! あった。ここでは `rand_chacha::ChaCha8Rng` に統一し、全ビット良質・seed から完全に
//! 再現可能な乱数を供給する。状態は小さく、rollout ごとに seed から作り直すだけで使える。

use poke_sho_rust::battle::Player;
use poke_sho_rust::battle_rng::{BattleRng, crit_denominator};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// 戦闘の確率要素 (16 段ダメージ乱数・急所・速度タイ・rollout 深さ) を供給する RNG。
/// `randomize` / `crit_enabled` が false のときは決定論モード (最大ロール / 急所なし)。
pub struct BattleChaCha {
    rng: ChaCha8Rng,
    randomize: bool,
    crit_enabled: bool,
}

impl BattleChaCha {
    /// u64 seed から作る (rollout 用)。
    pub fn from_u64(seed: u64, randomize: bool, crit_enabled: bool) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            randomize,
            crit_enabled,
        }
    }

    /// 4 ワード seed から作る (ローカル試合用)。16 バイトを 32 バイト seed の前半に詰める。
    pub fn from_seed_words(seed: [u32; 4], randomize: bool, crit_enabled: bool) -> Self {
        let mut bytes = [0u8; 32];
        for (i, w) in seed.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Self {
            rng: ChaCha8Rng::from_seed(bytes),
            randomize,
            crit_enabled,
        }
    }

    /// 公平なコイン (速度タイ用)。`true` で P1 が先手。
    pub fn first_player(&mut self) -> Player {
        if self.rng.r#gen::<bool>() {
            Player::P1
        } else {
            Player::P2
        }
    }

    /// `0..n` の一様乱数 (rollout 深さ選択用)。`n == 0` は 0 を返す。
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { self.rng.gen_range(0..n) }
    }

    /// `[0.0, 1.0)` の一様乱数 (方策サンプリング用)。
    pub fn unit(&mut self) -> f32 {
        self.rng.r#gen::<f32>()
    }
}

impl BattleRng for BattleChaCha {
    fn damage_roll(&mut self) -> u8 {
        if !self.randomize {
            return 100;
        }
        85 + self.rng.gen_range(0..16) as u8
    }

    fn is_crit(&mut self, stage: u8) -> bool {
        if !self.crit_enabled {
            return false;
        }
        self.rng.gen_range(0..crit_denominator(stage)) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_player_is_reproducible_per_seed() {
        // 同じ seed なら先手の列も完全に一致する (replay / lookahead 用)。
        let seq = |seed| {
            let mut r = BattleChaCha::from_seed_words(seed, false, false);
            (0..12).map(|_| r.first_player()).collect::<Vec<_>>()
        };
        assert_eq!(seq([2, 3, 4, 5]), seq([2, 3, 4, 5]));
        assert_ne!(seq([2, 3, 4, 5]), seq([6, 7, 8, 9]));
    }

    #[test]
    fn first_player_is_balanced_across_seeds() {
        // 旧 LCG は turn1 が常に P2 だった。ChaCha8 では seed ごとにばらけて ≈50/50。
        let mut p1 = 0;
        let n = 400;
        for g in 0..n {
            let seed = [g * 4 + 1, g * 4 + 2, g * 4 + 3, g * 4 + 4];
            let mut r = BattleChaCha::from_seed_words(seed, false, false);
            if r.first_player() == Player::P1 {
                p1 += 1;
            }
        }
        // 偏り < 10% (旧実装ならここで 0 になり落ちる)。
        let rate = p1 as f64 / n as f64;
        assert!((0.4..0.6).contains(&rate), "turn1 P1 率が偏っている: {rate}");
    }
}
