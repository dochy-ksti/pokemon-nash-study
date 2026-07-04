//! ブラウザ用の戦闘 RNG。`poke-env-rust` の `BattleChaCha` と同一挙動だが、
//! tokio/rand を引かないよう `rand_chacha` だけで最小再実装する。乱数・急所を有効化し、
//! 本番デプロイ設定 (16 段ダメージ乱数 + 急所あり) を再現する。seed は JS から注入する。

use poke_sho_rust::battle::Player;
use poke_sho_rust::battle_rng::{BattleRng, crit_denominator};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

/// ChaCha8 ベースの戦闘 RNG。`randomize`/`crit_enabled` は本番では常に true。
pub struct WasmRng {
    rng: ChaCha8Rng,
    randomize: bool,
    crit_enabled: bool,
}

impl WasmRng {
    pub fn from_u64(seed: u64, randomize: bool, crit_enabled: bool) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            randomize,
            crit_enabled,
        }
    }

    /// `0..n` の一様乱数 (剰余バイアスを避ける棄却サンプリング)。
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let zone = u32::MAX - (u32::MAX % n);
        loop {
            let v = self.rng.next_u32();
            if v < zone {
                return v % n;
            }
        }
    }

    /// 公平なコイン (速度タイ用)。`true` で P1 が先手。
    pub fn first_player(&mut self) -> Player {
        if self.below(2) == 0 {
            Player::P1
        } else {
            Player::P2
        }
    }
}

impl BattleRng for WasmRng {
    fn damage_roll(&mut self) -> u8 {
        if !self.randomize {
            return 100;
        }
        85 + self.below(16) as u8
    }

    fn is_crit(&mut self, stage: u8) -> bool {
        if !self.crit_enabled {
            return false;
        }
        self.below(crit_denominator(stage)) == 0
    }
}
