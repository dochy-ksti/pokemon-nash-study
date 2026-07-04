//! Common battle-event vocabulary: a typed mirror of the Pokemon Showdown
//! protocol messages that matter for parity.
//!
//! Both the simulator (`poke-sho-rust`) and the Showdown protocol parser
//! (`poke-env-rust`) produce sequences of these events. Parity is checked by
//! projecting both sides onto this vocabulary and comparing the sequences;
//! Showdown-only decoration (`|t:|`, `|turn|`, `|upkeep|`, `|-supereffective|`,
//! ...) is dropped during conversion.
//!
//! Entities are identified the way Showdown identifies them: a Pokemon by its
//! ident broken into parts, and a move by its normalized id string
//! (Showdown `toID`, e.g. `"tackle"`).

use serde::{Deserialize, Serialize};

/// Reference to a Pokemon, mirroring Showdown idents. The active-field form is
/// `p1a: Mew` and the request/party form is `p1: Mew`; we keep only the player
/// number and the label, since singles never has more than one active slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PokemonRef {
    /// 1-based player number (Showdown `p1` / `p2`).
    pub player: u8,
    /// The label after the colon in the ident: nickname, or species name when
    /// no nickname is set (exactly as Showdown prints it).
    pub name: String,
}

impl PokemonRef {
    pub fn new(player: u8, name: impl Into<String>) -> Self {
        Self {
            player,
            name: name.into(),
        }
    }
}

/// A single battle event in the common vocabulary.
///
/// Move and Pokemon identifiers use Showdown normalized forms so the simulator
/// and the Showdown parser land in the same key space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Event {
    /// `|move|<user>|<Move>|<target>`. `move_id` is the normalized id.
    Move {
        user: PokemonRef,
        move_id: String,
        target: PokemonRef,
    },
    /// `|-crit|<target>`.
    Crit { target: PokemonRef },
    /// `|-damage|<target>|<cur>/<max>` (or `0 fnt`). `hp` is the absolute HP
    /// remaining after the hit; `fainted` is true when the condition is `fnt`.
    Damage {
        target: PokemonRef,
        hp: u32,
        max_hp: u32,
        fainted: bool,
    },
    /// `|switch|<who>|<species>, L<level>|<cur>/<max>`. Brings a party member
    /// into the active slot (voluntary or forced replacement after a faint).
    Switch {
        who: PokemonRef,
        species: String,
        hp: u32,
        max_hp: u32,
    },
    /// `|faint|<target>`.
    Faint { target: PokemonRef },
    /// `|win|<player name>`.
    Win { player: String },
    /// `|tie|`.
    Tie,
    /// `|turn|<n>`. ターン境界マーカー。シミュレータ (`apply_turn`) は生成せず、
    /// Showdown 由来のイベント列をターン単位に分割する区切りとしてのみ使う。
    Turn { n: u32 },
}
