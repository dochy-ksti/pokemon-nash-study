# poke-sho-rust Phase1 Design

> **Status (updated):** Phase1 has moved from the original "damage = move power"
> toy to a Pokemon Showdown-compatible model: real Lv.50 stats computed from
> species base stats, the standard damage formula, and injectable randomness
> (16-step damage roll + critical hits). This document describes the current
> implementation. The original toy design is preserved in git history.

## Goal

Phase1 builds the smallest useful battle simulator for the new project, but with
real Pokemon mechanics so the same code path stays compatible with Pokemon
Showdown.

The game is:

- 1 player vs 1 player.
- Each player has exactly 1 Pokemon: **Mew** (loaded from the Phase1 team text).
- Both Pokemon use the same set, so stats are identical.
- Each Pokemon has exactly 2 moves:
  - `MoveId::Tackle` — Normal, Physical, power 40
  - `MoveId::Strength` — Normal, Physical, power 80
- Single-type Pokemon (Mew is Psychic); the type chart only fully encodes the
  Normal attacking row for now and defaults to neutral elsewhere.
- Physical/special split exists; both Phase1 moves are physical.
- No status conditions, abilities, items, boosts, weather, terrain, or switching.
- No accuracy checks yet (both moves are 100% accuracy).
- **Critical hits and the 16-step damage roll exist and are injected by the
  caller** (see [Randomness](#randomness)).
- Turn order is injected from outside so tests can be deterministic.

The expected AI lesson is still simple: Strength (power 80) is better than
Tackle (power 40).

## Important Design Constraint

The simulator must be deterministic when given deterministic inputs.

Do not call a global RNG inside the battle logic. Any randomness — turn order,
the damage roll, and critical hits — must come from explicit input. Turn order
is the `first_player` argument; the damage roll and crit come from a
`BattleRng` passed into `apply_turn`. This keeps tests easy and rollouts
reproducible.

## Crate Responsibility

Implement Phase1 battle rules inside `poke-sho-rust`.

Do not put battle rules in `poke-env-rust` or `poke-ai3`.

- `poke-env-rust` wraps this simulator (`local_showdown.rs`), converts state into
  observations/actions, and supplies a seeded `BattleRng`.
- `poke-ai3` trains with those observations.

## Module Layout

`lib.rs` exposes:

```rust
pub mod battle_rng;
pub mod damage;
pub mod moves;
pub mod phase1;
pub mod species;
pub mod team;
pub mod types;
```

- `types.rs` — `PokeType`, `TypeSet`, and the type-effectiveness chart.
- `species.rs` — `Stats`, `Nature`, `Species`, base stats (Mew), and the
  Lv.-based real-stat computation.
- `moves.rs` — `MoveCategory`, `MoveData`, and the Tackle/Strength definitions.
- `damage.rs` — the damage formula.
- `battle_rng.rs` — the `BattleRng` trait and the deterministic `MaxRoll`.
- `team.rs` — Pokemon Showdown team-text parsing and set resolution.
- `phase1.rs` — core battle types and turn resolution.

## Stats

Real stats follow the standard Generation 3+ / Showdown formulas:

```text
HP    = floor((2*Base + IV + floor(EV/4)) * Level / 100) + Level + 10
Other = floor((floor((2*Base + IV + floor(EV/4)) * Level / 100) + 5) * Nature)
```

`Nature` is 1.1 / 1.0 / 0.9 applied with integer flooring (`*11/10`, `*1`,
`*9/10`).

Mew has base stats all 100 and a single Psychic type. The Phase1 team text omits
EVs/IVs/level, so the resolver applies Showdown defaults — **IV 31, EV 0** — and
the phase convention of **level 50** (see `team::DEFAULT_LEVEL`). Serious is a
neutral nature. The resulting Lv.50 spread is:

| Stat | Value |
| ---- | ----- |
| HP   | 175   |
| Atk  | 120   |
| Def  | 120   |
| SpA  | 120   |
| SpD  | 120   |
| Spe  | 120   |

```rust
pub const PHASE1_MAX_HP: i32 = 175;
```

`PHASE1_MAX_HP` is kept as a constant for environment/normalization code and
equals the HP computed from Mew's base stats.

## Core Types

### Player

```rust
pub enum Player { P1, P2 }
```

Helpers: `index(self) -> usize`, `opponent(self) -> Player`. `P1`/`P2`
correspond to Showdown's `p1`/`p2`.

### MoveId

```rust
pub enum MoveId { Tackle, Strength }
```

Methods: `data(self) -> MoveData`, `power(self) -> u16`, `index(self) -> usize`.
The names map directly to the real Showdown moves (`tackle`, `strength`).

### Choice

```rust
pub struct Choice { pub move_id: MoveId }
```

A wrapper so later phases can add switching without replacing the caller API.

### PokemonState

```rust
pub struct PokemonState {
    pub hp: i32,
    pub max_hp: i32,
    pub stats: Stats,
    pub types: TypeSet,
    pub level: u16,
}
```

Rules:

- HP never goes below 0 after damage.
- A Pokemon is fainted when `hp <= 0`.
- Built from a resolved Showdown set (`max_hp == stats.hp`).

### BattleState

```rust
pub struct BattleState {
    pub pokemon: [PokemonState; 2],
    pub turn: u32,
}
```

Constructors/queries: `new_phase1()` (parses the included team text and builds
both identical Mews), `pokemon`, `hp`, `is_fainted`, `winner`, `is_done`,
`legal_choices`. `legal_choices` returns Tackle and Strength while the battle is
live, and an empty vector when the battle is done or the player has fainted.

## Randomness

```rust
pub trait BattleRng {
    /// Damage roll percent in 85..=100.
    fn damage_roll(&mut self) -> u8;
    /// Whether this hit is a critical hit, given the crit stage.
    fn is_crit(&mut self, stage: u8) -> bool;
}
```

- `MaxRoll` is the deterministic implementation: roll 100%, never crits
  (equivalent to "no random damage, no critical hits"). Used by tests and by the
  deterministic `poke-env-rust` `Game` wrapper.
- `poke-env-rust::local_showdown::LocalRng` is the seeded implementation used in
  rollouts. It honours two independent toggles, `randomize` (16-step roll) and
  `crit_enabled` (1/24 at stage 0). Both default to enabled and are threaded
  from `phase1-loop`'s `--random/--no-random` and `--crit/--no-crit` flags
  through PyO3 → `RustAsyncExecutor` → `game_start` → `create_local_game`.

Crit denominators by stage come from `battle_rng::crit_denominator`.

## Damage Formula

`damage::calc_damage` implements the Generation 5+ / Showdown formula with
integer flooring, then applies modifiers in Showdown's order:

```text
level_factor = floor(2*Level/5) + 2
base         = floor(floor(level_factor * Power * A / D) / 50) + 2
dmg          = base
             -> if crit:  floor(dmg * 3/2)
             -> floor(dmg * roll / 100)        // roll in 85..=100
             -> if STAB:  floor(dmg * 3/2)
             -> floor(dmg * type_effectiveness)
```

`A`/`D` are the attacker's Atk/SpA and the defender's Def/SpD depending on the
move category. A connecting damaging move deals at least 1 HP; an immune target
(effectiveness 0) takes 0. Status / zero-power moves deal 0.

For the Phase1 mirror match (Mew vs Mew, Normal physical moves, no STAB,
Psychic defender = neutral), the deterministic (max-roll, no-crit) damage is:

- Tackle: `floor(floor(22*40*120/120 / 50) + 2)` = **19**
- Strength: `floor(floor(22*80*120/120 / 50) + 2)` = **37**
- Strength with a critical hit at max roll = **55**

The `Event::Damage { amount, .. }` records actual HP lost (clamped to remaining
HP), so totals stay consistent with HP changes.

## Turn Resolution

```rust
pub fn apply_turn<R: BattleRng>(
    state: BattleState,
    p1_choice: Choice,
    p2_choice: Choice,
    first_player: Player,
    rng: &mut R,
) -> TurnResult
```

`first_player` and `rng` are injected by the caller. The input `state` is not
mutated in place; a new state is returned inside `TurnResult`.

### TurnResult

```rust
pub struct TurnResult { pub state: BattleState, pub events: Vec<Event> }
```

### Event

```rust
pub enum Event {
    MoveUsed { player: Player, move_id: MoveId },
    Crit { target: Player },
    Damage { target: Player, amount: i32, hp_remaining: i32 },
    Fainted { player: Player },
    Win { player: Player },
    Draw,
}
```

Event order matches execution order: `MoveUsed`, then `Crit` (only on a
critical hit that deals damage), then `Damage`, then `Fainted` + `Win` on a KO.
`Crit` maps to Showdown's `-crit` protocol line in `local_showdown.rs`.

## Action Order and Fainting

1. If `state.is_done()` already, return the same state and no events.
2. Determine order from `first_player`.
3. First player uses its move; apply damage to the opponent.
4. If the opponent faints, emit `Fainted` + `Win`; the second player does not act.
5. Otherwise the second player acts and applies damage.
6. On a KO emit `Fainted` + `Win`.
7. Increment `turn` by 1 exactly once if the input battle was not already done
   (a turn ending in a KO still increments `turn`).

A KO skips the second move, so Phase1 does not naturally produce a draw. `Draw`
is kept for later phases.

## Invalid Input Policy

- `apply_turn` is total and does not panic for normal use.
- On a finished battle it returns the unchanged state and empty events.
- A fainted player that somehow tries to act is skipped.

Team parsing uses a `TeamError` for unknown species/moves/natures and malformed
stat lines; `new_phase1` expects the bundled, known-good team text and unwraps.

## Showdown Compatibility

- Player and move names match Showdown (`p1`/`p2`, `tackle`/`strength`).
- `team.rs` parses the Showdown export text format (species, optional
  nickname/item/ability/tera/level, nature, EVs, IVs, move list).
- `local_showdown.rs` emits Showdown-style protocol items (`move`, `-crit`,
  `-damage`, `faint`, `win`) so the in-process backend mirrors the subprocess
  `pokemon-showdown` backend.

The in-process backend matches Showdown's damage *formula*; it does not
reproduce Showdown's exact PRNG sequence.

## Tests (in `src/`)

Each module carries `#[cfg(test)]` tests. Key cases:

- `species`: Mew Lv.50 Serious spread is 175 / 120×5; nature ±10%; nature parsing.
- `types`: Normal neutral vs Psychic, resisted by Rock, immune to Ghost.
- `moves`: lookup by id/name.
- `damage`: Tackle 19 / Strength 37 at max roll; min-roll flooring; crit before
  roll (55); immune target takes 0.
- `team`: parses and resolves the Phase1 team; nickname/item/EV/IV parsing;
  unknown species errors.
- `phase1`: initial state (HP 175, Atk 120); Tackle/Strength damage; crit emits
  an event and deals more; KO prevents the second move; second player can win
  when moving first; finished battle unchanged; HP never below zero.

## Completion Definition

- The public Phase1 API exists in `poke-sho-rust`.
- `cargo test -p poke-sho-rust` passes.
- `cargo test --workspace` passes.
- No code outside `poke-sho-rust` is required to test Phase1 battle rules.
