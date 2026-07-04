//! Rust battle simulator for the poke-ai3 project.
//!
//! The current learning scenario is 1v1 Cloyster (Def 特化) vs Goodra (SpD 特化),
//! and the agent should learn to choose between an equal-power physical move
//! (Crunch) and special move (Dark Pulse) based on the opponent's defensive bias.
//! Showdown message-format compatibility is preserved where practical so the
//! same AI can later be used against a Showdown server.
//!
//! `battle` holds the simulation engine and state model; `scenario` holds the
//! current scenario's species/moves/teams and Showdown interop. Renaming the
//! scenario for a future milestone touches only `scenario`.

pub mod battle;
pub mod battle_rng;
pub mod damage;
pub mod event;
pub mod global_ids;
pub mod moves;
pub mod party;
pub mod scenario;
pub mod species;
pub mod team;
pub mod turn;
pub mod types;
