//! Rust environment layer for agents that play through `poke-sho-rust`.
//!
//! This crate should own observation/action conversion, battle lifecycle
//! management, and any compatibility glue needed to run the same AI against a
//! Pokemon Showdown-compatible server.

pub mod battle_chacha;
pub mod local_showdown;
pub mod lookahead;
pub mod nash;
pub mod observation;
pub mod oracle;
pub mod protocol;
pub mod showdown_trait;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_loads() {
        assert_eq!(env!("CARGO_PKG_NAME"), "poke-env-rust");
    }
}
