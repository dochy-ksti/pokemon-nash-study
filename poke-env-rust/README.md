# poke-env-rust

Rust environment layer for AI agents that interact with `poke-sho-rust`.

## Role

- Convert battle state into model observations.
- Convert model policy outputs into legal battle actions.
- Manage battle sessions and future Pokemon Showdown-compatible IO.

This replaces the old Python `poke-env` dependency for the new Rust-first project.

