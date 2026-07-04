# poke-sho-rust

Rust battle simulator for the poke-ai3 effort.

## Role

- Reimplement Pokemon battle simulation in Rust, starting from tiny controllable games.
- Preserve compatibility with Pokemon Showdown message formats where practical.
- Provide deterministic, fast rollouts for training and evaluation.

## First Milestones

1. 1v1 identical Pokemon with two moves: power 40 and power 80.
2. Physical/special moves with attack, defense, special attack, and special defense.
3. Type matchups and type-aware move selection.
4. 3v3 battles with switching.
5. Speed-aware action order.

Phase1 design lives in [docs/phase1_design.md](docs/phase1_design.md).
