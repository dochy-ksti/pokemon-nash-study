# poke-ai3

AI training and inference project for the Rust-first Pokemon AI stack.

## Role

- Train a policy/value model for Pokemon battle decisions.
- Use modern BERT-style models for board-state encoding.
- Evaluate moves with short random rollouts and averaged value estimates.
- Use PyTorch for training.

The initial target is deliberately tiny: learn to choose the stronger move in a
minimal 1v1 game, then add mechanics step by step.

## Project Data

- `experiments/` stores experiment reports.
- `data/` stores local training/evaluation data.
