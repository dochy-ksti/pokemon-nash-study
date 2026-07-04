# AGENTS.md

This file provides guidance to AI agents when working in this repository.

## Language

- Always respond to the user in Japanese (日本語).

## Repository Layout

- `poke-sho-rust/` - New Rust battle simulator project. This is our code.
- `poke-env-rust/` - New Rust environment/interface layer. This is our code.
- `poke-ai3/` - New Rust-first AI training and inference project. This is our code.
- `poke_ai2/` - Abandoned Python project. Keep for reference only; do not modify unless explicitly requested.
- `poke_ai/` - Older abandoned project. Keep for reference only; do not modify.
- `poke-env/` - Third-party Python library. Do not modify.
- `pokemon-showdown/` - Third-party TypeScript simulator/server. Do not modify.
- `poke_poke/` - Reference project with similar training ideas. Refer to this only when explicitly instructed.
- `teams/` - Team data from the previous project. Treat as shared data; do not rewrite casually.

## New Project Direction

The active direction is now the Rust-first `poke-ai3` stack:

1. `poke-sho-rust` reimplements Pokemon battle simulation in Rust.
2. `poke-env-rust` provides the agent environment layer on top of the simulator.
3. `poke-ai3` trains and runs the AI. Training is expected to use PyTorch.

The long-term goal is an AI that can battle on Pokemon Showdown-compatible
interfaces and achieve a rating beyond top human players. Compatibility with
Pokemon Showdown message formats should be preserved where practical so the same
AI can eventually be used against a Showdown server.

The initial implementation should start from intentionally tiny games:

1. 1v1 identical Pokemon with two moves, power 40 and power 80, and learn to choose power 80.
2. Add physical/special moves and attack/defense/special-attack/special-defense differences.
3. Add Pokemon and move types, then learn type-aware move choice.
4. Add 3v3 battles and learn switching.
5. Add speed and learn speed-aware decisions.
6. After each deterministic phase works, add accuracy, critical hits, and 16-step damage rolls to test whether learning remains stable.

## Commands

Run Rust commands from the repository root unless a task explicitly targets one package.

```bash
cargo check --workspace
cargo test --workspace
cargo test -p poke-sho-rust
cargo test -p poke-env-rust
```

Use `uv` for Python. Do not use `pip`.

Run the current training loop from `poke-ai3-python/`. Prefer `make` over a bare
`uv run train-loop`: `uv run` の再ビルド判定はワークスペース依存先 (poke-env-rust /
poke-sho-rust) の変更を検知しないため、Rust を編集した後に `train-loop` を直接叩くと
`_native.so` が古いまま走る。`make` は実行前に必ず `maturin develop --release` を挟む。

```bash
cd poke-ai3-python
make train ARGS="--num-games 32 --sim-concurrency 16 --sims 64 \
  --search-turn-min 4 --search-turn-max 8 --no-random --no-crit --stage 3b \
  --max-epochs 5 --max-batch-size 439"
```

`make` を使わず `uv run train-loop` を直接叩いた場合でも、起動時に `_native.so` の鮮度を
検査し、Rust ソースより古ければ再ビルドを促して停止する (回避は `POKE_AI3_SKIP_FRESH_CHECK=1`)。
他のコマンドにも `make build` / `make eval-curve ARGS=...` 等のターゲットがある。

`--max-batch-size` の制約に注意: 同時 in-flight 推論数の理論上限は
`num-games * sim-concurrency * 2` なので、これを超える `--max-batch-size` を指定すると
バッチが永遠に埋まらず生成が停滞する (進捗ログが一切出ないハングに見える)。
`--max-batch-size` はこの理論上限 (`num-games * sim-concurrency * 2`) の 3/7 を推奨・既定とする。
例: num-games 32 * sim-concurrency 16 → 理論上限 1024 × 3/7 ≒ 439 が目安
(未指定なら train-loop / funnel が `num-games*sim-concurrency*2*3/7` を自動算出。
experiments/poke-ai3 20260703 の 12/36 epoch バッチ掃引で 2/5〜3/5 は横並び=ノイズ帯と確認し 3/7 を採用)。

## Rust Code Style

- Use Rust 2024 edition for the new crates.
- Keep crates small and responsibilities clear:
  - simulator rules and battle transitions in `poke-sho-rust`
  - observations, legal action mapping, and environment/session logic in `poke-env-rust`
  - model, training, evaluation, and executable tools in `poke-ai3`
- Prefer deterministic APIs and explicit RNG injection for simulation code.
- Early phase battle rules should start deterministic. Any randomness must be injected from callers.
- Add tests alongside new battle mechanics as they are introduced.
- Keep public APIs narrow until the small-game milestones stabilize.
- Keep each source file within 300 lines excluding test code. If a file would exceed that,
  split it by clear feature or responsibility boundaries before adding more code.

## Python Legacy Style

Only applies when explicitly asked to touch legacy Python projects:

- Python 3.12+ type hints; use `X | Y` and `| None`, not `Union` or `Optional`.
- Add `from __future__ import annotations` at the top of every Python file.
- Use `TYPE_CHECKING` guards for type-only imports when needed.
- Max line length: 100 characters.

## Workflow

- Do not implement code changes until explicitly instructed by the user.
- Do not use the auto-memory system. Do not write to or read from the memory
  directory (`~/.claude/projects/-home-dochy-pokemon-ai-proj/memory/`). All
  context must come from this repository (code, AGENTS.md, CLAUDE.md, docs,
  experiment records, git history) or the current conversation.
- If a command is expected to take more than five minutes, explain the exact command and all command-line arguments before running it.
- Do not modify abandoned or third-party projects unless the user explicitly requests it.
- Do not append a `Co-Authored-By` trailer (or any AI attribution) to git commit messages.
- If creating a handoff document, save it under `docs/poke-ai3/` (repository root).
- Handoff filenames must use JST timestamps from:

```bash
TZ=Asia/Tokyo date +%Y%m%d_%H%M
```

- Handoff filename format:

```text
YYYYMMDD_hhmm_このセッションでやったことを短くまとめた適切なタイトル.md
```

## Experiments

- Run benchmarks and training experiments one at a time.
- Record experiment purpose, command, and result in `experiments/poke-ai3/` (repository root).
- Store local training/evaluation data under `data/poke-ai3/` (repository root).
- Checkpoint cleanup: after an experiment finishes you may delete its checkpoints,
  EXCEPT the final checkpoint and any checkpoints the user asked you to keep. Keep
  those. Do not delete checkpoints while an experiment might still resume from them
  (e.g. mid-run work/snapshot checkpoints intended for `--start` continuation).
- Experiment filenames must use JST timestamps from `TZ=Asia/Tokyo date +%Y%m%d_%H%M`.
- Experiment filename format:

```text
YYYYMMDD_hhmm_この実験を短くまとめた適切なタイトル.md
```
