# poke-ai3-python 開発用 Makefile
#
# uv run の再ビルド判定はワークスペース依存先 (poke-env-rust / poke-sho-rust) の変更を
# 検知しないため、Rust を触った後に train-loop を直接叩くと _native.so が古いまま走る。
# このターゲットは実行前に必ず `maturin develop --release` を挟み、_native.so を最新に
# コピーしてから各コマンドを起動する。
#
# 使い方:
#   make build
#   make train ARGS="--num-games 32 --sims 64 --stage 3b --max-epochs 5 ..."
#   make eval-curve ARGS="..."

ARGS ?=

.PHONY: build train inspect-matchup eval-vs-rule eval-curve eval-ckpt-agreement eval-ckpt-vs-ckpt eval-hp-strategy eval-hp-grid play-gui export-policy-table

build:
	uv run maturin develop --release

train: build
	uv run train-loop $(ARGS)

inspect-matchup: build
	uv run inspect-matchup $(ARGS)

eval-vs-rule: build
	uv run eval-vs-rule $(ARGS)

eval-curve: build
	uv run eval-curve $(ARGS)

eval-ckpt-agreement: build
	uv run eval-ckpt-agreement $(ARGS)

eval-ckpt-vs-ckpt: build
	uv run eval-ckpt-vs-ckpt $(ARGS)

eval-hp-strategy: build
	uv run eval-hp-strategy $(ARGS)

eval-hp-grid: build
	uv run eval-hp-grid $(ARGS)

play-gui: build
	uv run play-gui $(ARGS)

export-policy-table: build
	uv run python scripts/export_policy_table.py $(ARGS)
