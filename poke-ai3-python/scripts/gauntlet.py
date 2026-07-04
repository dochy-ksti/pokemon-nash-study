#!/usr/bin/env python
"""連続学習チェックポイントの「新 > 旧 (全勝)」仮説を gauntlet で検証する orchestrator。

各ラウンドで:
  1. 現 checkpoint から N エポック追加学習し、新スナップショットを作る (train-loop)。
  2. その新スナップショットを「過去の強かった checkpoint」プール (最大 K 個) の全員と
     直接対戦させる (eval_ckpt_vs_ckpt, 先発有利相殺のため両 side)。
  3. 新スナップショットをプールに追加し、直近 K 個に絞る。
これを R ラウンド繰り返し、各新 ckpt が旧 ckpt 全員に勝ち越すか (>0.5) を表で出す。

学習・eval とも既存 CLI/module をサブプロセスで叩く (検証済み経路をそのまま使う)。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DATA = Path("/home/dochy/pokemon_ai_proj/data/poke-ai3")
GDIR = DATA / "gauntlet"

# 共通のシミュレーション設定 (学習時と揃える)。
SIM = [
    "--sim-concurrency", "16", "--sims", "64",
    "--search-turn-min", "6", "--search-turn-max", "12",
    "--no-random", "--no-crit", "--stage", "3b",
]

_RESULT = re.compile(r"RESULT a_win=(\d+) b_win=(\d+) draw=(\d+)")


def run(cmd: list[str]) -> str:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def head_to_head(new: Path, old: Path, n_per_side: int) -> tuple[int, int, int]:
    """new vs old を両 side で対戦させ、new 視点の (win, loss, draw) を返す。"""
    win = loss = draw = 0
    for new_is_p1 in (True, False):
        a, b = (new, old) if new_is_p1 else (old, new)
        out = run([
            "uv", "run", "python", "-m", "poke_ai3_train.eval_ckpt_vs_ckpt",
            "--checkpoint-a", str(a), "--checkpoint-b", str(b),
            "--num-games", "16", "--num-eval-games", str(n_per_side), *SIM,
        ])
        m = _RESULT.search(out)
        if m is None:
            raise SystemExit(f"RESULT 行が見つからない:\n{out[-1000:]}")
        a_win, b_win, d = int(m[1]), int(m[2]), int(m[3])
        # new_is_p1 なら new=A、そうでなければ new=B。
        win += a_win if new_is_p1 else b_win
        loss += b_win if new_is_p1 else a_win
        draw += d
    return win, loss, draw


def train_round(ckpt: Path, target_epochs: int) -> None:
    run([
        "uv", "run", "train-loop", "--num-games", "32", *SIM,
        "--max-epochs", str(target_epochs),
        "--checkpoint-path", str(ckpt),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=Path, required=True, help="開始 checkpoint (training_step を継承)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs-per-round", type=int, default=5)
    ap.add_argument("--pool-size", type=int, default=5, help="gauntlet プールの最大 K")
    ap.add_argument("--n-per-side", type=int, default=128, help="1 対戦あたりの片側試合数")
    ap.add_argument("--seed-pool", type=Path, nargs="*", default=[],
                    help="初期プールに入れる過去の強い checkpoint 群")
    ap.add_argument("--tag", type=str, default="g", help="作業 checkpoint 名の接頭辞")
    args = ap.parse_args()

    GDIR.mkdir(parents=True, exist_ok=True)
    import shutil
    ckpt = GDIR / f"{args.tag}_work.pt"
    shutil.copy(args.start, ckpt)
    # 開始時点の training_step を読む。
    import torch
    base_step = int(torch.load(ckpt, map_location="cpu").get("training_step", 0))
    print(f"start={args.start} base_step={base_step} rounds={args.rounds} "
          f"epochs/round={args.epochs_per_round} pool_size={args.pool_size} "
          f"n_per_side={args.n_per_side}")

    pool: list[Path] = list(args.seed_pool)
    table: list[str] = []
    for r in range(1, args.rounds + 1):
        target = base_step + r * args.epochs_per_round
        print(f"\n########## ROUND {r}: train -> ep{target} ##########", flush=True)
        train_round(ckpt, target)
        snap = GDIR / f"{args.tag}_ep{target}.pt"
        shutil.copy(ckpt, snap)
        print(f"new snapshot: {snap.name}  gauntlet pool: {[p.stem for p in pool]}", flush=True)

        all_beat = True
        for old in pool:
            w, l, d = head_to_head(snap, old, args.n_per_side)
            n = w + l + d
            wr = w / n if n else 0.0
            mark = "○" if wr > 0.5 else "●"
            line = f"  {mark} {snap.stem} vs {old.stem}: 勝率={wr:.3f} (W={w} L={l} D={d}, n={n})"
            print(line, flush=True)
            table.append(line)
            if wr <= 0.5:
                all_beat = False
        verdict = "全勝(>0.5)" if all_beat else "★敗北あり(<=0.5)"
        head = f"ROUND {r} (ep{target}) 仮説: {verdict}"
        print(f"==> {head}", flush=True)
        table.append(f"=== {head} ===")

        pool.append(snap)
        pool = pool[-args.pool_size:]

    print("\n\n================ SUMMARY ================")
    for line in table:
        print(line)


if __name__ == "__main__":
    main()
