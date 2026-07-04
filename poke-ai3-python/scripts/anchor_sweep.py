#!/usr/bin/env python
"""固定アンカー checkpoint に対する各世代スナップショットの勝率トレンドを測る。

gauntlet (直近 5 プール) は基準がドリフトするため「ep140 以降弱体化しているか」の判定に
向かない。本ツールは固定の anchor (例 ep105/ep130) に対し、指定スナップショット群を順に
両 side 対戦させ (各 n=256)、anchor 視点でなく「被験 ckpt 視点の勝率」を epoch 順に出す。
勝率が単調に下がるなら弱体化が本物。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SIM = [
    "--sim-concurrency", "16", "--sims", "64",
    "--search-turn-min", "6", "--search-turn-max", "12",
    "--no-random", "--no-crit", "--stage", "3b",
]
_RESULT = re.compile(r"RESULT a_win=(\d+) b_win=(\d+) draw=(\d+)")


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-1500:] + "\n" + proc.stderr[-1500:] + "\n")
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return proc.stdout


def h2h(sub: Path, anchor: Path, n_side: int) -> tuple[int, int, int]:
    win = loss = draw = 0
    for sub_is_p1 in (True, False):
        a, b = (sub, anchor) if sub_is_p1 else (anchor, sub)
        out = run([
            "uv", "run", "python", "-m", "poke_ai3_train.eval_ckpt_vs_ckpt",
            "--checkpoint-a", str(a), "--checkpoint-b", str(b),
            "--num-games", "16", "--num-eval-games", str(n_side), *SIM,
        ])
        m = _RESULT.search(out)
        a_win, b_win, d = int(m[1]), int(m[2]), int(m[3])
        win += a_win if sub_is_p1 else b_win
        loss += b_win if sub_is_p1 else a_win
        draw += d
    return win, loss, draw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=Path, nargs="+", required=True)
    ap.add_argument("--subjects", type=Path, nargs="+", required=True)
    ap.add_argument("--n-per-side", type=int, default=128)
    args = ap.parse_args()

    rows: list[str] = []
    for anchor in args.anchors:
        print(f"\n===== anchor: {anchor.stem} =====", flush=True)
        for sub in args.subjects:
            w, l, d = h2h(sub, anchor, args.n_per_side)
            n = w + l + d
            wr = w / n if n else 0.0
            line = f"  {sub.stem} vs {anchor.stem}: 勝率={wr:.3f} (W={w} L={l} D={d}, n={n})"
            print(line, flush=True)
            rows.append(line)

    print("\n================ SUMMARY ================")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
