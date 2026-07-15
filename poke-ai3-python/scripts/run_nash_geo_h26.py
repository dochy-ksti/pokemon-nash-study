"""幾何打ち切りゲーム (毎手 1-γ で終了・タイブレーク払い) の厳密 Nash を H=26 で解く。

solve_nash_backward(discount=0.99) は幾何打ち切りゲームの定常 Shapley 解を計算し、
同じ作用素で policy_eval / best_response して exploitability (BR gap) を検証する。
結果テーブル (policy/value/br, u16) は npz で保存し、web 形式への書き出しは別途行う。

使い方:
    cd poke-ai3-python
    setsid nohup uv run python scripts/run_nash_geo_h26.py --stage 3c \
      > /tmp/psro/nash_geo_h26_3c.log 2>&1 &
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from poke_ai3._native import solve_nash_cached, solve_nash_cached_full

H = 26
OUT = Path(__file__).resolve().parents[2] / "data" / "poke-ai3" / "nash_geo"
OUT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("3b", "3c", "3d", "3e"), required=True)
    parser.add_argument("--hp-buckets", type=int, default=H)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage
    h = args.hp_buckets
    output = OUT / f"nash_geo_h{h}_{stage}.npz"
    t0 = time.time()
    print(f"[h{h}:{stage}] start H={h} discount=0.99 (cached)", flush=True)
    # 3d/3e は各個体が3技 (Crunch/Dark Pulse/coverage) を持つので4行動の完全方策を解く。
    full = stage in ("3d", "3e")
    solver = solve_nash_cached_full if full else solve_nash_cached
    if full:
        pol, val, br, em, ex = solver(
            stage, h, True, True, 3000, 0.99, 5e-6, 1e-6, 3000, True
        )
    else:
        pol, val, br, em, ex = solver(
            stage, h, True, True, 3000, 0.99, 1e-6, 3000, True
        )
    pol = np.array(pol, dtype=np.uint16)
    val = np.array(val, dtype=np.uint16)
    br = np.array(br, dtype=np.uint16)
    np.savez_compressed(
        output,
        policy=pol, value=val, br=br,
        exploit_mean=em, exploit_max=ex, hp_buckets=h, discount=0.99,
        action_order=np.array(["Crunch", "Dark Pulse", "coverage", "Switch"]),
    )
    print(f"[h{h}:{stage}] saved -> {output}", flush=True)
    print(f"[h{h}] exploit mean={em:.5f} max={ex:.5f}", flush=True)
    def state_idx(t: int, aa: int, ac: int, ag: int, oa: int, oc: int, og: int) -> int:
        k = t
        for radix, value in ((2, aa), (h, ac), (h, ag), (2, oa), (h, oc), (h, og)):
            k = k * radix + value
        return k
    s0 = state_idx(0, 0, h - 1, h - 1, 0, h - 1, h - 1)
    s1 = state_idx(1, 0, h - 1, h - 1, 0, h - 1, h - 1)
    for name, s in (("team0", s0), ("team1", s1)):
        if full:
            p = pol.reshape(-1, 4)[s].astype(float) / 1000.0
            print(f"[h{h}] start {name}: V={val[s]/1000:.3f} BR={br[s]/1000:.3f} "
                  f"policy={p.tolist()}", flush=True)
        else:
            print(f"[h{h}] start {name}: V={val[s]/1000:.3f} BR={br[s]/1000:.3f} "
                  f"pswitch={pol[s]/1000:.3f}", flush=True)
    if full:
        p4 = pol.reshape(-1, 4)
        valid = p4[:, 0] != 0xFFFF
        ps = p4[valid, 3].astype(float) / 1000.0
        mean = p4[valid].astype(float).mean(axis=0) / 1000.0
        print(f"[h{h}] mean policy={mean.tolist()}", flush=True)
    else:
        valid = pol != 0xFFFF
        ps = pol[valid].astype(float) / 1000.0
    hist = np.histogram(ps, bins=[0, .001, .05, .2, .5, .8, .95, .999, 1.001])[0]
    print(f"[h{h}] valid={int(valid.sum())} pswitch hist={hist.tolist()}", flush=True)
    print(f"[h{h}] mixed-switch(.02-.98)={int(((ps > .02) & (ps < .98)).sum())}", flush=True)
    print(f"[h{h}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
