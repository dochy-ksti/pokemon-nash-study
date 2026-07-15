"""幾何打ち切りゲーム (毎手 1-γ で終了・タイブレーク払い) の厳密 Nash を H=26 で解く。

solve_nash_backward(discount=0.99) は幾何打ち切りゲームの定常 Shapley 解を計算し、
同じ作用素で policy_eval / best_response して exploitability (BR gap) を検証する。
結果テーブル (policy/value/br, u16) は npz で保存し、web 形式への書き出しは別途行う。

使い方:
    cd poke-ai3-python
    setsid nohup uv run python scripts/run_nash_geo_h26.py > /tmp/psro/nash_geo_h26.log 2>&1 &
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from poke_ai3._native import solve_nash_cached

H = 26
OUT = Path(__file__).resolve().parents[2] / "data" / "poke-ai3" / "nash_geo"
OUT.mkdir(parents=True, exist_ok=True)


def idx(t: int, aa: int, ac: int, ag: int, oa: int, oc: int, og: int) -> int:
    k = t
    k = k * 2 + aa
    k = k * H + ac
    k = k * H + ag
    k = k * 2 + oa
    k = k * H + oc
    k = k * H + og
    return k


def main() -> None:
    t0 = time.time()
    print(f"[h26] start H={H} discount=0.99 (cached)", flush=True)
    pol, val, br, em, ex = solve_nash_cached(
        "3b", H, True, True, 3000, 0.99, 1e-6, 3000, True
    )
    pol = np.array(pol, dtype=np.uint16)
    val = np.array(val, dtype=np.uint16)
    br = np.array(br, dtype=np.uint16)
    np.savez_compressed(
        OUT / "nash_geo_h26.npz",
        policy=pol, value=val, br=br,
        exploit_mean=em, exploit_max=ex, hp_buckets=H, discount=0.99,
    )
    print(f"[h26] saved -> {OUT / 'nash_geo_h26.npz'}", flush=True)
    print(f"[h26] exploit mean={em:.5f} max={ex:.5f}", flush=True)
    s0 = idx(0, 0, H - 1, H - 1, 0, H - 1, H - 1)
    s1 = idx(1, 0, H - 1, H - 1, 0, H - 1, H - 1)
    for name, s in (("team0", s0), ("team1", s1)):
        print(f"[h26] start {name}: V={val[s]/1000:.3f} BR={br[s]/1000:.3f} "
              f"pswitch={pol[s]/1000:.3f}", flush=True)
    valid = pol != 0xFFFF
    ps = pol[valid].astype(float) / 1000.0
    hist = np.histogram(ps, bins=[0, .001, .05, .2, .5, .8, .95, .999, 1.001])[0]
    print(f"[h26] valid={int(valid.sum())} pswitch hist={hist.tolist()}", flush=True)
    print(f"[h26] mixed(.02-.98)={int(((ps > .02) & (ps < .98)).sum())}", flush=True)
    print(f"[h26] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
