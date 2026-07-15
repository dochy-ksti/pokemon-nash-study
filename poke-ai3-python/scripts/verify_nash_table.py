"""厳密 Nash テーブル (nash_geo_h26_*.npz) の整合性を検証する。

検査項目:

1. 方策の量子化和が 999〜1001 に収まる (完全方策のみ)。
2. 控えが瀕死の状態で交代確率が非0でない。
3. ゼロ和ミラー恒等式 `V(s) + V(swap(s)) = 1`。
   `swap(s)` は P1↔P2 を入れ替えた状態で、cross-team only なので team も反転する。
   3c/3e は完全対称ゲームなので、この検査は解の正しさを強く保証する
   (3b/3d は非対称だが、swap は「相手側から見た同じ盤面」なので恒等式自体は成立する)。

使い方:
    cd poke-ai3-python
    uv run python scripts/verify_nash_table.py --stage 3e
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "poke-ai3" / "nash_geo"
SENTINEL = 0xFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--hp-buckets", type=int, default=26)
    return parser.parse_args()


def decode(k: np.ndarray, h: int) -> tuple[np.ndarray, ...]:
    """dense index を (team, aa, ac, ag, oa, oc, og) へ分解する。"""
    out = []
    for radix in (h, h, 2, h, h, 2, 2):
        out.append(k % radix)
        k = k // radix
    return tuple(reversed(out))


def encode(parts: tuple[np.ndarray, ...], h: int) -> np.ndarray:
    """(team, aa, ac, ag, oa, oc, og) を dense index へ。"""
    team, aa, ac, ag, oa, oc, og = parts
    k = team
    for radix, value in ((2, aa), (h, ac), (h, ag), (2, oa), (h, oc), (h, og)):
        k = k * radix + value
    return k


def main() -> None:
    args = parse_args()
    h = args.hp_buckets
    data = np.load(DATA / f"nash_geo_h26_{args.stage}.npz")
    value = np.asarray(data["value"], dtype=np.int64)
    policy = np.asarray(data["policy"], dtype=np.int64)
    total = value.size
    full = policy.size == total * 4
    rows = policy.reshape(total, 4 if full else 1)
    valid = rows[:, 0] != SENTINEL
    print(f"stage={args.stage} total={total} valid={int(valid.sum())} full={full}")

    ok = True

    if full:
        sums = rows[valid].sum(axis=1)
        lo, hi = int(sums.min()), int(sums.max())
        good = lo >= 999 and hi <= 1001
        ok &= good
        print(f"[{'ok' if good else 'NG'}] policy sum range = {lo}..{1001 if hi > 1001 else hi}"
              f" (expect 999..1001), actual max={hi}")

    # 2. 控えが瀕死なら交代不可。交代確率スロットは full なら 3、単値なら 0。
    team, aa, ac, ag, oa, oc, og = decode(np.arange(total), h)
    bench_hp = np.where(aa == 0, ag, ac)  # active が index0(Cloyster) なら控えは Goodra
    pswitch = rows[:, 3] if full else rows[:, 0]
    bad = valid & (bench_hp == 0) & (pswitch != 0)
    ok &= not bad.any()
    print(f"[{'ok' if not bad.any() else 'NG'}] switch prob != 0 while bench fainted: "
          f"{int(bad.sum())} states (expect 0)")

    # 3. ミラー恒等式 V(s) + V(swap(s)) = 1。
    swapped = encode((1 - team, oa, oc, og, aa, ac, ag), h)
    both = valid & valid[swapped]
    resid = value[both] + value[swapped[both]] - 1000
    max_err, mean_err = int(np.abs(resid).max()), float(np.abs(resid).mean())
    good = max_err <= 1
    ok &= good
    print(f"[{'ok' if good else 'NG'}] mirror |V(s)+V(swap(s))-1|: pairs={int(both.sum())} "
          f"max={max_err} (={max_err / 1000:.3f}) mean={mean_err:.5f} "
          f"(={mean_err / 1000:.8f}); expect max<=1 u16 step")

    print(f"exploit mean={float(data['exploit_mean']):.8f} max={float(data['exploit_max']):.8f}")
    print("RESULT:", "ok" if ok else "NG")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
