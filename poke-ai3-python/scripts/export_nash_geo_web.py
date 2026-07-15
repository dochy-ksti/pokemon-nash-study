"""幾何打ち切りゲームの厳密 Nash テーブルを web 形式へ書き出す。

policy（3b/3cはP(交代)、3d/3eは4行動完全方策）と value（均衡値 V）を u16 で書き、
meta.json を更新する。dense index / radix / sentinel は
既存 policy_table と同一なので配列はそのまま tofile する。

※ deploy 側のゲームは打ち切りルールを入れない方針。value は幾何打ち切りゲーム
   (γ=0.99) の均衡値なので実ゲームとは僅かに異なる (AI 自身の一貫推定値として表示)。

使い方:
    cd poke-ai3-python
    uv run python scripts/export_nash_geo_web.py --stage 3c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "poke-ai3" / "nash_geo"
WEB = ROOT / "web"
H = 26
SENTINEL = 0xFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("3b", "3c", "3d", "3e"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage
    species_order = {
        "3b": ["Cloyster", "Goodra-Hisui"],
        "3c": ["Cloyster", "Goodra"],
        "3d": ["Cloyster", "Goodra-Hisui"],
        "3e": ["Cloyster", "Goodra"],
    }[stage]
    npz = DATA / f"nash_geo_h26_{stage}.npz"
    if stage == "3b" and not npz.exists():
        npz = DATA / "nash_geo_h26.npz"
    d = np.load(npz)
    policy = np.asarray(d["policy"], dtype=np.uint16)
    value = np.asarray(d["value"], dtype=np.uint16)
    total = 8 * H**4
    # 3d/3e は各個体3技なので4行動の完全方策。3b/3c は P(交代) の1値。
    full = stage in ("3d", "3e")
    policy_width = 4 if full else 1
    expected_policy_shape = (total * policy_width,)
    assert policy.shape == expected_policy_shape, (policy.shape, expected_policy_shape)
    assert value.shape == (total,)

    (WEB / f"policy_{stage}.bin").write_bytes(policy.tobytes())
    (WEB / f"value_{stage}.bin").write_bytes(value.tobytes())

    meta = {
        "stage": stage,
        "hp_buckets": H,
        "prob_scale": 1000,
        "value_scale": 1000,
        "sentinel": SENTINEL,
        "max_move_slots": 4,
        "policy_width": policy_width,
        "policy_actions": (
            ["crunch", "darkpulse", "coverage", "switch"]
            if full else ["switch"]
        ),
        "source": "nash_geo_backward",
        "solver": {
            "method": "geometric-cutoff stationary Nash (Shapley) via cached backward VI",
            "discount": float(d["discount"]),
            "hp_buckets": H,
            "tiebreak": "alive-count then summed HP-fraction",
            "exploit_mean": float(d["exploit_mean"]),
            "exploit_max": float(d["exploit_max"]),
            "note": "deployed WITHOUT cutoff rule; table is exact Nash of the "
                    "geometric-cutoff game (in-game BR gap ~0). value column = "
                    "equilibrium win-prob V of the geometric game.",
        },
        "cross_team_only": True,
        "radix": [
            {"name": "ai_team", "size": 2},
            {"name": "ai_active", "size": 2},
            {"name": "ai_hp_cloyster", "size": H},
            {"name": "ai_hp_goodra", "size": H},
            {"name": "opp_active", "size": 2},
            {"name": "opp_hp_cloyster", "size": H},
            {"name": "opp_hp_goodra", "size": H},
        ],
        "total": total,
        "species_order": species_order,
    }
    (WEB / f"policy_{stage}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )

    rows = policy.reshape(total, policy_width)
    valid = rows[:, 0] != SENTINEL
    print(
        f"wrote web/policy_{stage}.bin ({policy.nbytes} bytes), "
        f"value_{stage}.bin ({value.nbytes} bytes)"
    )
    print(f"valid={int(valid.sum())} exploit mean={float(d['exploit_mean']):.5f} "
          f"max={float(d['exploit_max']):.5f}")
    probs = rows[valid].astype(float) / 1000.0
    if full:
        sums = rows[valid].astype(np.int64).sum(axis=1)
        assert int(sums.min()) >= 999 and int(sums.max()) <= 1001
        print(f"mean policy={probs.mean(axis=0).tolist()}")
        print(f"mixed={int(((probs > .02).sum(axis=1) >= 2).sum())}")
    else:
        ps = probs[:, 0]
        print(f"mixed(.02-.98)={int(((ps > .02) & (ps < .98)).sum())}")


if __name__ == "__main__":
    main()
