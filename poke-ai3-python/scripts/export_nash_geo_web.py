"""幾何打ち切りゲームの厳密 Nash テーブル (nash_geo_h26.npz) を web 形式へ書き出す。

policy (P(交代)) と value (勝率=均衡値 V) を u16 のまま web/policy_3b.bin /
web/value_3b.bin に書き、meta.json を更新する。dense index / radix / sentinel は
既存 policy_table と同一なので配列はそのまま tofile する。

※ deploy 側のゲームは打ち切りルールを入れない方針。value は幾何打ち切りゲーム
   (γ=0.99) の均衡値なので実ゲームとは僅かに異なる (AI 自身の一貫推定値として表示)。

使い方:
    cd poke-ai3-python
    uv run python scripts/export_nash_geo_web.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data" / "poke-ai3" / "nash_geo" / "nash_geo_h26.npz"
WEB = ROOT / "web"
H = 26
SENTINEL = 0xFFFF


def main() -> None:
    d = np.load(NPZ)
    policy = np.asarray(d["policy"], dtype=np.uint16)
    value = np.asarray(d["value"], dtype=np.uint16)
    total = 8 * H**4
    assert policy.shape == (total,), (policy.shape, total)
    assert value.shape == (total,)

    (WEB / "policy_3b.bin").write_bytes(policy.tobytes())
    (WEB / "value_3b.bin").write_bytes(value.tobytes())

    meta = {
        "stage": "3b",
        "hp_buckets": H,
        "prob_scale": 1000,
        "value_scale": 1000,
        "sentinel": SENTINEL,
        "max_move_slots": 4,
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
        "species_order": ["Cloyster", "Goodra-Hisui"],
    }
    (WEB / "policy_3b.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    valid = policy != SENTINEL
    print(f"wrote web/policy_3b.bin, value_3b.bin ({policy.nbytes} bytes each)")
    print(f"valid={int(valid.sum())} exploit mean={float(d['exploit_mean']):.5f} "
          f"max={float(d['exploit_max']):.5f}")
    ps = policy[valid].astype(float) / 1000.0
    print(f"mixed(.02-.98)={int(((ps > .02) & (ps < .98)).sum())}")


if __name__ == "__main__":
    main()
