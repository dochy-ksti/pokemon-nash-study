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
# u8 方策の量子化スケールと番兵。有効行の値は 0..254 に収まるので 255 と衝突しない。
U8_SCALE = 254
U8_SENTINEL = 255


def quantize_u8(rows: np.ndarray) -> np.ndarray:
    """完全方策 (u16, 0..1000) を u8 (0..254、番兵 255) へ量子化する。

    有効状態の各行は和が厳密に `U8_SCALE` になるよう最大剰余法で丸める。単純な四捨五入
    では和が 253〜255 に散り、255 が番兵と衝突しうる。確率が厳密に 0 の行動には増分を
    配らないので、「控えが瀕死なら交代確率 0」といった構造的な 0 は保存される。
    """
    valid = rows[:, 0] != SENTINEL
    out = np.full(rows.shape, U8_SENTINEL, dtype=np.uint8)
    v = rows[valid].astype(np.float64)
    v /= v.sum(axis=1, keepdims=True)  # 量子化和 999〜1001 を正規化してから配る

    target = v * U8_SCALE
    floor = np.floor(target).astype(np.int64)
    deficit = U8_SCALE - floor.sum(axis=1)  # 各行に配るべき残り (0..width-1)

    # 剰余の大きい順に +1。確率 0 の行動は候補から外す (剰余 -1 で最後尾へ落とす)。
    rem = np.where(v > 0.0, target - floor, -1.0)
    order = np.argsort(-rem, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(rows.shape[1])[None, :], axis=1)
    floor += (ranks < deficit[:, None]).astype(np.int64)

    assert (floor.sum(axis=1) == U8_SCALE).all(), "u8 量子化の行和が一致しない"
    assert floor.max() <= U8_SCALE and floor.min() >= 0
    assert not ((v == 0.0) & (floor != 0)).any(), "確率 0 の行動に増分が配られた"
    out[valid] = floor.astype(np.uint8)
    return out


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

    # 完全方策 (u16×4) は 27.9MiB になり Cloudflare Pages の 1ファイル 25MiB 上限を
    # 超えてデプロイ全体を失敗させる。u8 へ量子化して 7.0MiB に落とす。
    # 3b/3c の P(交代) は 7.0MiB なので u16 のまま据え置く。
    if full:
        out_policy = quantize_u8(policy.reshape(total, policy_width))
        prob_scale, sentinel, dtype = U8_SCALE, U8_SENTINEL, "u8"
    else:
        out_policy = policy
        prob_scale, sentinel, dtype = 1000, SENTINEL, "u16"

    (WEB / f"policy_{stage}.bin").write_bytes(out_policy.tobytes())
    (WEB / f"value_{stage}.bin").write_bytes(value.tobytes())

    meta = {
        "stage": stage,
        "hp_buckets": H,
        "prob_scale": prob_scale,
        "value_scale": 1000,
        "sentinel": sentinel,
        "policy_dtype": dtype,
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

    rows = out_policy.reshape(total, policy_width)
    valid = rows[:, 0] != sentinel
    mib = out_policy.nbytes / 1048576
    print(
        f"wrote web/policy_{stage}.bin ({out_policy.nbytes} bytes = {mib:.2f} MiB, "
        f"{dtype}), value_{stage}.bin ({value.nbytes} bytes)"
    )
    # Cloudflare Pages は 1 ファイル 25MiB 超のアセットがあるとデプロイ全体を失敗させる。
    assert mib < 25.0, f"policy_{stage}.bin が {mib:.2f} MiB で Pages の 25MiB 上限を超える"
    print(f"valid={int(valid.sum())} exploit mean={float(d['exploit_mean']):.5f} "
          f"max={float(d['exploit_max']):.5f}")
    probs = rows[valid].astype(float) / prob_scale
    if full:
        sums = rows[valid].astype(np.int64).sum(axis=1)
        assert (sums == prob_scale).all(), "u8 方策の行和が scale と一致しない"
        # 量子化誤差が方策を壊していないか、元の u16 と直接突き合わせる。
        ref = policy.reshape(total, policy_width)[valid].astype(float)
        ref /= ref.sum(axis=1, keepdims=True)
        err = np.abs(probs - ref)
        print(f"mean policy={probs.mean(axis=0).tolist()}")
        print(f"mixed={int(((probs > .02).sum(axis=1) >= 2).sum())}")
        print(f"u8 量子化誤差: max={err.max():.5f} mean={err.mean():.6f} "
              f"(1/{prob_scale}={1 / prob_scale:.5f})")
        assert err.max() <= 1.0 / prob_scale, "量子化誤差が 1 刻みを超えた"
    else:
        ps = probs[:, 0]
        print(f"mixed(.02-.98)={int(((ps > .02) & (ps < .98)).sum())}")


if __name__ == "__main__":
    main()
