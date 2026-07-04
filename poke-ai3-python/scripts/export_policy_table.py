"""静的Web対戦アプリ用のポリシーテーブルを前計算して書き出す。

3b/3c の全状態 (両サイド2体の各HPを `hp_buckets` 段に離散化 + アクティブ + チーム構成) を
正準列挙順で回し、AI(=P1視点)の P(交代) を u16 (0..1000, 0.1%精度) で密配列へ焼く。
runtime(JS)は同じ index 式でテーブルを引く。無効(アクティブ瀕死)は番兵 0xFFFF。

使い方の例:
    uv run python scripts/export_policy_table.py \
        --checkpoint ../data/poke-ai3/tournament/K1b5_ep280.pt \
        --stage 3b --hp-buckets 26 --out-dir ../web
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import poke_ai3
from poke_ai3 import MAX_MOVE_SLOTS, enumerate_policy_batch
from poke_ai3_train.agent import Agent

SENTINEL = 0xFFFF
PROB_SCALE = 1000  # P(交代) を 0..1000 (0.1%刻み) で格納


def _slice_obs(obs: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    """エンコード済み観測 dict の先頭次元を [lo, hi) にスライスする。"""
    return {k: v[lo:hi] for k, v in obs.items()}


def export(
    agent: Agent,
    stage: str,
    hp_buckets: int,
    dense_batch: int,
    infer_chunk: int,
) -> np.ndarray:
    total = 8 * hp_buckets**4
    table = np.full(total, SENTINEL, dtype=np.uint16)
    done = 0
    for start in range(0, total, dense_batch):
        count = min(dense_batch, total - start)
        obs, indices = enumerate_policy_batch(stage, hp_buckets, start, count)
        idx = np.asarray(indices, dtype=np.int64)
        n = idx.shape[0]
        for lo in range(0, n, infer_chunk):
            hi = min(lo + infer_chunk, n)
            policy, _ = agent.infer_encoded(_slice_obs(obs, lo, hi))
            p_switch = policy[:, MAX_MOVE_SLOTS:].sum(axis=1)
            vals = np.rint(p_switch * PROB_SCALE).clip(0, PROB_SCALE).astype(np.uint16)
            table[idx[lo:hi]] = vals
        done += count
        print(f"  {done}/{total} ({100 * done / total:.1f}%) valid_in_batch={n}", flush=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="export policy lookup table for the web app")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--stage", default="3b")
    ap.add_argument("--hp-buckets", type=int, default=26, help="HP離散化段数 (4%%→26)")
    ap.add_argument("--out-dir", type=Path, default=Path("../web"))
    ap.add_argument("--dense-batch", type=int, default=32768, help="密index列挙の1回あたり")
    ap.add_argument("--infer-chunk", type=int, default=16384, help="GPU推論の1回あたり")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent(device=args.device, checkpoint_path=args.checkpoint, infer_graph=False)
    table = export(agent, args.stage, args.hp_buckets, args.dense_batch, args.infer_chunk)

    bin_path = args.out_dir / f"policy_{args.stage}.bin"
    meta_path = args.out_dir / f"policy_{args.stage}.meta.json"
    table.tofile(bin_path)

    h = args.hp_buckets
    meta = {
        "stage": args.stage,
        "hp_buckets": h,
        "prob_scale": PROB_SCALE,
        "sentinel": SENTINEL,
        "max_move_slots": int(MAX_MOVE_SLOTS),
        "checkpoint": args.checkpoint.stem,
        # canonical order (most significant first)。JS はこの radix で mixed-radix index を組む。
        # クロスチーム限定なので opp_team = 1 - ai_team は次元に持たない。
        "cross_team_only": True,
        "radix": [
            {"name": "ai_team", "size": 2},
            {"name": "ai_active", "size": 2},
            {"name": "ai_hp_cloyster", "size": h},
            {"name": "ai_hp_goodra", "size": h},
            {"name": "opp_active", "size": 2},
            {"name": "opp_hp_cloyster", "size": h},
            {"name": "opp_hp_goodra", "size": h},
        ],
        "total": int(table.shape[0]),
        "species_order": ["Cloyster", "Goodra-Hisui" if args.stage == "3b" else "Goodra"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    n_valid = int((table != SENTINEL).sum())
    print(f"wrote {bin_path} ({table.nbytes} bytes, {n_valid} valid / {table.shape[0]})")
    print(f"wrote {meta_path}")
    _ = poke_ai3  # keep import (native freshness)


if __name__ == "__main__":
    main()
