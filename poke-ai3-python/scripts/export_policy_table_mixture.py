"""静的Web対戦アプリ用のポリシー/勝率テーブルを **σ 混合 (メタ Nash)** から前計算する。

単一 checkpoint 版 (export_policy_table.py) の混合戦略版。PSRO の結果 JSON が持つ
集団 Π (中心スナップショット列) と σ 混合を読み、各状態で全サポートネットを推論して
σ 加重平均した P(交代) と value を u16 で焼く。Web app はこの 1 枚のテーブルを状態ごとに
確率サンプリングするので、σ 加重平均テーブル = 混合戦略の (状態単位の) 自然な表現になる。

裾の極小重みネットは --sigma-floor で刈って再正規化する (コスト = サポート数に比例)。

使い方の例:
    uv run python scripts/export_policy_table_mixture.py \
        --psro-json ../data/poke-ai3/tournament/PSRO_nash3_psro.json \
        --sigma-floor 0.005 --stage 3b --hp-buckets 26 --out-dir ../web
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
PROB_SCALE = 1000   # P(交代) を 0..1000 (0.1%刻み) で格納
VALUE_SCALE = 1000  # value(勝率) を 0..1000 で格納


def _slice_obs(obs: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    return {k: v[lo:hi] for k, v in obs.items()}


def export_mixture(
    agents: list[Agent],
    weights: np.ndarray,
    stage: str,
    hp_buckets: int,
    dense_batch: int,
    infer_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """σ 加重平均した P(交代)/value テーブル (u16) を返す。"""
    total = 8 * hp_buckets**4
    ptable = np.full(total, SENTINEL, dtype=np.uint16)
    vtable = np.full(total, SENTINEL, dtype=np.uint16)
    done = 0
    for start in range(0, total, dense_batch):
        count = min(dense_batch, total - start)
        obs, indices = enumerate_policy_batch(stage, hp_buckets, start, count)
        idx = np.asarray(indices, dtype=np.int64)
        n = idx.shape[0]
        for lo in range(0, n, infer_chunk):
            hi = min(lo + infer_chunk, n)
            sub = _slice_obs(obs, lo, hi)
            p_acc = np.zeros(hi - lo, dtype=np.float64)
            v_acc = np.zeros(hi - lo, dtype=np.float64)
            for agent, w in zip(agents, weights):
                policy, value = agent.infer_encoded(sub)
                p_acc += w * policy[:, MAX_MOVE_SLOTS:].sum(axis=1)
                v_acc += w * np.asarray(value).reshape(-1)
            pv = np.rint(p_acc * PROB_SCALE).clip(0, PROB_SCALE).astype(np.uint16)
            vv = np.rint(v_acc * VALUE_SCALE).clip(0, VALUE_SCALE).astype(np.uint16)
            ptable[idx[lo:hi]] = pv
            vtable[idx[lo:hi]] = vv
        done += count
        print(f"  {done}/{total} ({100 * done / total:.1f}%) valid_in_batch={n}", flush=True)
    return ptable, vtable


def main() -> None:
    ap = argparse.ArgumentParser(description="export σ-mixture policy/value tables for the web app")
    ap.add_argument("--psro-json", type=Path, required=True, help="PSRO 結果 JSON (pool + sigma)")
    ap.add_argument("--sigma-floor", type=float, default=0.005,
                    help="この重み未満のサポートは刈って再正規化 (既定 0.005)")
    ap.add_argument("--stage", default="3b")
    ap.add_argument("--hp-buckets", type=int, default=26, help="HP離散化段数 (4%%→26)")
    ap.add_argument("--out-dir", type=Path, default=Path("../web"))
    ap.add_argument("--dense-batch", type=int, default=32768)
    ap.add_argument("--infer-chunk", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = json.loads(args.psro_json.read_text())
    pool = [Path(p) for p in d["pool"]]
    sigma = np.asarray(d["sigma"], dtype=np.float64)
    keep = np.where(sigma >= args.sigma_floor)[0]
    if keep.size == 0:
        raise SystemExit("no support net above sigma-floor")
    weights = sigma[keep] / sigma[keep].sum()
    members = [pool[i] for i in keep]
    cover = float(sigma[keep].sum())
    print(f"σ サポート {keep.size} 体 (floor={args.sigma_floor}, cover={cover:.4f}):", flush=True)
    for i, w in zip(keep, weights):
        print(f"  {pool[i].stem}: σ={sigma[i]:.4f} -> w={w:.4f}", flush=True)
    for m in members:
        if not m.exists():
            raise SystemExit(f"checkpoint not found: {m}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agents = [Agent(device=args.device, checkpoint_path=m, infer_graph=False) for m in members]
    ptable, vtable = export_mixture(
        agents, weights, args.stage, args.hp_buckets, args.dense_batch, args.infer_chunk
    )

    bin_path = args.out_dir / f"policy_{args.stage}.bin"
    vbin_path = args.out_dir / f"value_{args.stage}.bin"
    meta_path = args.out_dir / f"policy_{args.stage}.meta.json"
    ptable.tofile(bin_path)
    vtable.tofile(vbin_path)

    h = args.hp_buckets
    meta = {
        "stage": args.stage,
        "hp_buckets": h,
        "prob_scale": PROB_SCALE,
        "value_scale": VALUE_SCALE,
        "sentinel": SENTINEL,
        "max_move_slots": int(MAX_MOVE_SLOTS),
        "checkpoint": f"{args.psro_json.stem}_sigma_mix{keep.size}",
        "mixture": {
            "psro_json": args.psro_json.name,
            "sigma_floor": args.sigma_floor,
            "cover": cover,
            "members": [
                {"ckpt": pool[int(i)].stem, "sigma": float(sigma[int(i)]), "weight": float(w)}
                for i, w in zip(keep, weights)
            ],
        },
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
        "total": int(ptable.shape[0]),
        "species_order": ["Cloyster", "Goodra-Hisui" if args.stage == "3b" else "Goodra"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    n_valid = int((ptable != SENTINEL).sum())
    print(f"wrote {bin_path} ({ptable.nbytes} bytes, {n_valid} valid / {ptable.shape[0]})")
    print(f"wrote {vbin_path}")
    print(f"wrote {meta_path}")
    _ = poke_ai3


if __name__ == "__main__":
    main()
