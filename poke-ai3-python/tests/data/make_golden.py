"""観測エンコード (Rust encode_batch) の出力をゴールデン JSON として固定する。

初版は移行前の純 Python 実装の出力で固定した。相手控え観測拡張
(opp_move_* / opp_bench_*) でレイアウトが変わったため、拡張後の Rust 実装の
出力で再生成した (以後の回帰の基準):
    cd poke-ai3-python && uv run python tests/data/make_golden.py
出力: tests/data/encoding_golden.json
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch

from poke_ai3 import ACTION_DIM, NUM_BENCH, NUM_MOVES
from poke_ai3_train.encoding import encode_observations


def sample_states() -> list[dict]:
    # obs_encode.rs のテストと同系統: 控え 1 体 + 空き枠、技 2 つ + 0 埋め。
    mask1 = [False] * ACTION_DIM
    mask1[0] = True
    mask1[NUM_MOVES] = True
    bench1 = [None] * NUM_BENCH
    bench1[0] = {"species_gid": 9, "hp_frac": 0.25, "move_gids": [5]}
    opp_bench1 = [None] * NUM_BENCH
    opp_bench1[0] = {"species_gid": 13, "hp_frac": 0.5, "move_gids": [8, 9]}
    state1 = {
        "my_species_gid": 7,
        "opp_species_gid": 11,
        "my_exact_hp_frac": 0.75,
        "opp_quantized_hp_frac": 0.5,
        "my_move_gids": [3, 4],
        "opp_move_gids": [6],
        "my_bench": bench1,
        "opp_bench": opp_bench1,
        "legal_action_mask": mask1,
    }
    # 全部詰まった状態 (技フル、控えフル、全合法)。相手控えに瀕死 (hp 0) を含む。
    bench2 = [
        {"species_gid": 2 + i, "hp_frac": 1.0 - 0.125 * i, "move_gids": [1, 2]}
        for i in range(NUM_BENCH)
    ]
    opp_bench2 = [
        {"species_gid": 5 + i, "hp_frac": 0.0 if i == 0 else 0.99, "move_gids": [4]}
        for i in range(NUM_BENCH)
    ]
    state2 = {
        "my_species_gid": 1,
        "opp_species_gid": 2,
        "my_exact_hp_frac": 1.0,
        "opp_quantized_hp_frac": 1.0,
        "my_move_gids": list(range(1, NUM_MOVES + 1)),
        "opp_move_gids": list(range(5, 5 + NUM_MOVES)),
        "my_bench": bench2,
        "opp_bench": opp_bench2,
        "legal_action_mask": [True] * ACTION_DIM,
    }
    return [state1, state2]


def main() -> None:
    states = sample_states()
    encoded = encode_observations([{"state": s} for s in states], torch.device("cpu"))
    golden = {
        "states": states,
        "tensors": {
            f.name: getattr(encoded, f.name).to(torch.float64).tolist()
            for f in fields(encoded)
        },
    }
    out = Path(__file__).parent / "encoding_golden.json"
    out.write_text(json.dumps(golden, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
