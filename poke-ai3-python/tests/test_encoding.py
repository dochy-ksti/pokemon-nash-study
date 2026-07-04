"""観測エンコードの回帰テスト。

ゴールデン (tests/data/encoding_golden.json) は移行前の純 Python 実装
(encoding.py の旧 encode_observations) の出力を tests/data/make_golden.py で
固定したもの。現在の Rust 一本化経路 (encode_observation_states) が
同一テンソルを返すことを保証する。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch

from poke_ai3_train.encoding import EncodedObservations, encode_observations

GOLDEN_PATH = Path(__file__).parent / "data" / "encoding_golden.json"

EXPECTED_DTYPES = {
    "my_species": torch.long,
    "opp_species": torch.long,
    "move_gids": torch.long,
    "opp_move_gids": torch.long,
    "bench_species": torch.long,
    "bench_move_gids": torch.long,
    "opp_bench_species": torch.long,
    "opp_bench_move_gids": torch.long,
    "my_hp": torch.float32,
    "opp_hp": torch.float32,
    "move_present": torch.float32,
    "move_legal": torch.float32,
    "opp_move_present": torch.float32,
    "bench_hp": torch.float32,
    "bench_present": torch.float32,
    "bench_move_present": torch.float32,
    "opp_bench_hp": torch.float32,
    "opp_bench_present": torch.float32,
    "opp_bench_move_present": torch.float32,
    "switch_legal": torch.float32,
    "legal_action_mask": torch.bool,
}


def test_encode_observations_matches_golden() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    items = [{"state": s} for s in golden["states"]]
    encoded = encode_observations(items, torch.device("cpu"))
    for f in fields(EncodedObservations):
        actual = getattr(encoded, f.name)
        assert actual.dtype == EXPECTED_DTYPES[f.name], f.name
        expected = torch.tensor(golden["tensors"][f.name], dtype=torch.float64)
        assert actual.shape == expected.shape, f.name
        assert torch.equal(actual.to(torch.float64), expected), f.name


def test_mask_opp_obs_zeroes_opponent_extension() -> None:
    from poke_ai3_train.encoding import set_mask_opp_obs

    golden = json.loads(GOLDEN_PATH.read_text())
    items = [{"state": s} for s in golden["states"]]
    set_mask_opp_obs(True)
    try:
        masked = encode_observations(items, torch.device("cpu"))
    finally:
        set_mask_opp_obs(False)
    for name in (
        "opp_move_gids",
        "opp_move_present",
        "opp_bench_species",
        "opp_bench_hp",
        "opp_bench_present",
        "opp_bench_move_gids",
        "opp_bench_move_present",
    ):
        assert torch.count_nonzero(getattr(masked, name)) == 0, name
    # 既存観測 (相手 active 種族/HP・自軍側) はマスクの影響を受けない。
    plain = encode_observations(items, torch.device("cpu"))
    assert torch.equal(masked.opp_species, plain.opp_species)
    assert torch.equal(masked.opp_hp, plain.opp_hp)
    assert torch.equal(masked.bench_species, plain.bench_species)


def test_encode_observations_empty_batch() -> None:
    encoded = encode_observations([], torch.device("cpu"))
    assert encoded.my_species.shape == (0,)
    assert encoded.legal_action_mask.shape[0] == 0
