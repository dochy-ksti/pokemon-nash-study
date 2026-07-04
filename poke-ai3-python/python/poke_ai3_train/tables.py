"""グローバル ID 表 (Rust `global_ids` が export する TSV) のパーサ。

種族・技・タイプの ID と静的メタデータ (タイプ・威力・カテゴリ) は Rust 側の
表が唯一の正で、ここでは torch テンソルへ整形するだけ。ID は Embedding の
行番号として checkpoint に焼き付くため永久に不変。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from poke_ai3 import (
    MOVE_TABLE_TSV,
    MOVE_VOCAB,
    SPECIES_TABLE_TSV,
    SPECIES_VOCAB,
    TYPE_TABLE_TSV,
    TYPE_VOCAB,
)

# 単タイプ種族の type2 に使う「タイプ無し」 (type Embedding の追加行)。
TYPE_NONE = TYPE_VOCAB

# 技カテゴリ index (Rust MoveCategory と一致)。
CATEGORY_INDEX = {"Physical": 0, "Special": 1, "Status": 2}

# 威力の正規化分母 (最大級の技 ~250 を 1.0 近傍へ)。
_POWER_SCALE = 250.0


@dataclass(frozen=True)
class GlobalTables:
    """gid で引く静的メタデータテーブル (CPU テンソル)。"""

    species_gid: dict[str, int]
    move_gid: dict[str, int]
    type_gid: dict[str, int]
    # (SPECIES_VOCAB,) タイプ gid。単タイプの type2 は TYPE_NONE。
    species_type1: torch.Tensor
    species_type2: torch.Tensor
    # (MOVE_VOCAB,) タイプ gid / 正規化威力 / カテゴリ one-hot (MOVE_VOCAB, 3)。
    move_type: torch.Tensor
    move_power: torch.Tensor
    move_category: torch.Tensor


@lru_cache(maxsize=1)
def global_tables() -> GlobalTables:
    type_gid: dict[str, int] = {}
    for line in TYPE_TABLE_TSV.splitlines()[1:]:
        gid, name = line.split("\t")
        type_gid[name] = int(gid)
    assert len(type_gid) == TYPE_VOCAB

    species_gid: dict[str, int] = {}
    species_type1 = torch.full((SPECIES_VOCAB,), TYPE_NONE, dtype=torch.long)
    species_type2 = torch.full((SPECIES_VOCAB,), TYPE_NONE, dtype=torch.long)
    for line in SPECIES_TABLE_TSV.splitlines()[1:]:
        gid_s, name, _dex_num, types = line.split("\t")
        gid = int(gid_s)
        species_gid[name] = gid
        parts = types.split("/")
        species_type1[gid] = type_gid[parts[0]]
        if len(parts) > 1:
            species_type2[gid] = type_gid[parts[1]]
    assert len(species_gid) == SPECIES_VOCAB

    move_gid: dict[str, int] = {}
    move_type = torch.full((MOVE_VOCAB,), TYPE_NONE, dtype=torch.long)
    move_power = torch.zeros(MOVE_VOCAB, dtype=torch.float32)
    move_category = torch.zeros((MOVE_VOCAB, 3), dtype=torch.float32)
    for line in MOVE_TABLE_TSV.splitlines()[1:]:
        gid_s, name, mtype, category, base_power = line.split("\t")
        gid = int(gid_s)
        move_gid[name] = gid
        move_type[gid] = type_gid[mtype]
        move_power[gid] = float(base_power) / _POWER_SCALE
        move_category[gid, CATEGORY_INDEX[category]] = 1.0
    assert len(move_gid) == MOVE_VOCAB

    return GlobalTables(
        species_gid=species_gid,
        move_gid=move_gid,
        type_gid=type_gid,
        species_type1=species_type1,
        species_type2=species_type2,
        move_type=move_type,
        move_power=move_power,
        move_category=move_category,
    )


def species_gid(name: str) -> int:
    return global_tables().species_gid[name]


def move_gid(name: str) -> int:
    return global_tables().move_gid[name]
