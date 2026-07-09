"""観測 (Rust `StateForPlayer`) → モデル入力テンソル群への窓口。

エンコード (テンソル列の構築) は Rust `obs_encode::encode_batch` が唯一の実装。
このモジュールは Rust が返す numpy 配列 dict を torch テンソルの
`EncodedObservations` に変換するだけで、レイアウトの知識は持たない。

経路は 2 つ:
- 学習ホットパス: `RustAsyncExecutor.recv_observations` が返す dict を
  `encoded_from_arrays` で包む (ゼロコピー numpy → torch)。
- それ以外 (play_gui / 学習バッチ構築 / 診断ツール): `StateForPlayer` の dict 列を
  `encode_observations` に渡すと、JSON 経由で Rust `encode_observation_states` を
  呼んでエンコードする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from poke_ai3 import ENCODED_PACKED_LAYOUT, encode_observation_states


@dataclass(frozen=True)
class EncodedObservations:
    """バッチ化した観測。gid 系は long、特徴系は float32。

    - my/opp_species: (B,)、my/opp_hp: (B,)
    - move_gids/move_present/move_legal: (B, MAX_MOVE_SLOTS)。空きスロットは
      gid=0/present=0 (model 側で present を掛けて無効化する)。
    - opp_move_*: (B, MAX_MOVE_SLOTS)。相手 active の技 (神視点)。
    - bench_*: (B, NUM_BENCH)、bench_move_*: (B, NUM_BENCH, MAX_MOVE_SLOTS)。
      空き枠は present=0。switch_legal は交代枠の legal bit。
    - opp_bench_*: (B, NUM_BENCH) / (B, NUM_BENCH, MAX_MOVE_SLOTS)。相手控え
      (HP は量子化済み)。legal 系スカラは持たない。
    - legal_action_mask: (B, ACTION_DIM) bool (出力ロジットのマスク用)。
    """

    my_species: torch.Tensor
    opp_species: torch.Tensor
    my_hp: torch.Tensor
    opp_hp: torch.Tensor
    move_gids: torch.Tensor
    move_present: torch.Tensor
    move_legal: torch.Tensor
    opp_move_gids: torch.Tensor
    opp_move_present: torch.Tensor
    bench_species: torch.Tensor
    bench_hp: torch.Tensor
    bench_present: torch.Tensor
    bench_move_gids: torch.Tensor
    bench_move_present: torch.Tensor
    opp_bench_species: torch.Tensor
    opp_bench_hp: torch.Tensor
    opp_bench_present: torch.Tensor
    opp_bench_move_gids: torch.Tensor
    opp_bench_move_present: torch.Tensor
    switch_legal: torch.Tensor
    legal_action_mask: torch.Tensor

    def __getitem__(self, indices: torch.Tensor) -> EncodedObservations:
        """minibatch 用の行スライス。"""
        return EncodedObservations(
            **{f.name: getattr(self, f.name)[indices] for f in fields(self)}
        )


# Rust が返す dict のうちテンソルでないルーティング系キー (encoded.rs と一致させる)。
_ROUTING_KEYS = frozenset(
    {"game_id", "game_index", "player", "request_id",
     "empty_game_id", "empty_player", "empty_request_id"}
)

# 相手側拡張観測 (神視点) のマスク対象。A/B ベースライン (旧観測相当) 用。
_OPP_OBS_KEYS = (
    "opp_move_gids",
    "opp_move_present",
    "opp_bench_species",
    "opp_bench_hp",
    "opp_bench_present",
    "opp_bench_move_gids",
    "opp_bench_move_present",
)

_mask_opp_obs = False


def set_mask_opp_obs(enabled: bool) -> None:
    """相手側拡張観測 (相手 active 技・相手控え) をゼロ化するグローバルフラグ。

    A/B 比較のベースライン (旧観測相当・アーキテクチャ同一) 用。present が 0 になる
    ためモデル側でトークン内容が消え、位置埋め込みのみの定数トークンが残る。"""
    global _mask_opp_obs
    _mask_opp_obs = enabled


def _apply_opp_mask(enc: EncodedObservations) -> EncodedObservations:
    if not _mask_opp_obs:
        return enc
    return replace(enc, **{k: torch.zeros_like(getattr(enc, k)) for k in _OPP_OBS_KEYS})


def _column_layout(entries: list[list[Any]]) -> tuple[dict[str, tuple[int, int, tuple[int, ...]]], int]:
    """[(name, per-row shape)] → {name: (列オフセット, 列数, shape)} と全列数。"""
    columns: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    offset = 0
    for name, shape in entries:
        n = int(np.prod(shape)) if shape else 1
        columns[name] = (offset, n, tuple(shape))
        offset += n
    return columns, offset


_PACKED_LAYOUT = json.loads(ENCODED_PACKED_LAYOUT)
_I64_COLUMNS, PACKED_I64_WIDTH = _column_layout(_PACKED_LAYOUT["i64"])
_F32_COLUMNS, PACKED_F32_WIDTH = _column_layout(_PACKED_LAYOUT["f32"])


def packed_views(
    packed_i64: torch.Tensor, packed_f32: torch.Tensor, mask: torch.Tensor
) -> EncodedObservations:
    """パック行列 (B, K) の列スライスから `EncodedObservations` を組み立てる。

    スライス+reshape はビュー操作中心の軽量カーネルで、CUDA Graph キャプチャ内で
    呼べば replay 時の発行コストはゼロになる。"""
    batch = mask.shape[0]
    fields_: dict[str, torch.Tensor] = {"legal_action_mask": mask}
    for packed, columns in ((packed_i64, _I64_COLUMNS), (packed_f32, _F32_COLUMNS)):
        for name, (offset, n, shape) in columns.items():
            view = packed[:, offset : offset + n]
            fields_[name] = view.reshape(batch, *shape) if shape else view.reshape(batch)
    expected = {f.name for f in fields(EncodedObservations)}
    if set(fields_) != expected:
        raise KeyError(
            f"packed layout mismatch: missing={sorted(expected - set(fields_))} "
            f"extra={sorted(set(fields_) - expected)}"
        )
    return _apply_opp_mask(EncodedObservations(**fields_))


def encoded_from_arrays(
    arrays: dict[str, np.ndarray], device: torch.device
) -> EncodedObservations:
    """Rust エンコード済み numpy 配列 dict → torch テンソルの `EncodedObservations`。

    dict のキーは Rust 側 (`encoded.rs`) がこのクラスのフィールド名と一致させている。
    フィールド名・ルーティング系以外の未知キーは「Rust がフィールド追加したのに
    こちらが追従していない」事故 (サイレント欠落) なので assert で落とす。"""
    field_names = {f.name for f in fields(EncodedObservations)}
    unknown = set(arrays) - field_names - _ROUTING_KEYS
    if unknown:
        raise KeyError(
            f"Rust encoder returned unknown keys {sorted(unknown)}; "
            "add matching fields to EncodedObservations"
        )
    return _apply_opp_mask(
        EncodedObservations(
            **{
                name: torch.from_numpy(arrays[name]).to(device)
                for name in field_names
            }
        )
    )


def encode_observations(items: list[dict[str, Any]], device: torch.device) -> EncodedObservations:
    """`{"state": StateForPlayer dict}` の列を Rust でエンコードする。

    trajectory item (state 以外のキーを含む) をそのまま渡せる。"""
    states_json = json.dumps([item["state"] for item in items])
    return encoded_from_arrays(encode_observation_states(states_json), device)
