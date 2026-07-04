from __future__ import annotations

import torch
from torch import nn

from poke_ai3 import (
    ACTION_DIM,
    MAX_MOVE_SLOTS,
    MOVE_VOCAB_CAP,
    NUM_BENCH,
    SPECIES_VOCAB_CAP,
    TYPE_VOCAB,
)

from .models.modernbert_abs import ModernBertAbsConfig, ModernBertAbsEncoder
from .encoding import EncodedObservations
from .tables import global_tables

MASKED_LOGIT = -1.0e9

# トークン列: CLS, my_active, opp_active, 技スロット×MAX_MOVE_SLOTS, 自控え×NUM_BENCH,
# 相手控え×NUM_BENCH (末尾追加: 既存スライスをずらさない)。
NUM_TOKENS = 3 + MAX_MOVE_SLOTS + 2 * NUM_BENCH

# position 埋め込みの固定容量。実トークン数 (NUM_TOKENS) より大きく取り、将来の新トークン
# (天候・フィールド等のグローバル機能) を**末尾追加**しても既存 index がずれず checkpoint が
# 生きるようにする。新トークンは常に末尾へ足し、この容量を超えない限り形状不変。
POSITION_EMBEDDING_CAP = 32
assert POSITION_EMBEDDING_CAP >= NUM_TOKENS


def default_model_config() -> ModernBertAbsConfig:
    """既定モデル設定 (隠れ 128, 2 層)。位置埋め込みは余裕を持たせた固定容量。"""
    return ModernBertAbsConfig(max_position_embeddings=POSITION_EMBEDDING_CAP)


class PolicyValueModel(nn.Module):
    """Embedding + pointer 方式の policy/value ネット。

    - 種族・技・タイプはグローバル ID の Embedding (語彙は全 Showdown dex)。
      タイプ Embedding は種族のタイプと技のタイプで共有する (相性知識の汎化)。
    - 技・控えの静的メタデータ (タイプ・威力・カテゴリ) は Rust の ID 表由来の
      バッファから gid で引く。
    - policy は各行動トークン (技スロット / 控え) の hidden state から共有ヘッドで
      ロジットを読む pointer 方式。行動枠が増えてもヘッドは不変。value は CLS。
    """

    def __init__(self, config: ModernBertAbsConfig | None = None) -> None:
        super().__init__()
        self.config = config or default_model_config()
        if self.config.max_position_embeddings < NUM_TOKENS:
            raise ValueError(
                f"max_position_embeddings must be >= {NUM_TOKENS} for the token layout"
            )
        hidden = self.config.hidden_size
        self.cls_embedding = nn.Embedding(1, hidden)
        self.species_embedding = nn.Embedding(SPECIES_VOCAB_CAP, hidden)
        self.move_embedding = nn.Embedding(MOVE_VOCAB_CAP, hidden)
        # +1 行は「タイプ無し」(単タイプ種族の type2 / 空きスロット)。
        self.type_embedding = nn.Embedding(TYPE_VOCAB + 1, hidden)
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, hidden
        )
        # スカラ特徴の射影: ポケモントークン [hp] (全ポケモントークン共通で 1 回だけ)、
        # 技トークン [power, cat3, legal]、自控えトークン [switch_legal]
        # (相手控えは legal が無いため追加スカラ射影なし)。
        self.mon_feat = nn.Linear(1, hidden)
        self.move_feat = nn.Linear(5, hidden)
        self.bench_feat = nn.Linear(1, hidden)
        self.encoder = ModernBertAbsEncoder(self.config)
        # pointer 方式の policy ヘッド (技用と交代用は別、各トークンに共有適用)。
        self.move_head = nn.Linear(hidden, 1)
        self.switch_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

        tables = global_tables()
        # 静的メタデータ (学習対象でないため checkpoint には含めない)。
        self.register_buffer("species_type1", tables.species_type1, persistent=False)
        self.register_buffer("species_type2", tables.species_type2, persistent=False)
        self.register_buffer("move_type", tables.move_type, persistent=False)
        self.register_buffer("move_power", tables.move_power, persistent=False)
        self.register_buffer("move_category", tables.move_category, persistent=False)
        self.apply(self._init_local_weights)

    def forward(self, enc: EncodedObservations) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(self.embed_tokens(enc))
        cls_state = hidden_states[:, 0, :]
        move_states = hidden_states[:, 3 : 3 + MAX_MOVE_SLOTS, :]
        # 相手控えトークンは末尾にあるため、自控えだけを切り出す。
        bench_states = hidden_states[:, 3 + MAX_MOVE_SLOTS : 3 + MAX_MOVE_SLOTS + NUM_BENCH, :]
        move_logits = self.move_head(move_states).squeeze(-1)
        switch_logits = self.switch_head(bench_states).squeeze(-1)
        raw_logits = torch.cat([move_logits, switch_logits], dim=-1)
        logits = raw_logits.masked_fill(~enc.legal_action_mask, MASKED_LOGIT)
        values = self.value_head(cls_state).squeeze(-1)
        return logits, values

    def _mon_token(self, species: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
        """ポケモン 1 体のトークン: 種族 + タイプ 2 枠 + HP。species: (...,), hp: (...,)。"""
        return (
            self.species_embedding(species)
            + self.type_embedding(self.species_type1[species])
            + self.type_embedding(self.species_type2[species])
            + self.mon_feat(hp.unsqueeze(-1))
        )

    def _move_mix(self, gids: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """習得技 Embedding の present 加重平均。gids/present: (..., MAX_MOVE_SLOTS)。"""
        emb = self.move_embedding(gids)
        n_moves = present.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (emb * present.unsqueeze(-1)).sum(dim=-2) / n_moves

    def embed_tokens(self, enc: EncodedObservations) -> torch.Tensor:
        batch_size = enc.my_species.shape[0]
        device = enc.my_species.device
        cls_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_tok = self.cls_embedding(cls_ids)
        my_tok = self._mon_token(enc.my_species, enc.my_hp).unsqueeze(1)
        # 相手 active: 技は行動対象ではないので独立トークンにせず mix で混ぜ込む。
        opp_tok = (
            self._mon_token(enc.opp_species, enc.opp_hp)
            + self._move_mix(enc.opp_move_gids, enc.opp_move_present)
        ).unsqueeze(1)

        # 技スロットトークン: 技 Embedding + タイプ + [威力, カテゴリ3, legal]。
        # 空きスロットは present=0 で丸ごと無効化する。
        move_scalar = torch.cat(
            [
                self.move_power[enc.move_gids].unsqueeze(-1),
                self.move_category[enc.move_gids],
                enc.move_legal.unsqueeze(-1),
            ],
            dim=-1,
        )
        move_tok = (
            self.move_embedding(enc.move_gids)
            + self.type_embedding(self.move_type[enc.move_gids])
            + self.move_feat(move_scalar)
        ) * enc.move_present.unsqueeze(-1)

        # 自控えトークン: ポケモントークン (HP は _mon_token の 1 回だけ) +
        # 習得技 mix + [switch_legal]。
        bench_tok = (
            self._mon_token(enc.bench_species, enc.bench_hp)
            + self._move_mix(enc.bench_move_gids, enc.bench_move_present)
            + self.bench_feat(enc.switch_legal.unsqueeze(-1))
        ) * enc.bench_present.unsqueeze(-1)

        # 相手控えトークン: 自控えと同構成だが legal 系スカラは無い。
        opp_bench_tok = (
            self._mon_token(enc.opp_bench_species, enc.opp_bench_hp)
            + self._move_mix(enc.opp_bench_move_gids, enc.opp_bench_move_present)
        ) * enc.opp_bench_present.unsqueeze(-1)

        hidden_states = torch.cat(
            [cls_tok, my_tok, opp_tok, move_tok, bench_tok, opp_bench_tok], dim=1
        )
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long, device=device)
        return hidden_states + self.position_embeddings(position_ids).unsqueeze(0)

    def _init_local_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:  # type: ignore
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


def action_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=logits)


assert ACTION_DIM == MAX_MOVE_SLOTS + NUM_BENCH
