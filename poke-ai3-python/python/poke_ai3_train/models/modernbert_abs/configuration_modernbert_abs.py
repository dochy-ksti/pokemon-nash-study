from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModernBertAbsConfig:
    """Small ModernBERT-style encoder config for fixed Phase 1 slots."""

    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 256
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 5
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModernBertAbsConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})
