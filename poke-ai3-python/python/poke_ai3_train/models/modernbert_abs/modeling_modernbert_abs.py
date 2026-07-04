from __future__ import annotations

# Adapted from the Hugging Face Transformers ModernBERT encoder structure
# (Apache-2.0) for this repository's fixed-slot Phase 1 PPO model.

from collections.abc import Callable

import torch
from torch import nn

from .configuration_modernbert_abs import ModernBertAbsConfig


def require_flash_attention() -> Callable[..., torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention is required, but CUDA is not available")
    try:
        from flash_attn import flash_attn_func
    except ImportError as error:
        raise RuntimeError("FlashAttention is required; install flash-attn") from error
    return flash_attn_func


class ModernBertAbsAttention(nn.Module):
    def __init__(self, config: ModernBertAbsConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout_prob
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        flash_attn_func = require_flash_attention()
        attn = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )
        return self.out_dropout(self.out(attn.reshape(batch_size, seq_len, hidden_size)))


class ModernBertAbsMLP(nn.Module):
    def __init__(self, config: ModernBertAbsConfig) -> None:
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(hidden_states) * gate))


class ModernBertAbsLayer(nn.Module):
    def __init__(self, config: ModernBertAbsConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = ModernBertAbsAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ModernBertAbsMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertAbsEncoder(nn.Module):
    """ModernBERT-style encoder that expects absolute-positioned embeddings."""

    def __init__(self, config: ModernBertAbsConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            ModernBertAbsLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.apply(self._init_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_norm(hidden_states)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
