# ModernBERT Absolute-Position Adapter

This module is a small Phase 1 encoder adapted from the Hugging Face
Transformers ModernBERT implementation shape and naming, but narrowed to the
Pokemon Phase 1 PPO use case.

The upstream Transformers project is licensed under Apache-2.0. Keep that
license context in mind if this local adapter grows to copy larger upstream
sections verbatim.

Source reference:

- Transformers ModernBERT docs: https://huggingface.co/docs/transformers/main/en/model_doc/modernbert
- Transformers ModernBERT source, pinned by this project to `transformers==4.57.1`:
  https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/modernbert

The important local changes are:

- RoPE is not used.
- Absolute position embeddings are supplied by the Phase 1 model wrapper.
- The encoder consumes already-built hidden states instead of tokenizing every
  game-state slot.
- FlashAttention is required. No eager/CPU attention fallback is provided.

Keep the dependency version and this source reference in sync when updating the
vendored implementation.
