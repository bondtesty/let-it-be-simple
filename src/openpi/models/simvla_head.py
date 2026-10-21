"""SimVLA-style vanilla Transformer encoder action head.

A lightweight bidirectional Transformer that replaces the Gemma action expert
for flow-matching action generation. Follows the SimVLA paper (arXiv 2602.18224):
concatenate [cond_tokens | state | actions+time] → self-attention → action prediction.
"""

import dataclasses
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class SimVLAHeadConfig:
    depth: int = 4
    width: int = 768
    n_heads: int = 12
    mlp_dim: int = 3072
    pos_injection: Literal["input", "per_layer", "rope"] = "input"
    pos_init_std: float = 0.0


def _apply_rope(x: at.Float[at.Array, "b s h d"], positions: at.Int[at.Array, "b s"], max_wavelength: int = 10_000):
    """Applies rotary position embeddings to q/k tensors shaped [B, S, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)


class RotarySelfAttention(nnx.Module):
    """Self-attention with RoPE applied to query/key projections."""

    def __init__(self, width: int, n_heads: int, rngs: nnx.Rngs):
        if width % n_heads != 0:
            raise ValueError(f"width must be divisible by n_heads, got width={width}, n_heads={n_heads}")
        if (width // n_heads) % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got head_dim={width // n_heads}")
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.q_proj = nnx.Linear(width, width, rngs=rngs)
        self.k_proj = nnx.Linear(width, width, rngs=rngs)
        self.v_proj = nnx.Linear(width, width, rngs=rngs)
        self.out_proj = nnx.Linear(width, width, rngs=rngs)

    def __call__(
        self,
        x: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b s d"]:
        batch, seq_len, _ = x.shape

        def split_heads(y):
            return y.reshape(batch, seq_len, self.n_heads, self.head_dim)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch, seq_len))
        q = _apply_rope(q, positions)
        k = _apply_rope(k, positions)

        scale = self.head_dim**-0.5
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if mask is not None:
            logits = jnp.where(mask[:, None, None, :], logits, jnp.array(-1e30, dtype=logits.dtype))
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        out = out.reshape(batch, seq_len, self.width)
        return self.out_proj(out)


class TransformerBlock(nnx.Module):
    """Pre-LayerNorm Transformer encoder block with bidirectional self-attention."""

    def __init__(self, width: int, n_heads: int, mlp_dim: int, pos_injection: str, rngs: nnx.Rngs):
        self.attn_norm = nnx.LayerNorm(width, rngs=rngs)
        if pos_injection == "rope":
            self.attn = RotarySelfAttention(width, n_heads, rngs=rngs)
            self._use_rope_attention = True
        else:
            self.attn = nnx.MultiHeadAttention(
                num_heads=n_heads,
                in_features=width,
                decode=False,
                rngs=rngs,
            )
            self._use_rope_attention = False
        self.mlp_norm = nnx.LayerNorm(width, rngs=rngs)
        self.mlp_up = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.mlp_down = nnx.Linear(mlp_dim, width, rngs=rngs)

    def __call__(
        self, x: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"] | None = None
    ) -> at.Float[at.Array, "b s d"]:
        # Self-attention with pre-norm
        h = self.attn_norm(x)
        if self._use_rope_attention:
            h = self.attn(h, mask=mask)
        else:
            attn_mask = mask[:, None, None, :] if mask is not None else None
            h = self.attn(h, mask=attn_mask)
        x = x + h

        # MLP with pre-norm and GELU
        h = self.mlp_norm(x)
        h = self.mlp_up(h)
        h = nnx.gelu(h)
        h = self.mlp_down(h)

        return x + h


class SimVLAHead(nnx.Module):
    """SimVLA-style vanilla Transformer encoder for action generation.

    Takes concatenated [cond | state | action+time] tokens and processes them
    with bidirectional self-attention. No causal masking, no RoPE, no GQA.
    """

    def __init__(self, config: SimVLAHeadConfig, max_seq_len: int, rngs: nnx.Rngs):
        self.config = config
        if config.pos_init_std > 0:
            pos_init = jax.random.normal(rngs.params(), (1, max_seq_len, config.width)) * config.pos_init_std
        else:
            pos_init = jnp.zeros((1, max_seq_len, config.width))
        self.pos_emb = nnx.Param(pos_init)
        for i in range(config.depth):
            if config.pos_injection == "per_layer":
                if config.pos_init_std > 0:
                    layer_pos = jax.random.normal(rngs.params(), (1, max_seq_len, config.width)) * config.pos_init_std
                else:
                    layer_pos = jnp.zeros((1, max_seq_len, config.width))
                setattr(self, f"layer_pos_emb_{i}", nnx.Param(layer_pos))
            setattr(
                self,
                f"block_{i}",
                TransformerBlock(config.width, config.n_heads, config.mlp_dim, config.pos_injection, rngs=rngs),
            )
        self._depth = config.depth
        self.final_norm = nnx.LayerNorm(config.width, rngs=rngs)

    def __call__(
        self,
        tokens: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b s d"]:
        seq_len = tokens.shape[1]
        x = tokens + self.pos_emb.value[:, :seq_len, :] if self.config.pos_injection == "input" else tokens

        for i in range(self._depth):
            if self.config.pos_injection == "per_layer":
                x = x + getattr(self, f"layer_pos_emb_{i}").value[:, :seq_len, :]
            x = getattr(self, f"block_{i}")(x, mask=mask)

        return self.final_norm(x)
