import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class PerceiverResampler(nnx.Module):
    """Compress S tokens → N tokens via learnable cross-attention queries.

    N learnable queries attend to the full input sequence, producing a fixed-size
    compressed representation. Like Flamingo's perceiver resampler.
    """

    def __init__(self, n_queries: int, dim: int, n_heads: int, rngs: nnx.Rngs):
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.queries = nnx.Param(jax.random.normal(rngs.params(), (1, n_queries, dim)) * 0.02)
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        tokens: at.Float[at.Array, "b s d"],
        mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b n d"]:
        b, _, d = tokens.shape
        n = self.n_queries

        queries = jnp.broadcast_to(self.queries.value, (b, n, d))
        q = self.q_proj(self.norm_q(queries)).reshape(b, n, self.n_heads, self.head_dim)
        kv_in = self.norm_kv(tokens)
        k = self.k_proj(kv_in).reshape(b, -1, self.n_heads, self.head_dim)
        v = self.v_proj(kv_in).reshape(b, -1, self.n_heads, self.head_dim)

        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.scale
        attn_mask = mask[:, None, None, :]  # [B, 1, 1, S]
        attn = jnp.where(attn_mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, n, d)
        return self.out_proj(out)  # [B, N, D]
