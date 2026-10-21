"""Action patchification strategies for temporal compression of action sequences.

Patchifiers transform action sequences (B, H, D) ↔ (B, N, patch_dim) where N = H / patch_size.
Each method offers a different inductive bias for how consecutive action steps are grouped into tokens.
"""

from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


class ConcatPatchifier(nnx.Module):
    """Concatenate P consecutive steps into one token (default/existing behavior)."""

    def __init__(self, action_dim: int, action_horizon: int, patch_size: int):
        self.patch_size = patch_size
        self._action_dim = action_dim
        self.patch_dim = action_dim * patch_size
        self.num_tokens = action_horizon // patch_size

    def patchify(self, actions):
        """(B, H, D) → (B, N, P*D)"""
        if self.patch_size == 1:
            return actions
        return einops.rearrange(actions, "b (n p) d -> b n (p d)", p=self.patch_size)

    def unpatchify(self, tokens):
        """(B, N, P*D) → (B, H, D)"""
        if self.patch_size == 1:
            return tokens
        return einops.rearrange(tokens, "b n (p d) -> b (n p) d", p=self.patch_size)


class DeltaPatchifier(nnx.Module):
    """First step absolute + deltas within each patch, then concatenate.

    Exploits smoothness of action sequences — deltas between consecutive steps
    are typically small, making the representation easier to learn.
    """

    def __init__(self, action_dim: int, action_horizon: int, patch_size: int):
        self.patch_size = patch_size
        self._action_dim = action_dim
        self.patch_dim = action_dim * patch_size
        self.num_tokens = action_horizon // patch_size

    def patchify(self, actions):
        """(B, H, D) → (B, N, P*D) with [a₀, a₁-a₀, a₂-a₁, ...]"""
        if self.patch_size == 1:
            return actions
        x = einops.rearrange(actions, "b (n p) d -> b n p d", p=self.patch_size)
        # First sub-step stays absolute, rest become deltas
        deltas = jnp.diff(x, axis=2)  # (B, N, P-1, D)
        x = jnp.concatenate([x[:, :, :1, :], deltas], axis=2)  # (B, N, P, D)
        return einops.rearrange(x, "b n p d -> b n (p d)")

    def unpatchify(self, tokens):
        """(B, N, P*D) → (B, H, D) via cumsum to undo deltas."""
        if self.patch_size == 1:
            return tokens
        x = einops.rearrange(tokens, "b n (p d) -> b n p d", p=self.patch_size)
        x = jnp.cumsum(x, axis=2)
        return einops.rearrange(x, "b n p d -> b (n p) d")


class TemporalPosPatchifier(nnx.Module):
    """Add learned sub-step positional embeddings before concatenating.

    Helps the model distinguish which sub-step within a patch each D-dim
    slice corresponds to, without relying on implicit position in the concat.
    Pos embedding is input-only — unpatchify is the same as concat.
    """

    def __init__(self, action_dim: int, action_horizon: int, patch_size: int, *, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self._action_dim = action_dim
        self.patch_dim = action_dim * patch_size
        self.num_tokens = action_horizon // patch_size
        self.sub_pos_emb = nnx.Param(jax.random.normal(rngs.params(), (patch_size, action_dim)) * 0.02)

    def patchify(self, actions):
        """(B, H, D) → (B, N, P*D) with sub-step positional embeddings added."""
        if self.patch_size == 1:
            return actions
        x = einops.rearrange(actions, "b (n p) d -> b n p d", p=self.patch_size)
        x = x + self.sub_pos_emb.value[None, None, :, :]
        return einops.rearrange(x, "b n p d -> b n (p d)")

    def unpatchify(self, tokens):
        """(B, N, P*D) → (B, H, D). Pos embedding is input-only."""
        if self.patch_size == 1:
            return tokens
        return einops.rearrange(tokens, "b n (p d) -> b (n p) d", p=self.patch_size)


class Conv1dPatchifier(nnx.Module):
    """Learned Conv1d for patchification, ConvTranspose for de-patchification.

    Unlike concatenation, the convolution learns local temporal features
    with overlapping receptive fields within each patch window.
    """

    def __init__(self, action_dim: int, action_horizon: int, patch_size: int, *, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self._action_dim = action_dim
        self.patch_dim = action_dim * patch_size
        self.num_tokens = action_horizon // patch_size
        self.encode_conv = nnx.Conv(
            in_features=action_dim,
            out_features=self.patch_dim,
            kernel_size=(patch_size,),
            strides=(patch_size,),
            padding="VALID",
            rngs=rngs,
        )
        self.decode_conv = nnx.ConvTranspose(
            in_features=self.patch_dim,
            out_features=action_dim,
            kernel_size=(patch_size,),
            strides=(patch_size,),
            padding="VALID",
            rngs=rngs,
        )

    def patchify(self, actions):
        """(B, H, D) → (B, N, P*D) via Conv1d."""
        if self.patch_size == 1:
            return actions
        return self.encode_conv(actions)

    def unpatchify(self, tokens):
        """(B, N, P*D) → (B, H, D) via ConvTranspose."""
        if self.patch_size == 1:
            return tokens
        return self.decode_conv(tokens)


class DCTPatchifier(nnx.Module):
    """Frequency-domain patchification via DCT truncation.

    Keeps only the first N = H/P DCT coefficients (low-frequency components),
    discarding high-frequency detail. Natural compression for smooth action trajectories.
    Note: patch_dim = D (not P*D) since each DCT coefficient is D-dimensional.
    """

    def __init__(self, action_dim: int, action_horizon: int, patch_size: int):
        self.patch_size = patch_size
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self.patch_dim = action_dim  # Each DCT coeff is D-dimensional
        self.num_tokens = action_horizon // patch_size

    def patchify(self, actions):
        """(B, H, D) → (B, N, D) via DCT + truncation."""
        if self.patch_size == 1:
            return actions
        freq = jax.scipy.fft.dct(actions, type=2, axis=-2, norm="ortho")
        return freq[:, : self.num_tokens, :]

    def unpatchify(self, tokens):
        """(B, N, D) → (B, H, D) via zero-pad + IDCT."""
        if self.patch_size == 1:
            return tokens
        pad_width = self._action_horizon - self.num_tokens
        padded = jnp.pad(tokens, ((0, 0), (0, pad_width), (0, 0)))
        return jax.scipy.fft.idct(padded, type=2, axis=-2, norm="ortho")


PatchMethod = Literal["concat", "delta", "temporal_pos", "conv1d", "dct"]


def create_patchifier(
    action_dim: int,
    action_horizon: int,
    patch_size: int,
    method: PatchMethod = "concat",
    *,
    rngs: nnx.Rngs | None = None,
) -> ConcatPatchifier | DeltaPatchifier | TemporalPosPatchifier | Conv1dPatchifier | DCTPatchifier:
    """Factory to create the appropriate patchifier from config values."""
    if patch_size <= 1:
        return ConcatPatchifier(action_dim, action_horizon, 1)

    if method == "concat":
        return ConcatPatchifier(action_dim, action_horizon, patch_size)
    if method == "delta":
        return DeltaPatchifier(action_dim, action_horizon, patch_size)
    if method == "temporal_pos":
        assert rngs is not None, "temporal_pos patchifier requires rngs"
        return TemporalPosPatchifier(action_dim, action_horizon, patch_size, rngs=rngs)
    if method == "conv1d":
        assert rngs is not None, "conv1d patchifier requires rngs"
        return Conv1dPatchifier(action_dim, action_horizon, patch_size, rngs=rngs)
    if method == "dct":
        return DCTPatchifier(action_dim, action_horizon, patch_size)
    raise ValueError(f"Unknown action patch method: {method}")
