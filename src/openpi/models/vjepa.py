"""V-JEPA 2.1 video encoder for PI0.

Adapted from https://github.com/facebookresearch/vjepa2
V-JEPA learns dense visual representations via self-supervised video prediction.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video input (16x16x2 patches)."""

    patch_size: tuple[int, int, int] = (2, 16, 16)  # T, H, W
    embed_dim: int = 1024
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, T, H, W, C) video tensor
        Returns:
            (B, N, embed_dim) patch tokens where N = (T/pt) * (H/ph) * (W/pw)
        """
        B, T, H, W, C = x.shape
        pt, ph, pw = self.patch_size

        # 3D convolution for video patches
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)

        # Reshape to (B, N, embed_dim)
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        return x


class PatchEmbed2D(nn.Module):
    """2D patch embedding for image input (16x16 patches)."""

    patch_size: tuple[int, int] = (16, 16)
    embed_dim: int = 1024
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) image tensor
        Returns:
            (B, N, embed_dim) patch tokens where N = (H/ph) * (W/pw)
        """
        B, H, W, C = x.shape
        ph, pw = self.patch_size

        # 2D convolution for image patches
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)

        # Reshape to (B, N, embed_dim)
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        return x


def posemb_sincos_3d(t, h, w, width, temperature=10000.0, dtype=jnp.float32):
    """3D sine-cosine positional embeddings for video patches.

    Args:
        t, h, w: temporal, height, width dimensions
        width: embedding dimension (will be rounded down to nearest multiple of 6)
    """
    # Round down to nearest multiple of 6 for positional encoding
    # Then pad to actual width if needed
    pe_width = (width // 6) * 6
    dim_per_axis = pe_width // 6

    # Temporal positions
    pos_t = jnp.arange(t, dtype=dtype)
    # Spatial positions
    pos_h = jnp.arange(h, dtype=dtype)
    pos_w = jnp.arange(w, dtype=dtype)

    # Create 3D grid
    grid_t, grid_h, grid_w = jnp.meshgrid(pos_t, pos_h, pos_w, indexing='ij')

    # Compute embeddings for each dimension
    omega = jnp.arange(dim_per_axis, dtype=dtype) / (dim_per_axis - 1)
    omega = 1.0 / (temperature ** omega)

    # Temporal embeddings
    emb_t = jnp.einsum("i,d->id", grid_t.flatten(), omega)
    # Height embeddings
    emb_h = jnp.einsum("i,d->id", grid_h.flatten(), omega)
    # Width embeddings
    emb_w = jnp.einsum("i,d->id", grid_w.flatten(), omega)

    pe = jnp.concatenate([
        jnp.sin(emb_t), jnp.cos(emb_t),
        jnp.sin(emb_h), jnp.cos(emb_h),
        jnp.sin(emb_w), jnp.cos(emb_w),
    ], axis=1)

    # Pad to actual width if needed (e.g., 1024 -> 1020 + 4 zeros)
    if pe_width < width:
        pad_width = width - pe_width
        pe = jnp.pad(pe, ((0, 0), (0, pad_width)), mode='constant')

    return pe[None, :, :]


def posemb_sincos_2d(h, w, width, temperature=10000.0, dtype=jnp.float32):
    """2D sine-cosine positional embeddings for image patches."""
    # Round down to nearest multiple of 4 for positional encoding
    pe_width = (width // 4) * 4
    dim_per_axis = pe_width // 4

    y, x = jnp.mgrid[:h, :w]
    omega = jnp.arange(dim_per_axis, dtype=dtype) / (dim_per_axis - 1)
    omega = 1.0 / (temperature ** omega)

    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)

    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)

    # Pad to actual width if needed
    if pe_width < width:
        pad_width = width - pe_width
        pe = jnp.pad(pe, ((0, 0), (0, pad_width)), mode='constant')

    return pe[None, :, :]


class MlpBlock(nn.Module):
    """Transformer MLP block."""

    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class EncoderBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    mlp_dim: int | None = None
    num_heads: int = 16
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Pre-norm attention
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = sharding.activation_sharding_constraint(y)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = x + y

        # Pre-norm MLP
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = sharding.activation_sharding_constraint(y)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = x + y

        return x


class Encoder(nn.Module):
    """V-JEPA encoder (ViT backbone)."""

    depth: int = 24
    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 16
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        """
        Args:
            x: (B, N, embed_dim) patch tokens with positional encoding
        Returns:
            (B, N, embed_dim) encoded tokens
        """
        # Simple loop over encoder blocks
        # TODO: Implement scan for memory efficiency on very deep networks
        for lyr in range(self.depth):
            x = EncoderBlock(
                name=f"encoderblock_{lyr}",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)

        return x


class Module(nn.Module):
    """V-JEPA 2.1 vision encoder module.

    Supports both image and video inputs with modality-specific tokenizers.
    Designed to be a drop-in replacement for SigLIP in PI0.
    """

    # Model architecture
    variant: str = "vit_l"  # "vit_b", "vit_l", "vit_g"
    num_classes: int = 768  # Output dimension (project to match Gemma width)
    pool_type: str = "none"  # "none" = return all tokens (like SigLIP)
    dtype_mm: str = "float32"
    scan: bool = True

    def setup(self):
        # V-JEPA 2.1 configurations
        self.configs = {
            "vit_b": {
                "depth": 12,
                "embed_dim": 768,
                "mlp_dim": 3072,
                "num_heads": 12,
            },
            "vit_l": {
                "depth": 24,
                "embed_dim": 1024,
                "mlp_dim": 4096,
                "num_heads": 16,
            },
            "vit_g": {
                "depth": 40,
                "embed_dim": 1536,
                "mlp_dim": 6144,
                "num_heads": 24,
            },
        }

        cfg = self.configs.get(self.variant, self.configs["vit_l"])
        self.depth = cfg["depth"]
        self.embed_dim = cfg["embed_dim"]
        self.mlp_dim = cfg["mlp_dim"]
        self.num_heads = cfg["num_heads"]

        # Modality-specific tokenizers
        self.video_tokenizer = PatchEmbed3D(
            patch_size=(2, 16, 16),
            embed_dim=self.embed_dim,
            dtype_mm=self.dtype_mm,
        )
        self.image_tokenizer = PatchEmbed2D(
            patch_size=(16, 16),
            embed_dim=self.embed_dim,
            dtype_mm=self.dtype_mm,
        )

        # Learnable modality tokens
        self.modality_token_video = self.param(
            "modality_token_video",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        self.modality_token_image = self.param(
            "modality_token_image",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )

        # Transformer encoder
        self.encoder = Encoder(
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=0.0,
            dtype_mm=self.dtype_mm,
        )

        # Output projection to match Gemma width
        self.output_proj = nn.Dense(
            self.num_classes,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )

    def _add_positional_encoding(self, x, is_video: bool, patch_shape: tuple):
        """Add sine-cosine positional encoding."""
        if is_video:
            t, h, w = patch_shape
            pos_emb = posemb_sincos_3d(t, h, w, self.embed_dim, dtype=x.dtype)
        else:
            h, w = patch_shape
            pos_emb = posemb_sincos_2d(h, w, self.embed_dim, dtype=x.dtype)

        return x + pos_emb

    def __call__(self, x, train: bool = False):
        """
        Args:
            x: Input tensor
               - Video: (B, T, H, W, C) or (B, T, C, H, W)
               - Image: (B, H, W, C) or (B, C, H, W)
            train: Whether in training mode

        Returns:
            tokens: (B, N, num_classes) visual tokens (aggregated across frames for video)
            unused: None (for API compatibility with SigLIP)
        """
        deterministic = not train

        # Detect input format and normalize
        if x.ndim == 5:
            # Video input (B, T, C, H, W) -> (B, T, H, W, C)
            if x.shape[2] in [1, 3]:  # Channel-first
                x = jnp.transpose(x, (0, 1, 3, 4, 2))
            is_video = True

            B, T, H, W, C = x.shape
            pt, ph, pw = 2, 16, 16
            patch_t, patch_h, patch_w = T // pt, H // ph, W // pw

            # Process video using 3D tokenizer (expects 5D input)
            x = self.video_tokenizer(x)

            # Add modality token
            modality_token = jnp.broadcast_to(
                self.modality_token_video, (B, 1, self.embed_dim)
            )
            x = jnp.concatenate([modality_token, x], axis=1)

            # Add positional encoding (skip modality token)
            x = x.at[:, 1:].set(
                self._add_positional_encoding(x[:, 1:], True, (patch_t, patch_h, patch_w))
            )

        elif x.ndim == 4:
            # Image input (B, C, H, W) -> (B, H, W, C)
            if x.shape[1] in [1, 3]:  # Channel-first
                x = jnp.transpose(x, (0, 2, 3, 1))
            is_video = False

            B, H, W, C = x.shape
            ph, pw = 16, 16
            patch_h, patch_w = H // ph, W // pw

            # Tokenize
            x = self.image_tokenizer(x)

            # Add modality token
            modality_token = jnp.broadcast_to(
                self.modality_token_image, (B, 1, self.embed_dim)
            )
            x = jnp.concatenate([modality_token, x], axis=1)

            # Add positional encoding (skip modality token)
            x = x.at[:, 1:].set(
                self._add_positional_encoding(x[:, 1:], False, (patch_h, patch_w))
            )

        else:
            raise ValueError(f"Expected 4D (image) or 5D (video) input, got {x.ndim}D")

        # Encode
        x = self.encoder(x, deterministic)

        # Remove modality token from output
        x = x[:, 1:]

        # Project to output dimension
        x = self.output_proj(x)

        return x, None


def get_config(variant: str = "vit_l") -> dict:
    """Get V-JEPA configuration."""
    configs = {
        "vit_b": {"depth": 12, "embed_dim": 768, "mlp_dim": 3072, "num_heads": 12},
        "vit_l": {"depth": 24, "embed_dim": 1024, "mlp_dim": 4096, "num_heads": 16},
        "vit_g": {"depth": 40, "embed_dim": 1536, "mlp_dim": 6144, "num_heads": 24},
    }
    return configs.get(variant, configs["vit_l"])
