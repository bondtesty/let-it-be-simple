import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.action_patchifier import create_patchifier
import openpi.models.gemma as _gemma
from openpi.models.perceiver_resampler import PerceiverResampler
import openpi.models.siglip as _siglip
from openpi.models.simvla_head import SimVLAHead
from openpi.models.simvla_head import SimVLAHeadConfig
import openpi.models.vjepa as _vjepa
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.cond_head_mode = config.cond_head_mode
        self.n_cond_tokens = config.n_cond_tokens
        self.prefix_merge_strategy = config.prefix_merge_strategy
        self.simvla_use_perceiver = getattr(config, "simvla_use_perceiver", True)
        self.use_vjepa = config.use_vjepa
        self.vjepa_num_frames = config.vjepa_num_frames

        if self.cond_head_mode == "cond_gemma":
            if config.pi05:
                raise ValueError("cond_head_mode='cond_gemma' is only supported for PI0 (pi05=False)")
            if config.n_cond_tokens is None:
                raise ValueError("cond_head_mode='cond_gemma' requires n_cond_tokens to be set")
            if config.prefix_merge_strategy != "post":
                raise ValueError("cond_head_mode='cond_gemma' requires prefix_merge_strategy='post'")
        elif self.cond_head_mode == "simvla":
            if config.pi05:
                raise ValueError("cond_head_mode='simvla' is only supported for PI0 (pi05=False)")
            if config.simvla_use_perceiver and config.n_cond_tokens is None:
                raise ValueError("cond_head_mode='simvla' with perceiver requires n_cond_tokens to be set")
            if config.prefix_merge_strategy != "post":
                raise ValueError("cond_head_mode='simvla' requires prefix_merge_strategy='post'")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        if config.paligemma_num_layers is not None:
            k = config.paligemma_num_layers
            if k <= 0:
                raise ValueError("paligemma_num_layers must be positive")
            if k > paligemma_config.depth:
                raise ValueError(f"paligemma_num_layers ({k}) must be <= paligemma depth ({paligemma_config.depth})")
            paligemma_config = dataclasses.replace(paligemma_config, depth=k)
            # Only couple action expert depth if action_expert_num_layers is not set
            if config.action_expert_num_layers is None:
                if k > action_expert_config.depth:
                    raise ValueError(
                        f"paligemma_num_layers ({k}) must be <= action expert depth ({action_expert_config.depth})"
                    )
                action_expert_config = dataclasses.replace(action_expert_config, depth=k)

        if config.action_expert_num_layers is not None:
            a = config.action_expert_num_layers
            if a <= 0:
                raise ValueError("action_expert_num_layers must be positive")
            if a > _gemma.get_config(config.action_expert_variant).depth:
                raise ValueError(f"action_expert_num_layers ({a}) exceeds action expert max depth")
            action_expert_config = dataclasses.replace(action_expert_config, depth=a)

        # Whether encoder and action expert use separate Gemma modules (needed when depths differ)
        self._separate_action_expert = (
            self.cond_head_mode == "cond_gemma" and paligemma_config.depth != action_expert_config.depth
        )

        if config.n_cond_tokens is not None and (self.cond_head_mode != "simvla" or config.simvla_use_perceiver):
            if paligemma_config.width % config.n_resampler_heads != 0:
                raise ValueError(
                    f"n_resampler_heads ({config.n_resampler_heads}) must divide paligemma width ({paligemma_config.width})"
                )
            self.prefix_resampler = PerceiverResampler(
                n_queries=config.n_cond_tokens,
                dim=paligemma_config.width,
                n_heads=config.n_resampler_heads,
                rngs=rngs,
            )
        if self.cond_head_mode == "cond_gemma":
            self.cond_proj = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
        elif self.cond_head_mode == "simvla":
            simvla_width = config.simvla_width
            self.cond_proj = nnx.Linear(paligemma_config.width, simvla_width, rngs=rngs)

        # Determine the action head width for projection layers.
        head_width = config.simvla_width if self.cond_head_mode == "simvla" else action_expert_config.width
        self.patchifier = create_patchifier(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            patch_size=config.action_patch_size,
            method=config.action_patch_method,
            rngs=rngs,
        )
        patch_dim = self.patchifier.patch_dim

        # TODO: rewrite gemma in NNX. For now, use bridge.
        if self.cond_head_mode == "simvla":
            # SimVLA: encoder-only LLM + vanilla Transformer action head
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config],
                    embed_dtype=config.dtype,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
            action_llm = None
            # Create SimVLA head
            simvla_head_config = SimVLAHeadConfig(
                depth=config.simvla_depth,
                width=config.simvla_width,
                n_heads=config.simvla_n_heads,
                mlp_dim=config.simvla_mlp_dim,
                pos_injection=config.simvla_pos_injection,
                pos_init_std=config.simvla_pos_init_std,
            )
            # max seq len depends on whether perceiver is used
            num_act_tok = config.action_horizon // config.action_patch_size
            # V-JEPA produces more tokens than SigLIP (392 vs 196 per image with 4 frames)
            vision_tokens_per_image = 392 if config.use_vjepa else 256
            if config.simvla_use_perceiver:
                max_seq_len = (config.n_cond_tokens or 8) + 1 + num_act_tok
            else:
                # No perceiver: all VLM tokens (max_token_len text + 3*vision_tokens image) + state + actions
                max_seq_len = config.max_token_len + 3 * vision_tokens_per_image + 1 + num_act_tok
            self.simvla_head = SimVLAHead(simvla_head_config, max_seq_len=max_seq_len, rngs=rngs)
        elif self._separate_action_expert:
            # Separate modules for encoder and action expert (different depths)
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config],
                    embed_dtype=config.dtype,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
            action_llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[action_expert_config],
                    embed_dtype=config.dtype,
                )
            )
            action_llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        else:
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
            action_llm = None
        # Vision encoder: SigLIP or V-JEPA
        if config.use_vjepa:
            # V-JEPA 2.1 video encoder
            img = nnx_bridge.ToNNX(
                _vjepa.Module(
                    variant=config.vjepa_variant,
                    num_classes=paligemma_config.width,
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )
            # V-JEPA expects video input (B, T, C, H, W) or image (B, C, H, W)
            # Create dummy input for lazy_init
            if config.vjepa_num_frames > 1:
                # Video input
                dummy_input = jnp.zeros((1, config.vjepa_num_frames, *_model.IMAGE_RESOLUTION, 3))
            else:
                # Image input
                dummy_input = jnp.zeros((1, *_model.IMAGE_RESOLUTION, 3))
            img.lazy_init(dummy_input, train=False, rngs=rngs)
        else:
            # SigLIP image encoder
            siglip_kw = {}
            if config.siglip_num_layers is not None:
                siglip_kw["depth"] = config.siglip_num_layers
            img = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                    **siglip_kw,
                )
            )
            img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        if action_llm is not None:
            self.PaliGemma = nnx.Dict(llm=llm, img=img, action_llm=action_llm)
        else:
            self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(patch_dim, head_width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(head_width, head_width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(head_width, head_width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, head_width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * head_width, head_width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(head_width, head_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(head_width, patch_dim, rngs=rngs)

        self.noise_shift_alpha = config.noise_shift_alpha
        self.time_sampling = config.time_sampling
        self.logit_normal_mu = config.logit_normal_mu
        self.logit_normal_sigma = config.logit_normal_sigma
        self.pure_noise_input = config.pure_noise_input
        self.action_start_mode = config.action_start_mode
        self.action_start_training = config.action_start_training
        self.freq_loss_weight = config.freq_loss_weight
        self.freq_loss_decay = config.freq_loss_decay
        self.stop_encoder_gradients = config.stop_encoder_gradients
        self.prediction_type = config.prediction_type
        self.loss_weighting = config.loss_weighting
        self.gripper_loss_weight = config.gripper_loss_weight
        self.gripper_dim = config.gripper_dim
        self.action_loss_dim = config.action_loss_dim
        self.num_action_tokens = self.patchifier.num_tokens
        if config.action_start_mode == "learned":
            if config.learned_action_start_init == "zero":
                start = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
            elif config.learned_action_start_init == "normal":
                start = (
                    jax.random.normal(rngs.params(), (1, config.action_horizon, config.action_dim), dtype=jnp.float32)
                    * config.learned_action_start_init_std
                )
            else:
                raise ValueError(f"Unknown learned_action_start_init: {config.learned_action_start_init}")
            self.learned_action_start = nnx.Param(start)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images (or video clips for V-JEPA)
        for name in obs.images:
            image = obs.images[name]
            # Data loader now provides proper video frames for V-JEPA (B, T, H, W, C)
            image_tokens, _ = self.PaliGemma.img(image, train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        if self.n_cond_tokens is not None and self.prefix_merge_strategy == "pre":
            tokens = self.prefix_resampler(tokens, input_mask)
            input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
            ar_mask = jnp.zeros((tokens.shape[1],), dtype=jnp.bool_)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        # Patchify: group P consecutive action steps into one token
        noisy_actions = self.patchifier.patchify(noisy_actions)
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=action_tokens.shape[1])
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.num_action_tokens - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _unpatchify(self, x):
        """Unpatchify action output: (B, H//P, patch_dim) → (B, H, D)."""
        return self.patchifier.unpatchify(x)

    def _action_dim_weights(self, dim: int, dtype):
        weights = jnp.ones((dim,), dtype=dtype)
        if self.action_loss_dim is not None:
            weights = weights * (jnp.arange(dim) < self.action_loss_dim)
        if self.gripper_loss_weight > 0:
            weights = weights.at[self.gripper_dim].set(weights[self.gripper_dim] * (1.0 + self.gripper_loss_weight))
        return weights

    def _freq_weighted_loss(self, v_t, u_t):
        """Frequency-weighted MSE using 1D DCT along temporal axis (DeCo-inspired)."""
        horizon = v_t.shape[-2]
        v_coeff = jax.scipy.fft.dct(v_t, type=2, axis=-2, norm="ortho")
        u_coeff = jax.scipy.fft.dct(u_t, type=2, axis=-2, norm="ortho")
        freq_idx = jnp.arange(horizon, dtype=v_t.dtype)
        weights = jnp.exp(-freq_idx * self.freq_loss_decay)
        weights = weights / jnp.mean(weights)
        dim_weights = self._action_dim_weights(v_t.shape[-1], v_t.dtype)
        freq_sq_err = jnp.square(v_coeff - u_coeff) * dim_weights
        freq_mse = jnp.sum(freq_sq_err, axis=-1) / jnp.maximum(jnp.sum(dim_weights), 1.0)
        return freq_mse * weights[None, :]

    def _apply_freq_loss(self, v_t, u_t):
        """Compute spatial MSE + optional frequency-weighted loss, with optional gripper dim weighting."""
        dim_weights = self._action_dim_weights(v_t.shape[-1], v_t.dtype)
        sq_err = jnp.square(v_t - u_t) * dim_weights
        spatial_loss = jnp.sum(sq_err, axis=-1) / jnp.maximum(jnp.sum(dim_weights), 1.0)
        if self.freq_loss_weight > 0:
            return spatial_loss + self.freq_loss_weight * self._freq_weighted_loss(v_t, u_t)
        return spatial_loss

    def _apply_loss_weighting(self, loss, time):
        """Apply time-dependent loss weighting. loss: (*b, ah), time: (*b,)."""
        if self.loss_weighting == "v_weight":
            # v-prediction loss weight for x-prediction: 1/t² (emphasize low-noise precision)
            w = jnp.clip(1.0 / jnp.square(time + 1e-3), max=25.0)
            w = w / jnp.mean(w)
            return loss * w[..., None]
        if self.loss_weighting == "min_snr":
            # min-SNR-gamma: clamp SNR to prevent over-weighting low-noise samples
            snr = jnp.square(1.0 - time) / jnp.square(time + 1e-3)
            w = jnp.minimum(snr, 5.0) / (snr + 1e-3)
            w = w / jnp.mean(w)
            return loss * w[..., None]
        return loss

    def _sample_action_start(self, rng: at.KeyArrayLike, shape: tuple[int, ...]):
        if self.action_start_mode == "learned":
            start = self.learned_action_start.value
            return jnp.broadcast_to(start, shape).astype(jnp.float32)
        return jax.random.normal(rng, shape)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = self._sample_action_start(noise_rng, actions.shape)
        if self.time_sampling == "beta":
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            # Noise shift: bias timestep distribution toward higher noise levels
            if self.noise_shift_alpha is not None:
                alpha = self.noise_shift_alpha
                time = (alpha * time) / (1.0 + (alpha - 1.0) * time)
        elif self.time_sampling == "uniform":
            time = jax.random.uniform(time_rng, batch_shape, minval=0.001, maxval=1.0)
        elif self.time_sampling == "high_noise":
            time = jax.random.beta(time_rng, 3.0, 1.0, batch_shape) * 0.999 + 0.001
        elif self.time_sampling == "logit_normal":
            logit_time = jax.random.normal(time_rng, batch_shape) * self.logit_normal_sigma + self.logit_normal_mu
            time = jnp.clip(jax.nn.sigmoid(logit_time), 0.001, 1.0)
            if self.noise_shift_alpha is not None:
                alpha = self.noise_shift_alpha
                time = (alpha * time) / (1.0 + (alpha - 1.0) * time)
        else:
            raise ValueError(f"Unknown time_sampling: {self.time_sampling}")
        time_expanded = time[..., None, None]
        if self.action_start_training == "direct":
            x_t = noise
            time = jnp.ones(batch_shape, dtype=actions.dtype)
        else:
            x_t = noise if self.pure_noise_input else time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # For x-prediction, the target is the clean data, not the velocity
        if self.prediction_type == "x_prediction":
            u_t = actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        if self.cond_head_mode == "cond_gemma":
            return self._compute_loss_cond_gemma(
                prefix_tokens, prefix_mask, prefix_ar_mask, suffix_tokens, suffix_mask, suffix_ar_mask, u_t, time
            )

        if self.cond_head_mode == "simvla":
            return self._compute_loss_simvla(
                prefix_tokens, prefix_mask, prefix_ar_mask, suffix_tokens, suffix_mask, u_t, time
            )

        return self._compute_loss_default(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
            u_t,
            time,
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        sampling_time_shift: float = 0.0,
        solver: str = "euler",
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self._sample_action_start(rng, (batch_size, self.action_horizon, self.action_dim))

        # time_start: where on [1,0] the Euler walk begins. Default 1.0 (full trajectory).
        # Positive shift skips early denoising steps — e.g. shift=0.3 → start at t=0.7.
        time_start = 1.0 - sampling_time_shift

        # first fill KV cache with a forward pass of the (possibly compressed) prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        if self.cond_head_mode == "cond_gemma":
            if solver != "euler":
                raise ValueError(f"solver={solver!r} is only implemented for the simvla sampling branch")
            return self._sample_actions_cond_gemma(
                prefix_tokens, prefix_mask, prefix_ar_mask, observation, noise, batch_size, dt, time_start
            )

        if self.cond_head_mode == "simvla":
            return self._sample_actions_simvla(
                prefix_tokens, prefix_mask, prefix_ar_mask, observation, noise, batch_size, dt, time_start, solver
            )

        if solver != "euler":
            raise ValueError(f"solver={solver!r} is only implemented for the simvla sampling branch")
        return self._sample_actions_default(
            prefix_tokens, prefix_mask, prefix_ar_mask, observation, noise, batch_size, dt, time_start
        )

    # === compute_loss branch helpers ===

    def _compute_loss_cond_gemma(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        u_t,
        time,
    ):
        # === COND_GEMMA: compute_loss ===
        # Prefix encoding → concat → action LLM → unpatchify → apply losses
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if self._separate_action_expert:
            (prefix_out,), _ = self.PaliGemma.llm(
                [prefix_tokens],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )
        else:
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )

        cond_tokens_2b = self.prefix_resampler(prefix_out, prefix_mask)
        cond_tokens = self.cond_proj(cond_tokens_2b)
        cond_mask = jnp.ones(cond_tokens.shape[:2], dtype=jnp.bool_)
        cond_ar_mask = jnp.zeros((cond_tokens.shape[1],), dtype=jnp.bool_)

        head_tokens = jnp.concatenate([cond_tokens, suffix_tokens], axis=1)
        input_mask = jnp.concatenate([cond_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([cond_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        if self._separate_action_expert:
            (head_out,), _ = self.PaliGemma.action_llm(
                [head_tokens],
                mask=attn_mask,
                positions=positions,
            )
        else:
            (_, head_out), _ = self.PaliGemma.llm(
                [None, head_tokens],
                mask=attn_mask,
                positions=positions,
            )
        v_t = self._unpatchify(self.action_out_proj(head_out[:, -self.num_action_tokens :]))

        return self._apply_loss_weighting(self._apply_freq_loss(v_t, u_t), time)

    def _compute_loss_simvla(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        u_t,
        time,
    ):
        # === SIMVLA: compute_loss ===
        # PaliGemma encoder → (optional Perceiver) → SimVLA Transformer head → unpatchify → apply losses
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out,), _ = self.PaliGemma.llm(
            [prefix_tokens],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        if self.stop_encoder_gradients:
            prefix_out = jax.lax.stop_gradient(prefix_out)

        if self.simvla_use_perceiver:
            cond_tokens_2b = self.prefix_resampler(prefix_out, prefix_mask)
            cond_tokens = self.cond_proj(cond_tokens_2b)
            cond_mask = jnp.ones(cond_tokens.shape[:2], dtype=jnp.bool_)
        else:
            cond_tokens = self.cond_proj(prefix_out)
            cond_mask = prefix_mask

        head_tokens = jnp.concatenate([cond_tokens, suffix_tokens], axis=1)
        input_mask = jnp.concatenate([cond_mask, suffix_mask], axis=1)

        head_out = self.simvla_head(head_tokens, mask=input_mask)
        v_t = self._unpatchify(self.action_out_proj(head_out[:, -self.num_action_tokens :]))

        return self._apply_loss_weighting(self._apply_freq_loss(v_t, u_t), time)

    def _compute_loss_default(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
        u_t,
        time,
    ):
        # === DEFAULT: compute_loss ===
        # (prefix merge post + cond tokens) OR (no merge, full concat) → unpatchify → apply losses
        if self.n_cond_tokens is not None and self.prefix_merge_strategy == "post":
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )

            cond_tokens = self.prefix_resampler(prefix_out, prefix_mask)
            cond_mask = jnp.ones(cond_tokens.shape[:2], dtype=jnp.bool_)
            cond_ar_mask = jnp.zeros((cond_tokens.shape[1],), dtype=jnp.bool_)

            input_mask = jnp.concatenate([cond_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([cond_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [cond_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
        v_t = self._unpatchify(self.action_out_proj(suffix_out[:, -self.num_action_tokens :]))

        return self._apply_loss_weighting(self._apply_freq_loss(v_t, u_t), time)

    # === sample_actions branch helpers ===

    def _sample_actions_cond_gemma(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        observation,
        noise,
        batch_size,
        dt,
        time_start: float = 1.0,
    ):
        # === COND_GEMMA: sample_actions ===
        # Prefix encoding → concat → action LLM (per step) → unpatchify → Euler step
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if self._separate_action_expert:
            (prefix_out,), _ = self.PaliGemma.llm([prefix_tokens], mask=prefix_attn_mask, positions=positions)
        else:
            (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        cond_tokens_2b = self.prefix_resampler(prefix_out, prefix_mask)
        cond_tokens = self.cond_proj(cond_tokens_2b)
        cond_mask = jnp.ones(cond_tokens.shape[:2], dtype=jnp.bool_)
        cond_ar_mask = jnp.zeros((cond_tokens.shape[1],), dtype=jnp.bool_)

        _separate = self._separate_action_expert
        _x_pred_cg = self.prediction_type == "x_prediction"

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _ = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            head_tokens = jnp.concatenate([cond_tokens, suffix_tokens], axis=1)
            input_mask = jnp.concatenate([cond_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([cond_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            if _separate:
                (head_out,), _ = self.PaliGemma.action_llm([head_tokens], mask=attn_mask, positions=positions)
            else:
                (_, head_out), _ = self.PaliGemma.llm([None, head_tokens], mask=attn_mask, positions=positions)
            out = self._unpatchify(self.action_out_proj(head_out[:, -self.num_action_tokens :]))

            if _x_pred_cg:  # noqa: SIM108
                v_t = (x_t - out) / jnp.maximum(time, 1e-6)
            else:
                v_t = out

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, time_start))
        return x_0

    def _sample_actions_simvla(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        observation,
        noise,
        batch_size,
        dt,
        time_start: float = 1.0,
        solver: str = "euler",
    ):
        # === SIMVLA: sample_actions ===
        # Encode prefix once → run SimVLA head per denoising step → unpatchify → ODE step.
        # solver: "euler" (N head evals), "heun" (explicit trapezoidal RK2, 2N evals),
        # or "rk4" (classic 4th-order Runge-Kutta, 4N evals).
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out,), _ = self.PaliGemma.llm([prefix_tokens], mask=prefix_attn_mask, positions=positions)

        if self.simvla_use_perceiver:
            cond_tokens_2b = self.prefix_resampler(prefix_out, prefix_mask)
            cond_tokens = self.cond_proj(cond_tokens_2b)
            cond_mask = jnp.ones(cond_tokens.shape[:2], dtype=jnp.bool_)
        else:
            cond_tokens = self.cond_proj(prefix_out)
            cond_mask = prefix_mask

        _x_pred = self.prediction_type == "x_prediction"

        def velocity(x_t, time):
            suffix_tokens, suffix_mask, _, _ = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            head_tokens = jnp.concatenate([cond_tokens, suffix_tokens], axis=1)
            input_mask = jnp.concatenate([cond_mask, suffix_mask], axis=1)

            head_out = self.simvla_head(head_tokens, mask=input_mask)
            out = self._unpatchify(self.action_out_proj(head_out[:, -self.num_action_tokens :]))

            if _x_pred:
                # x-prediction: model outputs x_0_hat, convert to velocity
                return (x_t - out) / jnp.maximum(time, 1e-6)
            return out

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        if solver == "euler":

            def step(carry):
                x_t, time, _ = carry
                v_t = velocity(x_t, time)
                return x_t + dt * v_t, time + dt, v_t

            x_0, _, _ = jax.lax.while_loop(cond, step, (noise, time_start, jnp.zeros_like(noise)))
            return x_0

        if solver == "heun":
            # Explicit trapezoidal RK2. (No FSAL: k2 is evaluated at the Euler-predicted point,
            # not at the corrected point, so it cannot be reused as the next step's k1.)
            def step(carry):
                x_t, time, _ = carry
                k1 = velocity(x_t, time)
                k2 = velocity(x_t + dt * k1, time + dt)
                return x_t + dt * (k1 + k2) / 2, time + dt, k1

            x_0, _, _ = jax.lax.while_loop(cond, step, (noise, time_start, jnp.zeros_like(noise)))
            return x_0

        if solver == "rk4":

            def step(carry):
                x_t, time, _ = carry
                k1 = velocity(x_t, time)
                k2 = velocity(x_t + dt * k1 / 2, time + dt / 2)
                k3 = velocity(x_t + dt * k2 / 2, time + dt / 2)
                k4 = velocity(x_t + dt * k3, time + dt)
                return x_t + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, time + dt, k1

            x_0, _, _ = jax.lax.while_loop(cond, step, (noise, time_start, jnp.zeros_like(noise)))
            return x_0

        raise ValueError(f"Unsupported solver {solver!r}; expected one of: euler, heun, rk4")

    def _sample_actions_default(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        observation,
        noise,
        batch_size,
        dt,
        time_start: float = 1.0,
    ):
        # === DEFAULT: sample_actions ===
        # (prefix merge post) OR (no merge) → KV-cache prefix → per-step suffix → unpatchify → Euler step
        if self.n_cond_tokens is not None and self.prefix_merge_strategy == "post":
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

            prefix_tokens = self.prefix_resampler(prefix_out, prefix_mask)
            prefix_mask = jnp.ones(prefix_tokens.shape[:2], dtype=jnp.bool_)
            prefix_ar_mask = jnp.zeros((prefix_tokens.shape[1],), dtype=jnp.bool_)

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        _x_pred_std = self.prediction_type == "x_prediction"

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            out = self._unpatchify(self.action_out_proj(suffix_out[:, -self.num_action_tokens :]))

            if _x_pred_std:  # noqa: SIM108
                v_t = (x_t - out) / jnp.maximum(time, 1e-6)
            else:
                v_t = out

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, time_start))
        return x_0
