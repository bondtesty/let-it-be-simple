import dataclasses
import logging
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.action_patchifier import PatchMethod
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    paligemma_num_layers: int | None = None
    action_expert_num_layers: int | None = (
        None  # Independent action expert depth (cond_gemma only). Defaults to paligemma_num_layers.
    )
    siglip_num_layers: int | None = None  # Truncate SigLiP encoder layers. None = use all (27 for So400m).

    # V-JEPA encoder config (alternative to SigLIP)
    use_vjepa: bool = False  # Use V-JEPA 2.1 instead of SigLIP
    vjepa_variant: str = "vit_l"  # "vit_b", "vit_l", "vit_g"
    vjepa_num_frames: int = 4  # Number of video frames to input

    cond_head_mode: Literal["pi0", "cond_gemma", "simvla"] = "pi0"

    # SimVLA head config (only used when cond_head_mode="simvla")
    simvla_depth: int = 4
    simvla_width: int = 768
    simvla_n_heads: int = 12
    simvla_mlp_dim: int = 3072
    simvla_use_perceiver: bool = True  # False = feed all VLM tokens to head (paper default)
    simvla_pos_injection: Literal["input", "per_layer", "rope"] = "input"
    simvla_pos_init_std: float = 0.0

    # Noise shift alpha for flow matching. None = standard Beta(1.5,1).
    # Effective when time_sampling is "beta" or "logit_normal".
    noise_shift_alpha: float | None = None

    # Time sampling strategy for flow matching training.
    # "beta" (default): Beta(1.5, 1), optionally combined with noise_shift_alpha.
    # "uniform": Uniform(0.001, 1.0).
    # "high_noise": Beta(3.0, 1.0), strongly biased toward high noise levels.
    # "logit_normal": sigmoid(N(mu, sigma)); can also be combined with noise_shift_alpha.
    time_sampling: Literal["beta", "uniform", "high_noise", "logit_normal"] = "beta"
    logit_normal_mu: float = 0.0
    logit_normal_sigma: float = 1.0

    # If True, training always uses pure noise (x_t = noise) instead of the
    # standard noise-action interpolation. The timestep is still sampled and
    # embedded, but the input action tokens carry no clean action information.
    # This tests single-step generation capability.
    pure_noise_input: bool = False

    # Initial action token source for flow sampling/training. "noise" is the
    # standard diffusion/flow prior; "learned" replaces it with a trainable
    # action-space token that is also used at inference.
    action_start_mode: Literal["noise", "learned"] = "noise"
    learned_action_start_init: Literal["zero", "normal"] = "zero"
    learned_action_start_init_std: float = 1.0
    # "interpolate" uses x_t = t * start + (1 - t) * action. "direct" uses
    # x_t = start and t = 1, which is intended for one-step x-prediction runs.
    action_start_training: Literal["interpolate", "direct"] = "interpolate"

    # Time shift for sampling. 0.0 = start at t=1.0 (default).
    # Positive values skip the early denoising steps:
    #   shift=0.3 → start at t=0.7, steps cover [0.7, 0.0]
    # This concentrates Euler steps on the higher-quality portion of the trajectory,
    # similar to how training biases toward lower t via Beta(1.5,1).
    sampling_time_shift: float = 0.0

    # Frequency-weighted loss (DeCo-inspired). 0.0 = disabled.
    freq_loss_weight: float = 0.0
    freq_loss_decay: float = 1.0  # Exponential decay rate for DCT frequency weighting.

    # UNITE-style stop-gradient: prevent denoising gradients from flowing into VLM encoder.
    stop_encoder_gradients: bool = False

    # Prediction type: "velocity" (default, v-pred) or "x_prediction" (predict clean data x_0).
    prediction_type: Literal["velocity", "x_prediction"] = "velocity"

    # Loss weighting: "uniform" (default), "v_weight" (1/t² — emphasize low-noise precision),
    # "min_snr" (min-SNR-gamma clamped weighting).
    loss_weighting: Literal["uniform", "v_weight", "min_snr"] = "uniform"

    # Action patchification: group P consecutive action steps into one token. 1 = no patching.
    action_patch_size: int = 1
    action_patch_method: PatchMethod = "concat"

    # Gripper-aware loss: extra weight for the gripper action dimension.
    # The gripper dim (binary open/close) is qualitatively different from continuous EE pose dims.
    # This multiplies the MSE loss on the specified dim by (1 + gripper_loss_weight).
    # 0.0 = no extra weight (default, same as treating all dims equally).
    gripper_loss_weight: float = 0.0
    gripper_dim: int = 6  # Which action dim is the gripper (after padding to action_dim).
    # If set, only the first N action dimensions contribute to the flow-matching loss.
    # This is useful for datasets such as Bridge where model inputs may be padded to
    # action_dim=32, but only the first 7 action dimensions are semantically real.
    action_loss_dim: int | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    n_cond_tokens: int | None = None
    prefix_merge_strategy: Literal["pre", "post"] = "pre"
    n_resampler_heads: int = 8

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.action_patch_size > 1 and self.action_horizon % self.action_patch_size != 0:
            raise ValueError(
                f"action_horizon ({self.action_horizon}) must be divisible by action_patch_size ({self.action_patch_size})"
            )
        if self.time_sampling not in ("beta", "logit_normal") and self.noise_shift_alpha is not None:
            logging.warning(
                f"time_sampling is '{self.time_sampling}', so noise_shift_alpha={self.noise_shift_alpha} "
                "will be ignored. noise_shift_alpha only takes effect when time_sampling is 'beta' or 'logit_normal'."
            )
        if self.logit_normal_sigma <= 0:
            raise ValueError(f"logit_normal_sigma must be positive, got {self.logit_normal_sigma}")
        if self.learned_action_start_init_std <= 0:
            raise ValueError(
                f"learned_action_start_init_std must be positive, got {self.learned_action_start_init_std}"
            )
        if self.action_loss_dim is not None and not (0 < self.action_loss_dim <= self.action_dim):
            raise ValueError(
                f"action_loss_dim must be in [1, action_dim], got action_loss_dim={self.action_loss_dim} "
                f"and action_dim={self.action_dim}"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
