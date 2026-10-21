import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Regex for missing keys that should be kept from random init rather than loaded from checkpoint.
    # Default keeps LoRA weights from random init. Use ".*" to keep all new parameters (e.g., for SCD).
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class TruncatedPaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint, truncating Gemma layers."""

    num_layers: int

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        loaded_params = _truncate_layer_params(loaded_params, self.num_layers)
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0FromPaliGemmaWeightLoader(WeightLoader):
    """Loads PaliGemma weights and initializes action expert from Gemma.

    This properly initializes a full PI0 model from PaliGemma by:
    1. Loading PaliGemma backbone (SigLIP vision + main Gemma LLM)
    2. Copying Gemma weights to initialize the action expert (_1 suffix layers)
    3. Leaving action projection layers (action_in_proj, etc.) randomly initialized

    Use this for fair "PI0 from scratch" experiments where both the main Gemma
    and action expert start from pretrained Gemma weights.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        paligemma_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]

        # Build loaded params with PaliGemma backbone
        loaded_params = {"PaliGemma": paligemma_params}

        # Flatten to copy Gemma weights to action expert (_1 suffix)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # Find all Gemma LLM layer weights and copy to action expert
        # Pattern: PaliGemma/llm/layers/X/Y -> PaliGemma/llm/layers/X/Y_1
        gemma_layer_keys = [k for k in flat_loaded if k.startswith("PaliGemma/llm/layers/")]
        for k in gemma_layer_keys:
            # Create action expert key by adding _1 suffix before the last component
            parts = k.rsplit("/", 1)
            if len(parts) == 2:
                action_expert_key = f"{parts[0]}/{parts[1]}_1" if "_1" not in parts[1] else None
                if action_expert_key and action_expert_key not in flat_loaded:
                    flat_loaded[action_expert_key] = flat_loaded[k].copy()
                    logger.info("Copied %s -> %s for action expert init", k, action_expert_key)

        # Also copy final_norm to final_norm_1
        if "PaliGemma/llm/final_norm/scale" in flat_loaded:
            flat_loaded["PaliGemma/llm/final_norm_1/scale"] = flat_loaded["PaliGemma/llm/final_norm/scale"].copy()
            logger.info("Copied final_norm -> final_norm_1 for action expert init")

        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
        # Add all missing weights (RAE projections, etc.) from random init
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class StackedWeightLoader(WeightLoader):
    """Chains multiple weight loaders, applying them sequentially.

    Useful when loading from multiple sources, e.g. PaliGemma base weights
    followed by a pretrained encoder/decoder checkpoint.
    """

    loaders: tuple[WeightLoader, ...]

    def load(self, params: at.Params) -> at.Params:
        for loader in self.loaders:
            params = loader.load(params)
        return params


@dataclasses.dataclass(frozen=True)
class TruncatedCheckpointWeightLoader(WeightLoader):
    """Loads weights from a checkpoint, truncating scanned layer params.

    For models with nn.scan-based layers (e.g., Gemma), the layer params have shape
    [depth, ...]. This loader slices them to [:num_layers, ...] to support loading
    a full checkpoint into a model with fewer layers.
    """

    params_path: str
    missing_regex: str = ".*"
    num_layers: int | None = None

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.num_layers is not None:
            loaded_params = _truncate_layer_params(loaded_params, self.num_layers)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class ShapeTolerantCheckpointWeightLoader(WeightLoader):
    """Loads weights from a checkpoint, skipping keys with shape mismatches.

    Useful for warm-starting a model with a different architecture dimension
    (e.g., changing action_horizon changes pos_embed shape). All keys with
    matching shapes are loaded; mismatched keys fall back to random init.
    """

    params_path: str
    missing_regex: str = ".*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_shape_tolerant(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class TinyModelWeightLoader(WeightLoader):
    """Loads pretrained SigLiP weights for tiny models with different Gemma width.

    Truncates SigLiP encoder to siglip_num_layers, then uses shape-tolerant merge
    which skips Gemma weights (embedder, layers) that have width mismatches.
    Only SigLiP encoder blocks with matching shapes are loaded.
    """

    siglip_num_layers: int

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        loaded_params = _truncate_scanned_params(loaded_params, self.siglip_num_layers)
        return _merge_params_shape_tolerant(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class SelectiveTruncationWeightLoader(WeightLoader):
    """Loads PaliGemma weights with independent Gemma and SigLiP truncation depths.

    Unlike TruncatedPaliGemmaWeightLoader which only truncates Gemma layers
    (keys with /layers/), this loader truncates both the Gemma LLM backbone
    and the SigLiP vision encoder (keys with /encoderblock/) to independently
    specified depths. This is needed for experiments that use more SigLiP layers
    than Gemma layers (e.g., siglip_num_layers=12 with paligemma_num_layers=4).
    """

    gemma_num_layers: int
    siglip_num_layers: int

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        loaded_params = _truncate_params_selective(loaded_params, self.gemma_num_layers, self.siglip_num_layers)
        return _merge_params_shape_tolerant(loaded_params, params, missing_regex=".*")


def _truncate_params_selective(
    params: at.Params, gemma_num_layers: int, siglip_num_layers: int
) -> at.Params:
    """Truncate Gemma (/layers/) and SigLiP (/encoderblock/) to independent depths.

    Unlike _truncate_scanned_params which truncates both to the same depth,
    this allows different truncation depths for the LLM backbone vs the vision encoder.
    """
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    result = {}
    for k, v in flat.items():
        if "/layers/" in k and hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] > gemma_num_layers:
            result[k] = v[:gemma_num_layers]
        elif "/encoderblock/" in k and hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] > siglip_num_layers:
            result[k] = v[:siglip_num_layers]
        else:
            result[k] = v
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _truncate_scanned_params(params: at.Params, num_layers: int) -> at.Params:
    """Truncate scanned layer params (Gemma /layers/ and SigLiP /encoderblock/)."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    result = {}
    for k, v in flat.items():
        if ("/layers/" in k or "/encoderblock/" in k) and hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] > num_layers:
            result[k] = v[:num_layers]
        else:
            result[k] = v
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _truncate_layer_params(params: at.Params, num_layers: int) -> at.Params:
    """Truncate scanned layer params from [depth, ...] to [:num_layers, ...].

    Finds all keys containing '/layers/' (the nn.scan axis) and slices the first
    dimension. Other params are left unchanged.
    """
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    result = {}
    for k, v in flat.items():
        if "/layers/" in k and hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] > num_layers:
            result[k] = v[:num_layers]
        else:
            result[k] = v
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_shape_tolerant(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges loaded parameters, skipping keys with shape mismatches.

    Like _merge_params, but when a key exists in both loaded and reference params
    with different shapes, the loaded value is skipped (falls back to random init).
    Logs warnings for skipped keys.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_v = flat_ref[k]
            if hasattr(v, "shape") and hasattr(ref_v, "shape") and v.shape != ref_v.shape:
                logger.warning("Shape mismatch for %s: checkpoint %s vs model %s, skipping (random init)", k, v.shape, ref_v.shape)
                continue
            result[k] = v.astype(ref_v.dtype) if v.dtype != ref_v.dtype else v

    flat_loaded.clear()

    # Merge missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class VJEPAWeightLoader(WeightLoader):
    """Loads V-JEPA 2.1 encoder weights from converted JAX checkpoint.

    The checkpoint should be a pickle file containing flattened V-JEPA encoder parameters
    converted from the PyTorch checkpoint using scripts/convert_vjepa_torch_to_jax.py.

    Example:
        VJEPAWeightLoader(
            checkpoint_path="/path/to/vjepa2_1_vitl_jax.pkl",
            freeze_encoder=True,  # Freeze V-JEPA backbone during training
        )
    """

    checkpoint_path: str
    freeze_encoder: bool = False  # If True, sets require_grad=False (handled by freeze filter)

    def load(self, params: at.Params) -> at.Params:
        import pickle

        logger.info("Loading V-JEPA encoder weights from %s", self.checkpoint_path)

        # Load the converted JAX checkpoint
        with open(self.checkpoint_path, "rb") as f:
            vjepa_params = pickle.load(f)

        # V-JEPA params are already flattened with "encoder/..." keys
        # Need to wrap them under PaliGemma/img for PI0 compatibility
        loaded_params = {"PaliGemma": {"img": {"params": vjepa_params}}}

        # Merge with existing params, keeping random init for non-matching keys
        return _merge_params(loaded_params, params, missing_regex=".*")
