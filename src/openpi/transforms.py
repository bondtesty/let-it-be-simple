from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats
# Internal transform key: retained for action-coordinate reconstruction, but ignored by model observations.
ACTION_REFERENCE_STATE = "action_reference_state"


T = TypeVar("T")
S = TypeVar("S")


def _resize_with_pad_numpy(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """CPU-only resize-with-pad for dataloader workers."""
    try:
        import cv2
    except ImportError:
        return image_tools.resize_with_pad(images, height, width)

    cv2.setNumThreads(1)
    images = np.asarray(images)
    if images.ndim not in (3, 4, 5):
        raise ValueError(f"Unexpected image ndim={images.ndim}")

    if images.dtype == np.uint8:
        pad_value = 0
    elif images.dtype == np.float32:
        pad_value = -1.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    original_shape = images.shape
    has_batch_dim = images.ndim >= 4
    is_video = images.ndim == 5

    if is_video or has_batch_dim:
        flat = images.reshape(-1, *images.shape[-3:])
    else:
        flat = images[None]

    cur_height, cur_width = flat.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    out = np.empty((flat.shape[0], height, width, flat.shape[-1]), dtype=images.dtype)
    for i, image in enumerate(flat):
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[..., None]
        if images.dtype == np.float32:
            resized = np.clip(resized, -1.0, 1.0)
        out[i] = np.pad(
            resized,
            ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

    if is_video or has_batch_dim:
        return out.reshape(*original_shape[:-3], height, width, original_shape[-1])
    return out[0]


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


_DIT_FEATURES_CACHE: dict[str, np.ndarray] = {}


@dataclasses.dataclass(frozen=True)
class InjectDiTFeatures(DataTransformFn):
    """Inject precomputed DreamDojo DiT features by global frame index."""

    features_path: str

    def _get_features(self) -> np.ndarray:
        if self.features_path not in _DIT_FEATURES_CACHE:
            _DIT_FEATURES_CACHE[self.features_path] = np.load(self.features_path, mmap_mode="r")
        return _DIT_FEATURES_CACHE[self.features_path]

    def __call__(self, data: DataDict) -> DataDict:
        idx = int(data["index"])
        data["dit_features"] = np.array(self._get_features()[idx], dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # Optional flattened keys that should use quantile normalization even when
    # use_quantiles is false. This is useful for Bridge action-only q01/q99
    # normalization without changing proprio/state normalization.
    quantile_keys: frozenset[str] = frozenset()
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None:
            if self.use_quantiles:
                _assert_quantile_stats(self.norm_stats)
            elif self.quantile_keys:
                _assert_quantile_stats_for_keys(self.norm_stats, self.quantile_keys)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        data = self._apply(data, strict=self.strict)

        # Normalize state_history using the same stats as state if available.
        if "state_history" in data and "state" in (self.norm_stats or {}):
            fn = self._normalize_quantile if self.use_quantiles else self._normalize
            data["state_history"] = fn(data["state_history"], self.norm_stats["state"])

        return data

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _apply(self, data: DataDict, *, strict: bool) -> DataDict:
        if self.use_quantiles or not self.quantile_keys:
            return apply_tree(
                data,
                self.norm_stats,
                self._normalize_quantile if self.use_quantiles else self._normalize,
                strict=strict,
            )

        tree = flatten_dict(data)
        selector = flatten_dict(self.norm_stats)

        if strict:
            for k in selector:
                if k not in tree:
                    raise ValueError(f"Selector key {k} not found in tree")

        def transform(k: str, v):
            if k not in selector:
                return v
            fn = self._normalize_quantile if k in self.quantile_keys else self._normalize
            return fn(v, selector[k])

        return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    quantile_keys: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.norm_stats is not None:
            if self.use_quantiles:
                _assert_quantile_stats(self.norm_stats)
            elif self.quantile_keys:
                _assert_quantile_stats_for_keys(self.norm_stats, self.quantile_keys)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        if self.use_quantiles or not self.quantile_keys:
            return apply_tree(
                data,
                self.norm_stats,
                self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
                strict=True,
            )

        tree = flatten_dict(data)
        selector = flatten_dict(self.norm_stats)

        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

        def transform(k: str, v):
            if k not in selector:
                return v
            fn = self._unnormalize_quantile if k in self.quantile_keys else self._unnormalize
            return fn(v, selector[k])

        return unflatten_dict({k: transform(k, v) for k, v in tree.items()})

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        resized = {}
        for k, raw_img in data["image"].items():
            # Copy the loop var so we can reassign `img` freely below.
            img = raw_img.numpy() if hasattr(raw_img, "numpy") else raw_img
            img = np.ascontiguousarray(img)

            if img.ndim == 4:
                # Video input: could be (T, C, H, W) or (T, H, W, C).
                # Detect format by checking which axis has the channel count.
                if img.shape[1] in (3, 4):
                    # (T, C, H, W) - transpose to (T, H, W, C)
                    img = np.transpose(img, (0, 2, 3, 1))
                elif img.shape[-1] not in (3, 4):
                    # Neither axis has valid channels - skip
                    continue
                num_channels = img.shape[-1]
                if num_channels not in (3, 4):
                    continue
                resized[k] = _resize_with_pad_numpy(img, self.height, self.width)
            elif img.ndim == 3:
                # Image input: could be (C, H, W) or (H, W, C).
                # Detect format by checking which axis has the channel count.
                if img.shape[0] in (3, 4):
                    # (C, H, W) - transpose to (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[-1] not in (3, 4):
                    # Neither axis has valid channels - skip
                    continue
                num_channels = img.shape[-1]
                if num_channels not in (3, 4):
                    continue
                resized[k] = _resize_with_pad_numpy(img, self.height, self.width)
            else:
                raise ValueError(f"Unexpected image ndim={img.ndim} for key={k}, shape={img.shape}")
        data["image"] = resized
        return data


@dataclasses.dataclass(frozen=True)
class StackFramesToVideo(DataTransformFn):
    """Stack frame history into video format for V-JEPA encoder.

    When frame_history_steps > 0, LeRobot returns frames with shape (T, H, W, C).
    This transform ensures proper format for V-JEPA video encoder.

    Args:
        axis: Axis to stack frames. Default -1 stacks along channel dimension (H, W, T*C).
              Use axis=0 to keep frames separate (T, H, W, C) for V-JEPA.
        axis: int = 0 means output is (T, H, W, C) for video encoder.
    """

    axis: int = 0

    def __call__(self, data: DataDict) -> DataDict:
        # Images from LeRobot with delta_timestamps already have shape (T, H, W, C)
        # We just need to ensure the format is correct
        for k, v in data.get("image", {}).items():
            if v.ndim == 3:
                # Single frame (H, W, C) -> add time dimension (1, H, W, C)
                data["image"][k] = v[np.newaxis, ...]
                # Also expand image_masks to match
                if "image_masks" in data and k in data["image_masks"]:
                    # mask is (B,) -> (B, 1)
                    data["image_masks"][k] = data["image_masks"][k][:, np.newaxis]
            elif v.ndim == 4:
                # Already has time dimension (T, H, W, C)
                # Expand image_masks to (B, T)
                if "image_masks" in data and k in data["image_masks"]:
                    timesteps = v.shape[0]
                    # mask is (B,) -> (B, T) by repeating
                    data["image_masks"][k] = np.repeat(data["image_masks"][k][:, np.newaxis], timesteps, axis=1)
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if ACTION_REFERENCE_STATE in data:
            data[ACTION_REFERENCE_STATE] = pad_to_dim(data[ACTION_REFERENCE_STATE], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        if "state_history" in data:
            data["state_history"] = pad_to_dim(data["state_history"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class ZeroState(DataTransformFn):
    """Replace proprio state with zeros after normalization for state-free Bridge ablations."""

    def __call__(self, data: DataDict) -> DataDict:
        if "state" in data:
            data[ACTION_REFERENCE_STATE] = np.array(data["state"], copy=True)
            data["state"] = np.zeros_like(data["state"])
        if "state_history" in data:
            data["state_history"] = np.zeros_like(data["state_history"])
        return data


ConditionAblationMode: TypeAlias = Literal["full", "no_wrist", "no_state", "no_prompt", "no_image", "no_image_no_state"]


@dataclasses.dataclass(frozen=True)
class ConditionAblation(DataTransformFn):
    """Remove selected conditioning sources after normalization and image resize."""

    mode: ConditionAblationMode = "full"
    replacement_prompt: str = ""

    def __call__(self, data: DataDict) -> DataDict:
        if self.mode == "full":
            return data
        if self.mode == "no_wrist":
            self._remove_image(data, "left_wrist_0_rgb")
            self._remove_image(data, "right_wrist_0_rgb")
            return data
        if self.mode == "no_state":
            if "state" in data:
                data[ACTION_REFERENCE_STATE] = np.array(data["state"], copy=True)
                data["state"] = np.zeros_like(data["state"])
            if "state_history" in data:
                data["state_history"] = np.zeros_like(data["state_history"])
            return data
        if self.mode == "no_prompt":
            data["prompt"] = np.asarray(self.replacement_prompt)
            return data
        if self.mode == "no_image":
            self._remove_image(data, "base_0_rgb")
            self._remove_image(data, "left_wrist_0_rgb")
            self._remove_image(data, "right_wrist_0_rgb")
            return data
        if self.mode == "no_image_no_state":
            self._remove_image(data, "base_0_rgb")
            self._remove_image(data, "left_wrist_0_rgb")
            self._remove_image(data, "right_wrist_0_rgb")
            if "state" in data:
                data[ACTION_REFERENCE_STATE] = np.array(data["state"], copy=True)
                data["state"] = np.zeros_like(data["state"])
            if "state_history" in data:
                data["state_history"] = np.zeros_like(data["state_history"])
            return data
        raise ValueError(f"Unsupported condition ablation mode: {self.mode}")

    @staticmethod
    def _remove_image(data: DataDict, key: str) -> None:
        if "image" in data and key in data["image"]:
            data["image"][key] = np.zeros_like(data["image"][key])
        if "image_mask" in data and key in data["image_mask"]:
            data["image_mask"][key] = np.asarray(False)


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


def _assert_quantile_stats_for_keys(norm_stats: at.PyTree[NormStats], keys: frozenset[str]) -> None:
    flat_stats = flatten_dict(norm_stats)
    for k in keys:
        if k not in flat_stats:
            raise ValueError(f"quantile norm key {k!r} not found in norm stats.")
        v = flat_stats[k]
        if v.q01 is None or v.q99 is None:
            raise ValueError(f"quantile norm key {k!r} is missing q01 or q99.")
