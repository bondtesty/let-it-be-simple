import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SimplerEnvInputs(transforms.DataTransformFn):
    """Transform for SimplerEnv OXE datasets (Bridge, Fractal/RT-1).

    Supports multiple camera views via image_keys. Each image will be passed
    to the model without zero-padding.
    """

    model_type: _model.ModelType
    # Maps model image key -> data dict key (after repack).
    image_keys: dict[str, str] = dataclasses.field(default_factory=dict)

    def __call__(self, data: dict) -> dict:
        images = {}
        image_mask = {}
        for target_key, source_key in self.image_keys.items():
            # After repack, data uses target_key; during inference env also uses target_key.
            img = _parse_image(data[target_key])
            images[target_key] = img
            image_mask[target_key] = np.True_

        inputs = {
            "state": data.get("observation/state", np.zeros(7, dtype=np.float32)),
            "image": images,
            "image_mask": image_mask,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SimplerEnvOutputs(transforms.DataTransformFn):
    """Output transform for SimplerEnv (7-DoF action)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
