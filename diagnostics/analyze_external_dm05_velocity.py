#!/usr/bin/env python3
"""Probe the OpenDM DM0.5 velocity field on an exact SO-101 dataset subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from opendm.constants.robot import ROBOT_STATE_DESCS
from opendm.constants.robot import ActionMode
from opendm.constants.robot import RobotType
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.transforms import BuildAction
from opendm.data.transforms import ChatTokenization
from opendm.data.transforms import LoadImages
from opendm.data.transforms import Normalize
from opendm.data.transforms import PadAction
from opendm.data.transforms import Pipeline
from opendm.data.transforms import PixelTransform
from opendm.model.dm05.dm05_arch import DM05Config
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration
from opendm.model.dm05.dm05_utils import HISTORY_PAD_TOKEN_ID
from opendm.model.dm05.dm05_utils import make_suffix_attn_mask
from peft import PeftModel
import torch
from transformers import AutoProcessor

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
BASE_CHECKPOINT_REVISION = "a7bf3f465de915a44ed9a4ac6c403ab3466f2735"
ADAPTER_CHECKPOINT_REVISION = "8617f0c177271a19aef1fd92cfec51c4b45c866c"
DATASET_REVISION = "86e706e38bb67bcf8f866cc45f5ac8a598547438"
OPENDM_SOURCE_REVISION = "e41e501bb82e9c3cb8138c0fb4687faa5f98c690"
BASE_CHECKPOINT_SHA256 = "b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd"
ADAPTER_CHECKPOINT_SHA256 = "e0af75525aa2026a9ed723c4bd618b98f5aa5ca67e228f819e2a301e7b42b052"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def eligible_episode_files(dataset_root: Path) -> list[Path]:
    eligible = []
    for path in sorted((dataset_root / "jsonl").glob("*.jsonl")):
        first_row = json.loads(path.open().readline())
        image_paths = [dataset_root / "videos" / first_row[key]["url"] for key in ("images_1", "images_2")]
        if all(image_path.exists() for image_path in image_paths):
            eligible.append(path)
    return eligible


def build_transform(processor, adapter_dir: Path, dataset_root: Path, horizon: int) -> Pipeline:
    return Pipeline(
        [
            BuildAction(horizon, action_mode=ActionMode.RELATIVE),
            LoadImages(image_keys=["images_1", "images_2"], image_dir=str(dataset_root / "videos")),
            PixelTransform(transform_pipeline=NoAugmentationPipeline()),
            Normalize(
                norm_stats_path=str(adapter_dir / "norm_stats.json"),
                norm_keys=["state", "action"],
                use_quantiles=True,
            ),
            ChatTokenization(
                processor=processor,
                n_bins=256,
                max_length=1024,
                image_prompts=["Head", "Left wrist"],
                add_state=True,
            ),
            PadAction(32),
        ]
    )


def raw_sample(path: Path, horizon: int) -> tuple[dict, np.ndarray]:
    raw_lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in raw_lines]
    if len(rows) < 2:
        raise ValueError(f"Episode {path} has fewer than two frames")
    anchor = max(0, (len(rows) - horizon) // 2)
    data = copy.deepcopy(rows[anchor])
    data["raw_lines"] = raw_lines
    data["meta_data"] = {
        "frame_index": anchor,
        "robot_type": RobotType.SO101,
        "control_mode": "joint position",
        "state_desc": ROBOT_STATE_DESCS[RobotType.SO101],
    }
    physical = BuildAction(horizon, action_mode=ActionMode.RELATIVE)(copy.deepcopy(data))["action"]
    return data, physical


def move_inputs(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | None]:
    result = {}
    for key, value in inputs.items():
        if value is None:
            result[key] = None
        elif torch.is_tensor(value):
            tensor = value
            if key in ("pixel_values", "action"):
                tensor = tensor.to(dtype=dtype)
            result[key] = tensor.to(device)
    return result


def compute_prefix(model, inputs: dict[str, torch.Tensor | None]):
    return model._compute_prefix_cache(  # noqa: SLF001
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        token_type_ids=inputs["token_type_ids"],
        history_pixel_values=inputs.get("history_pixel_values"),
        history_mask=inputs.get("history_mask"),
    )


def fixed_velocity_prediction(
    model,
    inputs: dict[str, torch.Tensor | None],
    kv_cache,
    prefix_len: int,
    noise: torch.Tensor,
    time_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    action = inputs["action"]
    action_mask = inputs["action_mask"]
    time_tensor = torch.full((1,), time_value, device=action.device, dtype=action.dtype)
    x_t = time_tensor[:, None, None] * noise + (1.0 - time_tensor[:, None, None]) * action
    x_t = x_t * action_mask
    target = noise - action

    suffix_embeds = model.model.action_in_proj(x_t)
    adarms_cond = model._build_adarms_cond(time_tensor, suffix_embeds.dtype)  # noqa: SLF001
    suffix_len = suffix_embeds.shape[1]
    suffix_attn_mask = make_suffix_attn_mask(
        input_ids=inputs["input_ids"],
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        batch_size=1,
        device=suffix_embeds.device,
        dtype=suffix_embeds.dtype,
        pad_token_id=model.model.vlm.model.language_model.padding_idx,
        invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,),
    )
    suffix_position_ids = model._build_suffix_position_ids(  # noqa: SLF001
        prefix_len,
        suffix_len,
        suffix_embeds.device,
        input_ids=inputs["input_ids"],
        pad_token_id=model.model.vlm.model.language_model.padding_idx,
        invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,),
    )
    suffix_out = model._suffix_forward(  # noqa: SLF001
        suffix_embeds=suffix_embeds,
        attention_mask=suffix_attn_mask,
        position_ids=suffix_position_ids,
        past_key_values=kv_cache,
        adarms_cond=adarms_cond,
    )
    prediction = model.model.action_out_proj(suffix_out)
    return prediction, target


def metrics(prediction: torch.Tensor, target: torch.Tensor, dimensions: slice) -> tuple[float, float]:
    prediction = prediction[..., dimensions].float()
    target = target[..., dimensions].float()
    mse = torch.mean(torch.square(prediction - target))
    cosine = torch.nn.functional.cosine_similarity(prediction.flatten(), target.flatten(), dim=0)
    return float(mse.cpu()), float((1.0 - cosine).cpu())


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    if args.max_episodes <= 0 or args.noise_samples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("max-episodes, noise-samples, and bootstrap-samples must be positive")
    if any(level <= 0 or level > 1 for level in args.noise_levels):
        raise ValueError("noise levels must be in (0, 1]")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    config = DM05Config.from_pretrained(args.base_checkpoint_dir)
    # The released config requests FlashAttention2 for the vision tower.  Use
    # the model's supported SDPA path so this diagnostic does not depend on an
    # out-of-tree flash-attn build.
    config.vlm_config._attn_implementation = "sdpa"  # noqa: SLF001
    config.vlm_config.text_config._attn_implementation = "sdpa"  # noqa: SLF001
    config.vlm_config.vision_config._attn_implementation = "sdpa"  # noqa: SLF001
    config.action_config._attn_implementation = "sdpa"  # noqa: SLF001
    base_model = DM05ForConditionalGeneration.from_pretrained(
        args.base_checkpoint_dir,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_checkpoint_dir).merge_and_unload()
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.set_attention_implementation(
        llm_attn_implementation="sdpa",
        vision_attn_implementation="sdpa",
        action_attn_implementation="sdpa",
        bf16=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_checkpoint_dir, trust_remote_code=True)
    transform = build_transform(processor, args.adapter_checkpoint_dir, args.dataset_root, config.chunk_size)
    episode_files = eligible_episode_files(args.dataset_root)[: args.max_episodes]
    if not episode_files:
        raise ValueError("No JSONL episodes have both referenced camera videos")
    model_load_seconds = time.perf_counter() - load_started

    records = []
    sample_metadata = []
    prefix_seconds = 0.0
    suffix_seconds = 0.0
    for episode_index, path in enumerate(episode_files):
        raw, physical_action = raw_sample(path, config.chunk_size)
        inputs = move_inputs(transform(raw), device, torch.bfloat16)
        expected_action_shape = (1, config.chunk_size, 32)
        if tuple(inputs["action"].shape) != expected_action_shape:
            raise ValueError(f"Expected normalized action shape {expected_action_shape}, got {inputs['action'].shape}")
        if physical_action.shape != (1, config.chunk_size, 6):
            raise ValueError(f"Expected physical action shape (1, {config.chunk_size}, 6), got {physical_action.shape}")
        sample_id = path.stem
        sample_metadata.append(
            {
                "sample_id": sample_id,
                "episode": path.name,
                "frame_index": raw["meta_data"]["frame_index"],
                "prompt": raw["prompt"],
                "physical_action_abs_mean": float(np.abs(physical_action).mean()),
            }
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            kv_cache, prefix_len = compute_prefix(model, inputs)
        torch.cuda.synchronize(device)
        prefix_seconds += time.perf_counter() - started

        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(args.seed + episode_index * 100_003 + noise_index)
            noise = torch.randn(
                inputs["action"].shape, generator=generator, device=device, dtype=inputs["action"].dtype
            )
            for noise_level in args.noise_levels:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction, target = fixed_velocity_prediction(
                        model, inputs, kv_cache, prefix_len, noise, noise_level
                    )
                torch.cuda.synchronize(device)
                suffix_seconds += time.perf_counter() - started
                all_mse, all_cosine = metrics(prediction, target, slice(0, 6))
                continuous_mse, continuous_cosine = metrics(prediction, target, slice(0, 5))
                records.append(
                    {
                        "sample_id": sample_id,
                        "episode": path.name,
                        "frame_index": raw["meta_data"]["frame_index"],
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "all_action": {"mse": all_mse, "cosine_error": all_cosine},
                        "continuous_action": {"mse": continuous_mse, "cosine_error": continuous_cosine},
                    }
                )

    aggregate = []
    bootstrap_rng = np.random.default_rng(args.seed)
    sample_ids = [sample["sample_id"] for sample in sample_metadata]
    for noise_level in args.noise_levels:
        level_records = [record for record in records if record["noise_level"] == noise_level]
        row = {"noise_level": noise_level}
        for group in ("all_action", "continuous_action"):
            row[group] = {}
            for metric_name in ("mse", "cosine_error"):
                values = np.asarray(
                    [
                        np.mean(
                            [record[group][metric_name] for record in level_records if record["sample_id"] == sample_id]
                        )
                        for sample_id in sample_ids
                    ]
                )
                row[group][metric_name] = {
                    "sample_macro_mean": float(values.mean()),
                    "sample_bootstrap_95ci": bootstrap_interval(values, args.bootstrap_samples, bootstrap_rng),
                }
        aggregate.append(row)

    result = {
        "schema_version": "openpi.external_dm05_velocity_probe.v1",
        "source_revision": OPENDM_SOURCE_REVISION,
        "base_checkpoint": str(args.base_checkpoint_dir),
        "base_checkpoint_revision": BASE_CHECKPOINT_REVISION,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "adapter_checkpoint": str(args.adapter_checkpoint_dir),
        "adapter_checkpoint_revision": ADAPTER_CHECKPOINT_REVISION,
        "adapter_checkpoint_sha256": ADAPTER_CHECKPOINT_SHA256,
        "dataset_repo": "Dexmal/so101_pick_cube",
        "dataset_revision": DATASET_REVISION,
        "protocol": {
            "episodes": len(episode_files),
            "selection": "lexicographically first eligible JSONLs; one centered anchor per episode",
            "condition": "two exact RGB video frames, quantile-normalized 6D state tokens, exact task prompt",
            "target": "H50 x 6 current-row actions; first five joints relative to anchor state, gripper absolute",
            "normalization": "adapter norm_stats q01/q99 clipping and [-1,1] mapping, then exact 32D zero padding",
            "normalization_gate": "official OpenDM BuildAction/Normalize/ChatTokenization/PadAction transform order",
            "time_convention": "noise_level t uses x_t=t*noise+(1-t)*action and target noise-action",
            "training_time_schedule": "Beta(1.5,1.0), then t=0.999*sample+0.001",
            "noise_levels": args.noise_levels,
            "noise_samples_per_episode": args.noise_samples,
            "seed": args.seed,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "attention_implementation": {"llm": "sdpa", "vision": "sdpa", "action": "sdpa"},
            "model_load_seconds": model_load_seconds,
            "prefix_seconds_total": prefix_seconds,
            "suffix_seconds_total": suffix_seconds,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        },
        "aggregate": aggregate,
        "sample_metadata": sample_metadata,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol": result["protocol"], "runtime": result["runtime"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
