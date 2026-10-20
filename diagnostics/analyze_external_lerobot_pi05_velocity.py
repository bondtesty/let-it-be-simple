#!/usr/bin/env python3
"""Probe a public LeRobot PI0.5 checkpoint at fixed flow times.

The checkpoint's native path is

    x_t = t * noise + (1 - t) * action
    velocity = noise - action

so ``t`` is directly the reported noise fraction.  The probe loads the saved
LeRobot preprocessor, including action/state normalization and state-token
construction, and reuses the model's inference-time cached condition prefix.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

from analyze_external_starvla_velocity import aggregate
from analyze_external_starvla_velocity import episode_path
from analyze_external_starvla_velocity import git_revision
from analyze_external_starvla_velocity import metric_rows
from analyze_external_starvla_velocity import select_anchors
import numpy as np
import pandas as pd
from PIL import Image
from safetensors.torch import load_file
import torch

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
EXPECTED_CHECKPOINT_BYTES = 7_473_096_344
EXPECTED_CHECKPOINT_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_inline_image(cell: dict) -> np.ndarray:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise TypeError(f"Expected inline LeRobot image bytes, got {type(cell)}")
    return np.asarray(Image.open(BytesIO(cell["bytes"])).convert("RGB"), dtype=np.uint8).copy()


def load_anchor(dataset_root: Path, anchor: dict, horizon: int) -> dict[str, np.ndarray]:
    table = pd.read_parquet(
        episode_path(dataset_root, anchor["episode_index"]),
        columns=["image", "wrist_image", "state", "actions", "frame_index", "task_index"],
    )
    frame_index = anchor["frame_index"]
    row = table.iloc[frame_index]
    if int(row["frame_index"]) != frame_index or int(row["task_index"]) != anchor["task_index"]:
        raise RuntimeError("LIBERO parquet/metadata anchor mismatch")
    actions = np.stack(table.iloc[frame_index : frame_index + horizon]["actions"].to_numpy()).astype(np.float32)
    state = np.asarray(row["state"], dtype=np.float32)
    if actions.shape != (horizon, 7) or state.shape != (8,):
        raise RuntimeError(f"Expected H{horizon}x7 actions and D8 state, got {actions.shape} and {state.shape}")
    return {
        "observation.images.image": decode_inline_image(row["image"]),
        "observation.images.image2": decode_inline_image(row["wrist_image"]),
        "observation.state": state,
        "action": actions,
    }


def normalization_audit(checkpoint_dir: Path, dataset_root: Path) -> dict:
    tensors = load_file(str(checkpoint_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    local = json.loads((dataset_root / "meta" / "stats.json").read_text())
    dataset_frames = int(json.loads((dataset_root / "meta" / "info.json").read_text())["total_frames"])
    checkpoint_counts = {key: int(tensors[f"{key}.count"].item()) for key in ("action", "observation.state")}
    if any(count != dataset_frames for count in checkpoint_counts.values()):
        raise RuntimeError(f"Checkpoint/local frame-count mismatch: {checkpoint_counts} versus {dataset_frames}")
    pairs = {"action": "actions", "observation.state": "state"}
    fields = ("mean", "std", "min", "max")
    comparisons = {}
    for checkpoint_key, dataset_key in pairs.items():
        comparisons[checkpoint_key] = {}
        for field in fields:
            reference = np.asarray(local[dataset_key][field], dtype=np.float64)
            checkpoint = tensors[f"{checkpoint_key}.{field}"].double().numpy()
            comparisons[checkpoint_key][field] = float(np.max(np.abs(checkpoint - reference)))
    maximum = max(value for group in comparisons.values() for value in group.values())
    if maximum > 1e-5:
        raise RuntimeError(f"Checkpoint/local normalization mismatch: max abs diff {maximum:.3e}")
    return {
        "checkpoint_stats_file": str(checkpoint_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"),
        "dataset_stats_file": str(dataset_root / "meta" / "stats.json"),
        "max_abs_differences": comparisons,
        "maximum_abs_difference": maximum,
        "passed_threshold": 1e-5,
        "dataset_frames": dataset_frames,
        "checkpoint_counts": checkpoint_counts,
    }


def load_policy(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(args.lerobot_root / "src"))
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    config.device = str(device)
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = PI05Policy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor


def encode_prefix(policy, batch: dict[str, torch.Tensor]):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    images, image_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    core = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = core.embed_prefix(images, image_masks, tokens, token_masks)
    prefix_att_2d_masks = core._prepare_attention_masks_4d(  # noqa: SLF001
        make_attention_masks(prefix_pad_masks, prefix_att_masks)
    )
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = core.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values


def make_attention_masks(pad_masks: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
    """Import lazily after the requested LeRobot source root is on sys.path."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    return make_att_2d_masks(pad_masks, attention_masks)


def aggregate_pi05(records: list[dict], noise_levels: list[float]) -> list[dict]:
    """Aggregate metrics while preserving PI0.5's native ``t=noise_level`` convention."""
    rows = aggregate(records, noise_levels)
    for row in rows:
        row["model_time"] = row["noise_level"]
    return rows


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.samples_per_task <= 0 or args.noise_samples <= 0:
        raise ValueError("sample counts must be positive")
    if any(level <= 0 or level > 1 for level in args.noise_levels):
        raise ValueError("noise levels must lie in (0, 1]")
    checkpoint_file = args.checkpoint_dir / "model.safetensors"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    checkpoint_bytes = checkpoint_file.stat().st_size
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError(f"Checkpoint size mismatch: {checkpoint_bytes} != {EXPECTED_CHECKPOINT_BYTES}")
    checkpoint_sha256 = file_sha256(checkpoint_file)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}")

    checkpoint_config = json.loads((args.checkpoint_dir / "config.json").read_text())
    horizon = int(checkpoint_config["chunk_size"])
    if horizon != 50 or int(checkpoint_config["max_action_dim"]) != 32:
        raise RuntimeError(f"Unexpected PI0.5 shape contract: H{horizon}, {checkpoint_config['max_action_dim']}D")
    audit = normalization_audit(args.checkpoint_dir, args.dataset_root)
    selected = select_anchors(
        args.dataset_root,
        dataset_kind="libero",
        horizon=horizon,
        max_samples=args.max_samples,
        samples_per_task=args.samples_per_task,
        seed=args.selection_seed,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    policy, preprocessor = load_policy(args, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    from lerobot.policies.utils import prepare_observation_for_inference

    records = []
    condition_seconds = 0.0
    head_seconds = 0.0
    for sample_index, anchor in enumerate(selected):
        raw_frame = load_anchor(args.dataset_root, anchor, horizon)
        frame = prepare_observation_for_inference(raw_frame, device, task=anchor["task"])
        batch = preprocessor(frame)
        actions = policy.prepare_action(batch)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prefix_pad_masks, past_key_values = encode_prefix(policy, batch)
        synchronize(device)
        condition_seconds += time.perf_counter() - started

        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(
                args.noise_seed + sample_index * args.noise_samples + noise_index
            )
            noise = torch.randn(actions.shape, generator=generator, device=device, dtype=actions.dtype)
            target = noise - actions
            for noise_level in args.noise_levels:
                timestep = torch.full((1,), noise_level, device=device, dtype=torch.float32)
                x_t = noise_level * noise + (1.0 - noise_level) * actions
                synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction = policy.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        x_t=x_t,
                        timestep=timestep,
                    )
                synchronize(device)
                head_seconds += time.perf_counter() - started
                records.append(
                    {
                        **anchor,
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "model_time": noise_level,
                        **metric_rows(prediction[..., :7], target[..., :7]),
                    }
                )

        del batch, actions, prefix_pad_masks, past_key_values, noise, target, x_t, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema_version": "openpi.external_lerobot_pi05_velocity.v1",
        "checkpoint": {
            "directory": str(args.checkpoint_dir),
            "file": str(checkpoint_file),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "framework": "LeRobot PI0.5",
            "source_revision": git_revision(args.lerobot_root),
        },
        "dataset": {
            "kind": "libero",
            "root": str(args.dataset_root),
            "episodes": 1693,
            "frames": 273465,
            "same_source_evidence": (
                "Checkpoint normalizer count is 273465 and its action/state mean, std, min, and max match the "
                "local physical-intelligence/libero metadata within the fail-closed threshold."
            ),
            "normalization_audit": audit,
        },
        "protocol": {
            "sample_count": len(selected),
            "samples_per_task": args.samples_per_task,
            "selection": (
                f"task-balanced deterministic nonboundary valid-H{horizon} anchors; checkpoint training data, "
                "not held out"
            ),
            "selection_seed": args.selection_seed,
            "noise_seed": args.noise_seed,
            "noise_samples_per_anchor": args.noise_samples,
            "noise_levels": args.noise_levels,
            "fixed_noise_per_anchor_across_times": True,
            "horizon": horizon,
            "action_dim": 7,
            "condition": "two RGB cameras plus task text with normalized D8 state discretized into text tokens",
            "normalization": "saved LeRobot mean/std preprocessor for D8 state and H50x7 actions",
            "time_convention": "PI0.5 t=0 data, t=1 pure Gaussian noise; reported noise_level=t",
            "field_path": "official cached-prefix inference denoise_step at explicit x_t and timestep",
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "condition_seconds": condition_seconds,
            "action_head_seconds": head_seconds,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        },
        "aggregate": aggregate_pi05(records, args.noise_levels),
        "selected_anchors": selected,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
