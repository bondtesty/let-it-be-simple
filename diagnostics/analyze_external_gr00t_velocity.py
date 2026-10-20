#!/usr/bin/env python3
"""Probe a GR00T checkpoint's learned velocity field at fixed noise levels.

The implementation follows GR00T's native linear path, where model time zero
is noise and model time one is data. Results are reported in the paper's
opposite noise-level convention; prediction and target are both negated, which
leaves MSE and cosine error unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import av
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.data.types import VLAStepData
import gr00t.model  # noqa: F401  # Registers the custom HF model and processor.
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoProcessor

STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
BEHAVIOR_STATE_KEYS = (
    "robot_pos",
    "robot_ori_cos",
    "robot_ori_sin",
    "robot_2d_ori",
    "robot_2d_ori_cos",
    "robot_2d_ori_sin",
    "robot_lin_vel",
    "robot_ang_vel",
    "arm_left_qpos",
    "arm_left_qpos_sin",
    "arm_left_qpos_cos",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "arm_right_qpos",
    "arm_right_qpos_sin",
    "arm_right_qpos_cos",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "trunk_qpos",
)
BEHAVIOR_ACTION_KEYS = ("base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper")
DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CHECKPOINT_REVISION = "e76ae6880ee960f09b40cc63058b6e2b0e3fbef0"
SOURCE_REVISION = "e29d8fc50b0e4745120ae3fb72447986fe638aa6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("bridge", "behavior"), default="bridge")
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument(
        "--anchors-per-task",
        type=int,
        default=1,
        help="Number of evenly spaced valid anchors from the held-out episode; zero uses all anchors.",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--store-endpoint-predictions", action="store_true")
    parser.add_argument(
        "--condition-ablation",
        choices=("none", "state_mask_token", "mean_image", "text_attention_mask"),
        default="none",
        help=(
            "Inference-only condition ablation. state_mask_token uses the checkpoint's training-time state-dropout "
            "token; mean_image zeros already-normalized pixels; text_attention_mask masks non-image VLM tokens."
        ),
    )
    parser.add_argument("--noise-levels", type=float, nargs="+")
    parser.add_argument(
        "--dense-noise-points",
        type=int,
        default=0,
        help="Use this many linearly spaced levels in [0.001, 1]; mutually exclusive with --noise-levels.",
    )
    return parser.parse_args()


def rgb_frames(video_path: Path, count: int) -> list[np.ndarray]:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise RuntimeError(f"Expected {count} frames from {video_path}, decoded {len(frames)}")
    return frames


def split_columns(values: np.ndarray, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {key: values[..., index : index + 1] for index, key in enumerate(keys)}


def slice_columns(values: np.ndarray, keys: tuple[str, ...], metadata: dict) -> dict[str, np.ndarray]:
    return {key: values[..., metadata[key]["start"] : metadata[key]["end"]] for key in keys}


def load_bridge_steps(
    dataset_root: Path,
    task_entry: dict,
    anchors_per_task: int,
) -> list[tuple[VLAStepData, int, int]]:
    episode = task_entry["episodes"][0]
    table = pd.read_parquet(dataset_root / episode["parquet"])
    valid_length = len(table) - 8 + 1
    if valid_length <= 0:
        raise ValueError(f"Episode {episode['episode_index']} is shorter than H8")
    if anchors_per_task == 0 or anchors_per_task >= valid_length:
        anchor_indices = np.arange(valid_length)
    else:
        anchor_indices = np.unique(np.linspace(0, valid_length - 1, anchors_per_task).round().astype(int))
    raw_states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    raw_actions = np.stack(table["action"].to_numpy()).astype(np.float32)
    frames = rgb_frames(dataset_root / episode["video"], int(anchor_indices[-1]) + 1)
    steps = []
    for frame_index in anchor_indices:
        raw_state = raw_states[frame_index : frame_index + 1]
        raw_action = raw_actions[frame_index : frame_index + 8]
        step = VLAStepData(
            images={"image_0": [frames[frame_index]]},
            states=split_columns(raw_state, STATE_KEYS),
            actions=split_columns(raw_action, ACTION_KEYS),
            text=task_entry["task"],
            embodiment=EmbodimentTag.OXE_WIDOWX,
        )
        steps.append((step, int(episode["episode_index"]), int(frame_index)))
    return steps


def load_behavior_steps(
    dataset_root: Path,
    task_entry: dict,
    anchors_per_task: int,
) -> list[tuple[VLAStepData, int, int]]:
    episode = task_entry["episodes"][0]
    table = pd.read_parquet(dataset_root / episode["parquet"])
    horizon = 32
    valid_length = len(table) - horizon + 1
    if valid_length <= 0:
        raise ValueError(f"Episode {episode['episode_index']} is shorter than H{horizon}")
    if anchors_per_task == 0 or anchors_per_task >= valid_length:
        anchor_indices = np.arange(valid_length)
    else:
        anchor_indices = np.unique(np.linspace(0, valid_length - 1, anchors_per_task).round().astype(int))

    modality = json.loads((dataset_root / task_entry["modality"]).read_text())
    task_catalog = {
        row["task_index"]: row["task"]
        for line in (dataset_root / task_entry["dataset_key"] / "meta/tasks.jsonl").read_text().splitlines()
        if (row := json.loads(line))["task"]
    }
    raw_states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    raw_actions = np.stack(table["action"].to_numpy()).astype(np.float32)
    frames_by_key = {
        key: rgb_frames(dataset_root / path, int(anchor_indices[-1]) + 1) for key, path in episode["videos"].items()
    }
    steps = []
    for frame_index in anchor_indices:
        raw_state = raw_states[frame_index : frame_index + 1]
        raw_action = raw_actions[frame_index : frame_index + horizon]
        annotation_index = int(table.iloc[frame_index]["annotation.human.coarse_action"])
        text = task_catalog.get(annotation_index, task_entry["task"])
        if not text:
            raise RuntimeError(f"Could not decode annotation index {annotation_index} for {task_entry['dataset_key']}")
        step = VLAStepData(
            images={key: [frames[frame_index]] for key, frames in frames_by_key.items()},
            states=slice_columns(raw_state, BEHAVIOR_STATE_KEYS, modality["state"]),
            actions=slice_columns(raw_action, BEHAVIOR_ACTION_KEYS, modality["action"]),
            text=text,
            embodiment=EmbodimentTag.BEHAVIOR_R1_PRO,
        )
        steps.append((step, int(episode["episode_index"]), int(frame_index)))
    return steps


def load_steps(
    dataset_root: Path,
    task_entry: dict,
    anchors_per_task: int,
    profile: str,
) -> list[tuple[VLAStepData, int, int]]:
    if profile == "behavior":
        return load_behavior_steps(dataset_root, task_entry, anchors_per_task)
    return load_bridge_steps(dataset_root, task_entry, anchors_per_task)


def task_key(task_entry: dict) -> str:
    return task_entry.get("dataset_key") or task_entry["task"]


def fixed_velocity_prediction(
    head: torch.nn.Module,
    backbone_output,
    action_input,
    noise: torch.Tensor,
    model_time: float,
    condition_ablation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official action-head computation with fixed noise and time."""
    # Do not call process_backbone_output here because it mutates its input and
    # this probe deliberately reuses one backbone output across fixed times.
    vl_embeds = head.vlln(backbone_output.backbone_features)
    embodiment_id = action_input.embodiment_id
    state_features = head.state_encoder(action_input.state, embodiment_id)
    if condition_ablation == "state_mask_token":
        if head.mask_token is None:
            raise RuntimeError("The checkpoint does not define a learned state mask token")
        state_features = head.mask_token.expand(state_features.shape[0], -1, -1)
    actions = action_input.action
    time_tensor = torch.full(
        (actions.shape[0], 1, 1),
        model_time,
        device=actions.device,
        dtype=actions.dtype,
    )
    noisy_trajectory = (1.0 - time_tensor) * noise + time_tensor * actions
    target_velocity = actions - noise
    time_bucket = (time_tensor[:, 0, 0] * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy_trajectory, time_bucket, embodiment_id)
    if head.config.add_pos_embed:
        position_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
        action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
    state_action_features = torch.cat((state_features, action_features), dim=1)
    if head.config.use_alternate_vl_dit:
        model_output, _ = head.model(
            hidden_states=state_action_features,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            timestep=time_bucket,
            return_all_hidden_states=True,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output, _ = head.model(
            hidden_states=state_action_features,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            timestep=time_bucket,
            return_all_hidden_states=True,
        )
    prediction = head.action_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
    return prediction, target_velocity


def masked_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    prediction_values = prediction[mask]
    target_values = target[mask]
    mse = torch.mean((prediction_values - target_values) ** 2)
    denominator = torch.linalg.vector_norm(prediction_values) * torch.linalg.vector_norm(target_values)
    cosine = torch.sum(prediction_values * target_values) / denominator.clamp_min(1e-12)
    return {"mse": float(mse.float().cpu()), "cosine_error": float((1.0 - cosine).float().cpu())}


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ablate_backbone_input(backbone_input, model: torch.nn.Module, condition_ablation: str) -> None:
    """Apply a documented inference-only ablation before the VLM backbone."""
    if condition_ablation == "mean_image":
        # Pixel values have already passed the checkpoint image normalization;
        # zero therefore represents the normalization center, not raw black.
        pixel_values = backbone_input["pixel_values"]
        if isinstance(pixel_values, list):
            backbone_input["pixel_values"] = [torch.zeros_like(value) for value in pixel_values]
        else:
            backbone_input["pixel_values"] = torch.zeros_like(pixel_values)
    elif condition_ablation == "text_attention_mask":
        image_tokens = backbone_input["input_ids"] == model.backbone.model.config.image_token_index
        backbone_input["attention_mask"] = backbone_input["attention_mask"].bool() & image_tokens


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.bootstrap_samples <= 0 or args.noise_samples <= 0 or args.anchors_per_task < 0:
        raise ValueError("max-samples, bootstrap-samples, and noise-samples must be positive")
    if args.noise_levels is not None and args.dense_noise_points:
        raise ValueError("--noise-levels and --dense-noise-points are mutually exclusive")
    if args.dense_noise_points:
        if args.dense_noise_points < 2:
            raise ValueError("dense-noise-points must be zero or at least two")
        noise_levels = np.linspace(0.001, 1.0, args.dense_noise_points).tolist()
    else:
        noise_levels = list(args.noise_levels or DEFAULT_NOISE_LEVELS)
    if any(level <= 0 or level > 1 for level in noise_levels):
        raise ValueError("noise levels must be in (0, 1]")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # Passing a torch.device here is rejected by some CUDA/Python builds;
        # after set_device, the no-argument form is unambiguous.
        torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((args.dataset_root / "probe_manifest.json").read_text())
    task_entries = manifest["tasks"][: args.max_samples]

    processor = AutoProcessor.from_pretrained(
        args.checkpoint_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.eval()
    config = AutoConfig.from_pretrained(args.checkpoint_dir, trust_remote_code=True, local_files_only=True)
    config.use_flash_attention = False
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.checkpoint_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Match Gr00tPolicy: the constructor keeps trainable backbone layers in
    # fp32 for training, but inference explicitly casts the whole model to bf16.
    model.eval().to(device=device, dtype=torch.bfloat16)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    attention_implementations = {
        "eagle": model.backbone.model.config._attn_implementation,  # noqa: SLF001
        "text": model.backbone.model.config.text_config._attn_implementation,  # noqa: SLF001
        "vision": model.backbone.model.config.vision_config._attn_implementation,  # noqa: SLF001
    }
    if any(implementation != "sdpa" for implementation in attention_implementations.values()):
        raise RuntimeError(f"Expected an all-SDPA compatibility path, got {attention_implementations}")

    records = []
    backbone_seconds = 0.0
    action_head_seconds = 0.0
    anchors_by_task = {}
    sample_index = 0
    for task_entry in task_entries:
        task_steps = load_steps(args.dataset_root, task_entry, args.anchors_per_task, args.profile)
        entry_key = task_key(task_entry)
        anchors_by_task[entry_key] = len(task_steps)
        for step, episode_index, frame_index in task_steps:
            processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            collated = processor.collator([processed])["inputs"]
            backbone_input, action_input = model.prepare_input(collated)
            ablate_backbone_input(backbone_input, model, args.condition_ablation)
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                backbone_output = model.backbone(backbone_input)
            synchronize(device)
            backbone_seconds += time.perf_counter() - started

            continuous_mask = action_input.action_mask.bool().clone()
            if args.profile == "behavior":
                continuous_mask[..., 14] = False
                continuous_mask[..., 22] = False
            else:
                continuous_mask[..., 6] = False
            for noise_index in range(args.noise_samples):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + sample_index * args.noise_samples + noise_index
                )
                noise = torch.randn(
                    action_input.action.shape,
                    generator=generator,
                    device=device,
                    dtype=action_input.action.dtype,
                )
                for noise_level in noise_levels:
                    model_time = 1.0 - noise_level
                    synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        prediction, target = fixed_velocity_prediction(
                            model.action_head,
                            backbone_output,
                            action_input,
                            noise,
                            model_time,
                            args.condition_ablation,
                        )
                    synchronize(device)
                    action_head_seconds += time.perf_counter() - started
                    record = {
                        "task": entry_key,
                        "task_text": step.text,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "model_time": model_time,
                        "all_action": masked_metrics(prediction, target, action_input.action_mask.bool()),
                        "continuous_action": masked_metrics(prediction, target, continuous_mask),
                    }
                    if args.store_endpoint_predictions and noise_level == 1.0:
                        one_step_action = noise + prediction
                        horizon, action_dim = (32, 23) if args.profile == "behavior" else (8, 7)
                        record["one_step_action_prediction"] = (
                            one_step_action[0, :horizon, :action_dim].float().cpu().numpy().tolist()
                        )
                        record["target_action"] = (
                            action_input.action[0, :horizon, :action_dim].float().cpu().numpy().tolist()
                        )
                    records.append(record)
            sample_index += 1

    aggregate = []
    for level_index, noise_level in enumerate(noise_levels):
        level_records = [record for record in records if record["noise_level"] == noise_level]
        row = {"noise_level": noise_level, "model_time": 1.0 - noise_level}
        for group in ("all_action", "continuous_action"):
            row[group] = {}
            for metric in ("mse", "cosine_error"):
                task_values = []
                for task_entry in task_entries:
                    entry_key = task_key(task_entry)
                    per_noise = [record[group][metric] for record in level_records if record["task"] == entry_key]
                    expected_records = anchors_by_task[entry_key] * args.noise_samples
                    if len(per_noise) != expected_records:
                        raise RuntimeError(f"Expected {expected_records} records for {entry_key}, got {len(per_noise)}")
                    task_values.append(np.mean(per_noise))
                values = np.asarray(task_values, dtype=np.float64)
                row[group][metric] = {
                    "task_macro_mean": float(values.mean()),
                    "task_bootstrap_95ci": bootstrap_interval(
                        values,
                        args.bootstrap_samples,
                        args.seed + 100 * level_index + (0 if group == "all_action" else 1),
                    ),
                }
        aggregate.append(row)

    if args.profile == "behavior":
        condition = "three 256x256 RGB views, checkpoint-selected 82D R1Pro state, exact per-frame coarse-action text"
        raw_target = "H32 x 23 BEHAVIOR R1Pro action chunk at each selected training-distribution anchor"
        evaluated_mask = "first H32 x 23; continuous_action excludes left/right gripper dimensions 14 and 22"
    else:
        condition = "one image_0 frame, normalized 8D state, exact task text"
        raw_target = "H8 x 7 absolute Bridge action chunk at each selected held-out anchor"
        evaluated_mask = "first H8 x 7; continuous_action excludes gripper dimension 6"

    result = {
        "schema_version": "openpi.external_gr00t_velocity_probe.v1",
        "checkpoint": args.checkpoint_name or args.checkpoint_dir.name,
        "checkpoint_revision": manifest.get(
            "checkpoint_revision", CHECKPOINT_REVISION if args.profile == "bridge" else None
        ),
        "source_revision": SOURCE_REVISION,
        "dataset_repo": manifest["repo_id"],
        "dataset_revision": manifest.get("revision"),
        "protocol": {
            "condition": {
                "none": condition,
                "state_mask_token": (
                    "images and text unchanged; normalized state replaced by the checkpoint's learned state-dropout token"
                ),
                "mean_image": (
                    "normalized state and text unchanged; all normalized image pixels replaced by their zero center"
                ),
                "text_attention_mask": (
                    "images and normalized state unchanged; non-image VLM tokens masked before the backbone"
                ),
            }[args.condition_ablation],
            "condition_ablation": args.condition_ablation,
            "condition_ablation_caution": (
                "state_mask_token matches the checkpoint's training-time state dropout; image/text ablations are "
                "inference interventions and may be out of distribution"
            ),
            "profile": args.profile,
            "raw_target": raw_target,
            "model_tensor_shape": [50, 128],
            "evaluated_mask": evaluated_mask,
            "normalization": "actual checkpoint processor; verified separately against manual transform",
            "fixed_noise": (
                f"{args.noise_samples} seeded Gaussian samples per anchor; each is shared over all noise levels"
            ),
            "time_convention": (
                "reported noise_level t uses x_t=(1-t)*action+t*noise and target noise-action; "
                "GR00T model_time=1-t and target action-noise, so both prediction and target are negated"
            ),
            "attention_kernel": "PyTorch SDPA compatibility override; weights and processor unchanged",
            "samples": len(task_entries),
            "tasks": len(task_entries),
            "anchors_per_task_request": args.anchors_per_task,
            "anchors_by_task": anchors_by_task,
            "total_anchors": sum(anchors_by_task.values()),
            "noise_samples_per_task": args.noise_samples,
            "noise_levels": noise_levels,
            "stores_endpoint_predictions": args.store_endpoint_predictions,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "official_pinned_torch": "2.7.1",
            "runtime_override": "Torch 2.10/cu128 for RTX 5090 sm_120 support",
            "attention_implementations": attention_implementations,
            "model_load_seconds": model_load_seconds,
            "backbone_seconds_total": backbone_seconds,
            "action_head_seconds_total": action_head_seconds,
            "peak_cuda_memory_gib": (torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None),
        },
        "aggregate": aggregate,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol": result["protocol"], "runtime": result["runtime"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
