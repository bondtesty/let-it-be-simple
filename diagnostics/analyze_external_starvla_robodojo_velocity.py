#!/usr/bin/env python3
"""Probe StarVLA RoboDojo flow checkpoints on a selective official-data subset.

Supported released frameworks are QwenGR00T and QwenPI_v3. Both use the native
StarVLA path ``x_t=(1-t)*noise+t*action`` with target ``action-noise``. The
reported axis is ``noise_level=1-t``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import av
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F  # noqa: N812

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
GRIPPER_DIMS = (6, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max-samples", type=int, default=35)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--checkpoint-file", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_file is not None:
        return args.checkpoint_file
    candidates = sorted((args.checkpoint_dir / "checkpoints").glob("*.pt"))
    candidates += sorted((args.checkpoint_dir / "checkpoints").glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint under {args.checkpoint_dir / 'checkpoints'}")
    return candidates[-1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode_video_frame(path: Path, frame_index: int) -> Image.Image:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24")).resize((224, 224))
    raise RuntimeError(f"Could not decode frame {frame_index} from {path}")


def select_anchors(dataset_root: Path, *, horizon: int, max_samples: int, seed: int) -> list[dict]:
    episodes = {int(row["episode_index"]): row for row in read_jsonl(dataset_root / "meta" / "episodes.jsonl")}
    tasks = {int(row["task_index"]): row["task"] for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")}
    available = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    rng = np.random.default_rng(seed)
    selected = []
    seen_tasks = set()
    for parquet in available:
        episode_index = int(parquet.stem.removeprefix("episode_"))
        episode = episodes[episode_index]
        task = episode["tasks"][0]
        task_index = next(index for index, description in tasks.items() if description == task)
        if task_index in seen_tasks or int(episode["length"]) < horizon:
            continue
        valid_count = int(episode["length"]) - horizon + 1
        low = min(10, valid_count - 1)
        high = max(low + 1, valid_count - 10)
        frame_index = int(rng.integers(low, high)) if high > low else low
        selected.append(
            {
                "task": task,
                "task_index": task_index,
                "episode_index": episode_index,
                "episode_length": int(episode["length"]),
                "frame_index": frame_index,
                "parquet": str(parquet.relative_to(dataset_root)),
            }
        )
        seen_tasks.add(task_index)
        if len(selected) == max_samples:
            break
    return sorted(selected, key=lambda row: row["task_index"])


def normalize_q99(values: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    result = values.astype(np.float32, copy=True)
    result[..., mask] = 2.0 * (values[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return result


def load_anchor(
    dataset_root: Path,
    anchor: dict,
    *,
    horizon: int,
    state_stats: dict,
    action_stats: dict,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    table = pd.read_parquet(
        dataset_root / anchor["parquet"],
        columns=["observation.state", "action", "frame_index", "task_index"],
    )
    index = anchor["frame_index"]
    if int(table.iloc[index]["frame_index"]) != index or int(table.iloc[index]["task_index"]) != anchor["task_index"]:
        raise RuntimeError("RoboDojo parquet metadata mismatch")
    raw_state = np.asarray(table.iloc[index]["observation.state"], dtype=np.float32)[None]
    raw_action = np.stack(table.iloc[index : index + horizon]["action"].to_numpy()).astype(np.float32)
    if raw_state.shape != (1, 14) or raw_action.shape != (horizon, 14):
        raise RuntimeError(f"Unexpected state/action shapes: {raw_state.shape}, {raw_action.shape}")

    chunk = anchor["episode_index"] // 1000
    images = [
        decode_video_frame(
            dataset_root / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{anchor['episode_index']:06d}.mp4",
            index,
        )
        for camera in CAMERA_KEYS
    ]
    return images, normalize_q99(raw_state, state_stats), normalize_q99(raw_action, action_stats)


def load_model(args: argparse.Namespace, checkpoint_file: Path, device: torch.device):
    sys.path.insert(0, str(args.starvla_root))
    from accelerate import init_empty_weights
    from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
    import starVLA.model.modules.action_model.GR00T_ActionHeader as groot_head_module
    import starVLA.model.modules.action_model.LayerwiseFM_ActionHeader as layerwise_head_module
    from transformers import AutoConfig
    from transformers import Qwen3VLForConditionalGeneration

    config_path = args.checkpoint_dir / "config.full.yaml"
    if not config_path.is_file():
        config_path = args.checkpoint_dir / "config.yaml"
    config = OmegaConf.load(config_path)
    framework = str(config.framework.name)
    if framework not in {"QwenGR00T", "QwenPI_v3"}:
        raise ValueError(f"Unsupported framework: {framework}")
    config.framework.qwenvl.base_vlm = args.base_vlm
    config.framework.qwenvl.attn_implementation = "sdpa"
    if "trainer" not in config:
        config.trainer = {}
    config.trainer.pretrained_checkpoint = None

    original_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained.__func__
    original_groot_beta = groot_head_module.Beta
    original_layerwise_beta = layerwise_head_module.Beta

    @classmethod
    def empty_from_pretrained(cls, model_id, *unused_args, **unused_kwargs):
        base_config = AutoConfig.from_pretrained(model_id)
        base_config._attn_implementation = "sdpa"  # noqa: SLF001
        return cls(base_config)

    def placeholder_beta(*unused_args, **unused_kwargs):
        return None

    Qwen3VLForConditionalGeneration.from_pretrained = empty_from_pretrained
    groot_head_module.Beta = placeholder_beta
    layerwise_head_module.Beta = placeholder_beta
    try:
        with init_empty_weights(include_buffers=False):
            model = Qwen_GR00T(config) if framework == "QwenGR00T" else Qwen_PI_v3(config)
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = classmethod(original_from_pretrained)
        groot_head_module.Beta = original_groot_beta
        layerwise_head_module.Beta = original_layerwise_beta
    beta_class = original_groot_beta if framework == "QwenGR00T" else original_layerwise_beta
    model.action_model.beta_dist = beta_class(
        config.framework.action_model.noise_beta_alpha,
        config.framework.action_model.noise_beta_beta,
    )

    if checkpoint_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_file), device="cpu")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    meta_tensors = [name for name, tensor in [*model.named_parameters(), *model.named_buffers()] if tensor.is_meta]
    if meta_tensors:
        raise RuntimeError(f"Checkpoint left meta tensors: {meta_tensors[:20]}")
    return model.to(device=device, dtype=torch.bfloat16).eval(), framework, config


def encode_condition(
    model,
    framework: str,
    images: list[Image.Image],
    task: str,
    normalized_state: np.ndarray,
):
    instructions = [task]
    if framework == "QwenPI_v3":
        instructions = model.add_discretized_state_to_instruction(instructions, [normalized_state])
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=[images], instructions=instructions)
    attention_mask = qwen_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if framework == "QwenGR00T":
            condition = output.hidden_states[-1]
        else:
            layers = list(output.hidden_states[-model.num_action_dit_layers :])
            condition = model._project_vl_hidden_for_action(layers)  # noqa: SLF001
    state = (
        torch.from_numpy(normalized_state).unsqueeze(0).to(device=output.hidden_states[-1].device, dtype=torch.bfloat16)
        if framework == "QwenGR00T"
        else None
    )
    return condition, attention_mask, state, instructions[0]


def fixed_velocity_prediction(
    head,
    framework: str,
    condition,
    attention_mask: torch.Tensor | None,
    state: torch.Tensor | None,
    actions: torch.Tensor,
    noise: torch.Tensor,
    model_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = actions.shape[0]
    t = torch.full((batch_size,), model_time, device=actions.device, dtype=actions.dtype)
    noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target = actions - noise
    bucket = (t * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy, bucket)
    if head.config.add_pos_embed:
        ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
        action_features = action_features + head.position_embedding(ids).unsqueeze(0)
    state_features = head.state_encoder(state) if state is not None else None
    future = head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
    suffix = (
        torch.cat((state_features, future, action_features), dim=1)
        if state_features is not None
        else torch.cat((future, action_features), dim=1)
    )
    if framework == "QwenGR00T":
        output = head.model(
            hidden_states=suffix,
            encoder_hidden_states=condition,
            encoder_attention_mask=attention_mask,
            timestep=bucket,
            return_all_hidden_states=False,
        )
    else:
        output = head.model(
            hidden_states=suffix,
            encoder_hidden_states=condition,
            encoder_attention_mask=attention_mask,
            timestep=bucket,
            return_pre_output=True,
        )
    return head.action_decoder(output)[:, -actions.shape[1] :], target


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, dict[str, float]]:
    joint_dims = [index for index in range(14) if index not in GRIPPER_DIMS]
    groups = {
        "all_action": (prediction, target),
        "joint_action": (prediction[..., joint_dims], target[..., joint_dims]),
        "gripper_action": (prediction[..., list(GRIPPER_DIMS)], target[..., list(GRIPPER_DIMS)]),
    }
    result = {}
    for name, (raw_prediction, raw_target) in groups.items():
        pred = raw_prediction.float()
        truth = raw_target.float()
        result[name] = {
            "mse": float((pred - truth).square().mean().cpu()),
            "cosine_error": float((1.0 - F.cosine_similarity(pred.flatten(), truth.flatten(), dim=0)).cpu()),
            "prediction_norm": float(pred.norm().cpu()),
            "target_norm": float(truth.norm().cpu()),
        }
    return result


def aggregate(records: list[dict], noise_levels: list[float]) -> list[dict]:
    rows = []
    for level in noise_levels:
        selected = [record for record in records if math.isclose(record["noise_level"], level)]
        row = {"noise_level": level, "model_time": 1.0 - level}
        for group in ("all_action", "joint_action", "gripper_action"):
            values = np.asarray([record[group]["mse"] for record in selected], dtype=np.float64)
            row[group] = {
                "mse_mean": float(values.mean()),
                "mse_se": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.noise_samples <= 0:
        raise ValueError("sample counts must be positive")
    checkpoint_file = resolve_checkpoint(args)
    stats = json.loads((args.checkpoint_dir / "dataset_statistics.json").read_text())["arx_x5"]
    if int(stats["num_trajectories"]) != 3500 or int(stats["num_transitions"]) != 1859602:
        raise RuntimeError("Unexpected RoboDojo checkpoint statistics")
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != 3500 or int(info["total_frames"]) != 1859602:
        raise RuntimeError("RoboDojo dataset/checkpoint count mismatch")

    config_path = args.checkpoint_dir / "config.full.yaml"
    config = OmegaConf.load(config_path if config_path.is_file() else args.checkpoint_dir / "config.yaml")
    horizon = int(config.framework.action_model.action_horizon)
    selected = select_anchors(
        args.dataset_root,
        horizon=horizon,
        max_samples=args.max_samples,
        seed=args.selection_seed,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model, framework, model_config = load_model(args, checkpoint_file, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - started
    records = []
    condition_seconds = 0.0
    head_seconds = 0.0
    rendered_prompts = {}
    for sample_index, anchor in enumerate(selected):
        images, state_values, action_values = load_anchor(
            args.dataset_root,
            anchor,
            horizon=horizon,
            state_stats=stats["state"],
            action_stats=stats["action"],
        )
        synchronize(device)
        started = time.perf_counter()
        condition, attention_mask, state, rendered_prompt = encode_condition(
            model,
            framework,
            images,
            anchor["task"],
            state_values,
        )
        synchronize(device)
        condition_seconds += time.perf_counter() - started
        rendered_prompts[str(anchor["task_index"])] = rendered_prompt
        actions = torch.from_numpy(action_values).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(
                args.noise_seed + sample_index * args.noise_samples + noise_index
            )
            noise = torch.randn(actions.shape, generator=generator, device=device, dtype=actions.dtype)
            for noise_level in args.noise_levels:
                synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                    prediction, target = fixed_velocity_prediction(
                        model.action_model,
                        framework,
                        condition,
                        attention_mask,
                        state,
                        actions,
                        noise,
                        1.0 - noise_level,
                    )
                synchronize(device)
                head_seconds += time.perf_counter() - started
                records.append(
                    {
                        **anchor,
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "model_time": 1.0 - noise_level,
                        **metrics(prediction, target),
                    }
                )

    result = {
        "schema_version": "openpi.external_starvla_robodojo_velocity.v1",
        "checkpoint": {
            "directory": str(args.checkpoint_dir),
            "file": str(checkpoint_file),
            "framework": framework,
            "base_vlm": args.base_vlm,
            "source_revision": git_revision(args.starvla_root),
        },
        "dataset": {
            "root": str(args.dataset_root),
            "official_episodes": int(info["total_episodes"]),
            "official_frames": int(info["total_frames"]),
            "official_tasks": int(info["total_tasks"]),
            "subset_episode_files": len(list((args.dataset_root / "data").rglob("episode_*.parquet"))),
            "subset_video_files": len(list((args.dataset_root / "videos").rglob("episode_*.mp4"))),
            "same_source_evidence": "checkpoint and official dataset trajectory/transition counts match exactly",
        },
        "protocol": {
            "sample_count": len(selected),
            "selection_seed": args.selection_seed,
            "selection": f"one deterministic nonboundary valid-H{horizon} anchor per available training task",
            "noise_samples_per_anchor": args.noise_samples,
            "noise_seed": args.noise_seed,
            "noise_levels": args.noise_levels,
            "time_convention": "StarVLA t=0 pure Gaussian noise, t=1 data; reported noise_level=1-t",
            "fixed_noise_per_anchor_across_times": True,
            "horizon": horizon,
            "action_dim": 14,
            "condition": (
                "three RGB cameras resized to 224x224 plus task text and q99-normalized state; QwenGR00T uses the "
                "state MLP, QwenPI_v3 injects 256-bin state tokens into text"
            ),
            "normalization": "all 14 state/action dimensions use checkpoint q01/q99 to [-1,1]",
            "attention_implementation": "sdpa (hardcoded by the current official Qwen3 StarVLA interface)",
            "model_config_name": str(model_config.framework.name),
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "condition_seconds": condition_seconds,
            "action_head_seconds": head_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        },
        "rendered_prompts": rendered_prompts,
        "selected_anchors": selected,
        "aggregate": aggregate(records, args.noise_levels),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "checkpoint": result["checkpoint"],
        "protocol": result["protocol"],
        "runtime": result["runtime"],
        "aggregate": result["aggregate"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
