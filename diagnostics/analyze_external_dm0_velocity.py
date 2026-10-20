#!/usr/bin/env python3
"""Probe the Dexbotic DM0 flow head on a selective Table30v2 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import av
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.model.dm0.dm0_utils import make_attn_mask_2d
from dexbotic.model.dm0.dm0_utils import make_attn_mask_4d
from dexbotic.tokenization.process import DM0Tokenization
from hardware.dosw1.convert_table30v2_dosw1_to_dexdata import build_state
from hardware.dosw1.convert_table30v2_dosw1_to_dexdata import is_static_frame
from hardware.dosw1.convert_table30v2_dosw1_to_dexdata import is_valid_state
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CHECKPOINT_REVISION = "5c5b6f1344179df6b207e50dc01898d295d29476"
DATASET_REVISION = "c58ad2cc76ce722ea54de51f9fae03012a698f47"
DEXBOTIC_SOURCE_REVISION = "6356c98e6b75d3f4fbc8765913d64ddfd9fe0823"
CHECKPOINT_SHA256 = {
    "model-00001-of-00002.safetensors": "fcf0773b90b487da95265c0a2d79a750b58acd3d67761a55c4b8e774ef3f20ad",
    "model-00002-of-00002.safetensors": "9dbccf999cb0dacbe3310fedad8e11abbbe6d9b4b151b0723350ce74f206a617",
}
CONTINUOUS_INDICES = tuple(range(6)) + tuple(range(7, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def frame_at(video_path: Path, frame_index: int) -> Image.Image:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def filtered_episode_states(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    left_rows = read_jsonl(episode_dir / "states" / "left_states.jsonl")
    right_rows = read_jsonl(episode_dir / "states" / "right_states.jsonl")
    meta = json.loads((episode_dir / "meta" / "episode_meta.json").read_text())
    frame_count = min(int(meta.get("frames", len(left_rows))), len(left_rows), len(right_rows))
    states = []
    frame_indices = []
    previous_state = None
    for frame_index in range(frame_count):
        state = build_state(left_rows[frame_index], right_rows[frame_index])
        if not is_valid_state(state, gripper_min=-1e-3, gripper_max=8e-2):
            continue
        if previous_state is not None and is_static_frame(
            previous_state,
            state,
            joint_atol=5e-4,
            gripper_atol=1e-3,
        ):
            continue
        states.append(state)
        frame_indices.append(frame_index)
        previous_state = state
    return np.asarray(states, dtype=np.float32), np.asarray(frame_indices, dtype=np.int64)


def load_samples(dataset_root: Path, checkpoint_dir: Path, horizon: int, max_episodes: int) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
    if manifest["revision"] != DATASET_REVISION:
        raise ValueError(f"Dataset revision mismatch: {manifest['revision']} != {DATASET_REVISION}")
    task_root = dataset_root / "stack_bowls"
    prompt = json.loads((task_root / "task_desc.json").read_text())["prompt"]
    norm = json.loads((checkpoint_dir / "norm_stats.json").read_text())["action"]
    lower = np.asarray(norm["min"], dtype=np.float32)
    upper = np.asarray(norm["max"], dtype=np.float32)
    if lower.shape != (32,) or upper.shape != (32,):
        raise ValueError(f"Expected D32 checkpoint action statistics, got {lower.shape} and {upper.shape}")

    samples = []
    for episode_name in manifest["episodes"][:max_episodes]:
        episode_dir = task_root / "data" / episode_name
        states, frame_indices = filtered_episode_states(episode_dir)
        if len(states) <= horizon:
            raise ValueError(f"{episode_name} has only {len(states)} filtered frames")
        anchor = (len(states) - horizon - 1) // 2
        current = states[anchor]
        future = states[anchor + 1 : anchor + 1 + horizon]
        physical_actions = future - current
        physical_actions[:, (6, 13)] = future[:, (6, 13)]
        padded_actions = np.pad(physical_actions, ((0, 0), (0, 32 - physical_actions.shape[1])))
        model_actions = ((padded_actions - lower) / (upper - lower + 1e-6) * 2.0 - 1.0).astype(np.float32)
        original_frame = int(frame_indices[anchor])
        videos = episode_dir / "videos"
        images = [
            frame_at(videos / video_name, original_frame)
            for video_name in ("cam_high_rgb.mp4", "cam_left_wrist_rgb.mp4", "cam_right_wrist_rgb.mp4")
        ]
        samples.append(
            {
                "sample_id": episode_name,
                "episode": episode_name,
                "filtered_frame_index": anchor,
                "original_frame_index": original_frame,
                "prompt": prompt,
                "action": model_actions,
                "physical_action": physical_actions,
                "images": images,
            }
        )
    return samples, manifest


def prepare_batch(model, tokenizer, sample: dict, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    images = model.process_images(sample["images"]).to(dtype=model.dtype)
    image_masks = torch.ones(3, dtype=torch.bool)
    tokens = DM0Tokenization(tokenizer)([{"from": "human", "value": sample["prompt"]}])["input_ids"]
    input_ids = torch.from_numpy(tokens).long()
    attention_mask = input_ids != tokenizer.pad_token_id
    return {
        "input_ids": input_ids.unsqueeze(0).repeat(batch_size, 1).to(device),
        "attention_mask": attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device),
        "images": images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device),
        "image_masks": image_masks.unsqueeze(0).repeat(batch_size, 1).to(device),
        "actions": torch.from_numpy(sample["action"])
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .to(device=device, dtype=model.model.action_in_proj.weight.dtype),
    }


def fixed_velocity_prediction(
    model,
    inputs: dict[str, torch.Tensor],
    noise: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = inputs["actions"]
    noisy_actions = times[:, None, None] * noise + (1.0 - times[:, None, None]) * actions
    target = noise - actions
    prefix_hidden, prefix_padding, prefix_attention = model.get_prefix_hidden_states(
        inputs["input_ids"], inputs["attention_mask"], inputs["images"], inputs["image_masks"]
    )
    suffix_hidden, suffix_padding, suffix_attention = model.get_suffix_hidden_states(noisy_actions, times)
    if model.config.bf16:
        prefix_hidden = prefix_hidden.to(torch.bfloat16)
        suffix_hidden = suffix_hidden.to(torch.bfloat16)
    padding_mask = torch.cat([prefix_padding, suffix_padding], dim=1)
    attention_pattern = torch.cat([prefix_attention, suffix_attention], dim=1)
    attention_mask = make_attn_mask_4d(
        make_attn_mask_2d(padding_mask=padding_mask, attn_mask=attention_pattern), dtype=prefix_hidden.dtype
    )
    prefix_positions = torch.cumsum(prefix_padding, dim=1) - 1
    suffix_positions = torch.sum(prefix_padding, dim=-1)[:, None] + torch.cumsum(suffix_padding, dim=1) - 1
    positions = torch.cat([prefix_positions, suffix_positions], dim=1)
    (_, suffix_output), _ = model._merged_attention_forward(  # noqa: SLF001
        module_list=[model.model.llm, model.model.action_expert.model],
        attention_mask=attention_mask,
        position_ids=positions,
        past_key_values=None,
        input_embeds_list=[prefix_hidden, suffix_hidden],
        use_cache=False,
    )
    prediction = model.model.action_out_proj(suffix_output[:, -model.config.chunk_size :])
    return prediction, target


def metrics(prediction: torch.Tensor, target: torch.Tensor, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    prediction = prediction[..., indices].float()
    target = target[..., indices].float()
    mse = torch.mean(torch.square(prediction - target), dim=(-2, -1))
    cosine = torch.nn.functional.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    return mse.cpu().numpy(), (1.0 - cosine).cpu().numpy()


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
    model = (
        DM0ForCausalLM.from_pretrained(
            args.checkpoint_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=False)
    samples, manifest = load_samples(args.dataset_root, args.checkpoint_dir, model.config.chunk_size, args.max_episodes)
    model_load_seconds = time.perf_counter() - load_started

    records = []
    forward_seconds = 0.0
    level_count = len(args.noise_levels)
    for sample_index, sample in enumerate(samples):
        inputs = prepare_batch(model, tokenizer, sample, level_count, device)
        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(args.seed + sample_index * 100_003 + noise_index)
            base_noise = torch.randn(
                inputs["actions"].shape[1:], generator=generator, device=device, dtype=inputs["actions"].dtype
            )
            noise = base_noise.unsqueeze(0).repeat(level_count, 1, 1)
            times = torch.tensor(args.noise_levels, device=device, dtype=inputs["actions"].dtype)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction, target = fixed_velocity_prediction(model, inputs, noise, times)
            torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - started
            all_mse, all_cosine = metrics(prediction, target, tuple(range(14)))
            continuous_mse, continuous_cosine = metrics(prediction, target, CONTINUOUS_INDICES)
            for level_index, noise_level in enumerate(args.noise_levels):
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "episode": sample["episode"],
                        "filtered_frame_index": sample["filtered_frame_index"],
                        "original_frame_index": sample["original_frame_index"],
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "all_action": {
                            "mse": float(all_mse[level_index]),
                            "cosine_error": float(all_cosine[level_index]),
                        },
                        "continuous_action": {
                            "mse": float(continuous_mse[level_index]),
                            "cosine_error": float(continuous_cosine[level_index]),
                        },
                    }
                )

    aggregate = []
    bootstrap_rng = np.random.default_rng(args.seed)
    sample_ids = [sample["sample_id"] for sample in samples]
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
        "schema_version": "openpi.external_dm0_velocity_probe.v1",
        "checkpoint": str(args.checkpoint_dir),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_revision": DEXBOTIC_SOURCE_REVISION,
        "dataset_repo": manifest["repo_id"],
        "dataset_revision": manifest["revision"],
        "dataset_probe": {
            "source_file": manifest["source_file"],
            "source_byte_range": manifest["source_byte_range"],
            "source_prefix_sha256": manifest["source_prefix_sha256"],
        },
        "protocol": {
            "tasks": 1,
            "episodes": len(samples),
            "selection": manifest["selection"] + "; one centered H50-valid filtered anchor per episode",
            "condition": "three exact RGB video frames and exact task prompt; DM0 does not consume proprioception",
            "target": "H50 x 14 bimanual future joint states relative to anchor; gripper dimensions 6 and 13 absolute",
            "normalization": "checkpoint D32 quantile min/max mapping to [-1,1] after zero padding",
            "normalization_gate": (
                "exact public Table30v2-to-DexData validity/static-frame filters followed by the released "
                "AddAction/Pad/AddTrajectory/DeltaAction/ActionNorm semantics"
            ),
            "time_convention": "noise_level t uses x_t=t*noise+(1-t)*action and target noise-action",
            "training_time_schedule": "Beta(1.5,1.0), then t=0.999*sample+0.001",
            "training_objective": "native DM0 flow-matching action loss; checkpoint config also records ar_loss=true",
            "noise_levels": args.noise_levels,
            "noise_samples_per_episode": args.noise_samples,
            "seed": args.seed,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "model_load_seconds": model_load_seconds,
            "forward_seconds_total": forward_seconds,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        },
        "aggregate": aggregate,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol": result["protocol"], "runtime": result["runtime"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
