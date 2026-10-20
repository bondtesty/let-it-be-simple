#!/usr/bin/env python3
"""Probe the Dexbotic PI0 velocity field on a selective ManiSkill2 subset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import av
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM
from dexbotic.model.pi0.pi0_arch import make_attn_mask
from dexbotic.model.pi0.pi0_arch import make_attn_mask_4d
from dexbotic.tokenization.process import Pi0Tokenization
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CHECKPOINT_REVISION = "a9cf8e950eb0af56a1ed4fc199dc07e11099f6db"
DEXBOTIC_SOURCE_REVISION = "6356c98e6b75d3f4fbc8765913d64ddfd9fe0823"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument(
        "--action-normalization",
        choices=("checkpoint-minmax", "none"),
        default="checkpoint-minmax",
    )
    parser.add_argument(
        "--action-target",
        choices=("step-delta", "anchor-relative"),
        default="step-delta",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def frame_at(video_path: Path, frame_index: int) -> Image.Image:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def periodic_delta(future: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = future - current
    for dimension in (3, 4, 5):
        delta[:, dimension] = (delta[:, dimension] + math.pi) % (2 * math.pi) - math.pi
    delta[:, 6] = future[:, 6]
    return delta


def load_samples(
    dataset_root: Path,
    checkpoint_dir: Path,
    horizon: int,
    max_episodes: int,
    action_normalization: str,
    action_target: str,
) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
    norm = json.loads((checkpoint_dir / "norm_stats.json").read_text())["default"]
    lower = np.asarray(norm["min"], dtype=np.float32)
    upper = np.asarray(norm["max"], dtype=np.float32)
    if lower.shape != (7,) or upper.shape != (7,):
        raise ValueError(f"Expected 7D min/max normalization, got {lower.shape} and {upper.shape}")

    samples = []
    for episode in manifest["episodes"][:max_episodes]:
        rows = [json.loads(line) for line in (dataset_root / episode["jsonl"]).read_text().splitlines()]
        if len(rows) <= horizon:
            raise ValueError(f"{episode['episode']} has only {len(rows)} frames")
        anchor = (len(rows) - horizon - 1) // 2
        current = np.asarray(rows[anchor]["state"], dtype=np.float32)
        future = np.asarray([row["state"] for row in rows[anchor + 1 : anchor + 1 + horizon]], dtype=np.float32)
        if action_target == "step-delta":
            action_states = np.asarray([row["state"] for row in rows[anchor : anchor + horizon]], dtype=np.float32)
            physical_actions = periodic_delta(future, action_states)
        elif action_target == "anchor-relative":
            physical_actions = periodic_delta(future, current)
        else:
            raise ValueError(f"Unsupported action target: {action_target}")
        if action_normalization == "checkpoint-minmax":
            model_actions = 2.0 * (physical_actions - lower) / (upper - lower + 1e-6) - 1.0
        elif action_normalization == "none":
            model_actions = physical_actions
        else:
            raise ValueError(f"Unsupported action normalization: {action_normalization}")
        padded_state = np.pad(current, (0, 32 - len(current))).astype(np.float32)
        padded_actions = np.pad(model_actions, ((0, 0), (0, 32 - model_actions.shape[1]))).astype(np.float32)
        samples.append(
            {
                "sample_id": f"{episode['task']}/{episode['episode']}",
                "task": episode["task"],
                "episode": episode["episode"],
                "frame_index": anchor,
                "prompt": rows[anchor]["prompt"],
                "state": padded_state,
                "action": padded_actions,
                "physical_action": physical_actions,
                "image": frame_at(dataset_root / episode["video"], rows[anchor]["images_1"]["frame_idx"]),
            }
        )
    return samples, manifest


def prepare_batch(model, tokenizer, sample: dict, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    image = model.process_images([sample["image"]]).to(dtype=model.dtype)
    images = torch.cat([image, torch.zeros_like(image).repeat(2, 1, 1, 1)], dim=0)
    image_masks = torch.tensor([True, False, False])
    tokenization = Pi0Tokenization(tokenizer)
    tokens = tokenization([{"value": sample["prompt"]}])["input_ids"]
    input_ids = torch.from_numpy(tokens).long()
    attention_mask = input_ids != tokenizer.pad_token_id
    return {
        "input_ids": input_ids.unsqueeze(0).repeat(batch_size, 1).to(device),
        "attention_mask": attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device),
        "images": images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device),
        "image_masks": image_masks.unsqueeze(0).repeat(batch_size, 1).to(device),
        "states": torch.from_numpy(sample["state"])
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device=device, dtype=model.dtype),
        "actions": torch.from_numpy(sample["action"])
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .to(device=device, dtype=model.dtype),
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
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
        inputs["input_ids"], inputs["attention_mask"], inputs["images"], inputs["image_masks"]
    )
    suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(inputs["states"], noisy_actions, times)
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
    attention_mask = make_attn_mask_4d(make_attn_mask(input_mask, ar_mask))
    positions = torch.cumsum(input_mask, dim=1) - 1
    position_embeddings = model.model.llm.rotary_emb(prefix_tokens, positions)
    (_, suffix_output), _, _, _ = model._inner_forward_mot(  # noqa: SLF001
        [model.model.llm, model.model.action_expert],
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    prediction = model.model.action_out_proj(suffix_output[:, -model.config.chunk_size :])
    return prediction, target


def metrics(prediction: torch.Tensor, target: torch.Tensor, dimensions: slice) -> tuple[np.ndarray, np.ndarray]:
    prediction = prediction[..., dimensions].float()
    target = target[..., dimensions].float()
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
    model = Pi0ForCausalLM.from_pretrained(
        args.checkpoint_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    tokenizer.model_max_length = 48
    samples, manifest = load_samples(
        args.dataset_root,
        args.checkpoint_dir,
        model.config.chunk_size,
        args.max_episodes,
        args.action_normalization,
        args.action_target,
    )
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
            all_mse, all_cosine = metrics(prediction, target, slice(0, 7))
            continuous_mse, continuous_cosine = metrics(prediction, target, slice(0, 6))
            for level_index, noise_level in enumerate(args.noise_levels):
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "task": sample["task"],
                        "episode": sample["episode"],
                        "frame_index": sample["frame_index"],
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
        "schema_version": "openpi.external_dexbotic_pi0_velocity_probe.v1",
        "checkpoint": str(args.checkpoint_dir),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": DEXBOTIC_SOURCE_REVISION,
        "dataset_repo": manifest["repo_id"],
        "dataset_revision": manifest["revision"],
        "protocol": {
            "tasks": len({sample["task"] for sample in samples}),
            "episodes": len(samples),
            "selection": manifest["selection"],
            "condition": "one RGB frame, raw padded 7D ManiSkill state, exact task prompt; two image slots masked",
            "target": (
                "H50 x 7 consecutive one-step deltas, with gripper absolute"
                if args.action_target == "step-delta"
                else "H50 x 7 future states relative to the anchor state, with gripper absolute"
            ),
            "action_target_profile": args.action_target,
            "normalization": (
                "checkpoint min/max mapped to [-1,1] before 32D zero padding"
                if args.action_normalization == "checkpoint-minmax"
                else "no action normalization before 32D zero padding"
            ),
            "action_normalization_profile": args.action_normalization,
            "normalization_gate": (
                "checkpoint 7D stats shape asserted; the released default-only profile is not applied by its "
                "inference transform, and matched smoke tests select raw step deltas over three alternatives"
            ),
            "release_protocol_caveat": (
                "the public benchmark config omits AddAction although the public JSONL has no action field; "
                "step-delta targets are reconstructed from successive states and supported by checkpoint stats "
                "and the matched target/normalization gate, not by a self-consistent public training pipeline"
            ),
            "time_convention": "noise_level t uses x_t=t*noise+(1-t)*action and target noise-action",
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
