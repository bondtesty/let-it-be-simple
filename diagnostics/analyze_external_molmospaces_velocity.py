#!/usr/bin/env python3
"""Probe a native OpenPI pi0.5 checkpoint on a selective MolmoSpaces subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import types

from analyze_openpi_droid_velocity import make_input_transform
import av
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from openpi.training import config as _config

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
SHOULDER_CAMERA = "droid_shoulder_light_randomization"
WRIST_CAMERA = "wrist_camera_zed_mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_droid_jointpos_polaris_molmospaces_lora")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def decode_json_row(row: np.ndarray) -> dict:
    return json.loads(bytes(row).split(b"\0", 1)[0])


def frame_at(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Frame {frame_index} not found in {video_path}")


def video_path(dataset_root: Path, row: dict, camera: str) -> Path:
    root = (dataset_root / row["output_h5"]).parent
    stem = f"{row['episode_id']}__{row['direction']}__success__{camera}.mp4"
    return root / stem


def load_raw_samples(dataset_root: Path, horizon: int, max_episodes: int) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
    rows = manifest["episodes"][:max_episodes]
    samples = []
    for row in rows:
        h5_path = dataset_root / row["output_h5"]
        with h5py.File(h5_path) as trajectory:
            group = trajectory["traj_0"]
            states = [decode_json_row(value) for value in group["obs/agent/qpos"]]
            action_rows = [decode_json_row(value) for value in group["actions/joint_pos"]]
            scene = json.loads(group["obs_scene"][()])
        valid_actions = [value for value in action_rows if "arm" in value and "gripper" in value]
        if len(valid_actions) < horizon:
            raise ValueError(f"{row['episode_id']} has only {len(valid_actions)} valid actions")
        anchor = (len(valid_actions) - horizon) // 2
        state_row = states[anchor]
        state = np.asarray(
            [*state_row["arm"], float(np.mean(state_row["gripper"]))],
            dtype=np.float32,
        )
        actions = np.asarray(
            [[*value["arm"], float(value["gripper"][0]) / 255.0] for value in valid_actions[anchor : anchor + horizon]],
            dtype=np.float32,
        )
        samples.append(
            {
                "episode_id": f"{row['episode_id']}__{row['direction']}",
                "direction": row["direction"],
                "house_index": row["house_index"],
                "frame_index": anchor,
                "instruction": scene["task_description"],
                "raw_state": state,
                "raw_actions": actions,
                "raw": {
                    "observation/exterior_image_1_left": frame_at(
                        video_path(dataset_root, row, SHOULDER_CAMERA), anchor
                    ),
                    "observation/wrist_image_left": frame_at(video_path(dataset_root, row, WRIST_CAMERA), anchor),
                    "observation/joint_position": state[:7],
                    "observation/gripper_position": state[7:8],
                    "actions": actions,
                    "prompt": scene["task_description"],
                },
            }
        )
    return samples, manifest


def numpy_to_jax_observation(rows: list[dict]) -> _model.Observation:
    stacked = jax.tree.map(lambda *values: np.stack(values), *rows)

    def to_array(value):
        array = np.asarray(value)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return jnp.asarray(array)

    return _model.Observation.from_dict(jax.tree.map(to_array, stacked))


def fixed_velocity_prediction(
    model,
    observation: _model.Observation,
    actions: jax.Array,
    noise: jax.Array,
    time_values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    observation = _model.preprocess_observation(None, observation, train=False)
    noisy_actions = time_values[:, None, None] * noise + (1.0 - time_values[:, None, None]) * actions
    target = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, noisy_actions, time_values
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_output), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    prediction = model._unpatchify(  # noqa: SLF001
        model.action_out_proj(suffix_output[:, -model.num_action_tokens :])
    )
    return prediction, target


def metrics(prediction: np.ndarray, target: np.ndarray, dimensions: slice) -> tuple[np.ndarray, np.ndarray]:
    prediction = prediction[..., dimensions]
    target = target[..., dimensions]
    mse = np.mean(np.square(prediction - target), axis=(-2, -1))
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    denominator = np.linalg.norm(prediction_flat, axis=1) * np.linalg.norm(target_flat, axis=1)
    cosine = np.sum(prediction_flat * target_flat, axis=1) / np.maximum(denominator, 1e-12)
    return mse, 1.0 - cosine


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    if args.max_episodes <= 0 or args.noise_samples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("max-episodes, noise-samples, and bootstrap-samples must be positive")
    if any(level <= 0 or level > 1 for level in args.noise_levels):
        raise ValueError("noise levels must be in (0, 1]")

    config = _config.get_config(args.config_name)
    if config.model.action_horizon != 15 or config.model.action_dim < 8 or not config.model.pi05:
        raise ValueError(f"Expected pi0.5 H15 D>=8, got {config.model}")
    samples, manifest = load_raw_samples(args.dataset_root, config.model.action_horizon, args.max_episodes)

    data_config = config.data.create(args.checkpoint_dir / "assets", config.model)
    physical_transform = _transforms.compose(data_config.data_transforms.inputs)
    transform = make_input_transform(config, args.checkpoint_dir)
    transformed_samples = []
    for sample in samples:
        expected_delta = sample["raw_actions"][:, :7] - sample["raw_state"][None, :7]
        physical = physical_transform(copy.deepcopy(sample["raw"]))
        if not np.allclose(physical["actions"][:, :7], expected_delta, atol=1e-6):
            raise RuntimeError(f"Joint-position delta transform mismatch for {sample['episode_id']}")
        if not np.allclose(physical["actions"][:, 7], sample["raw_actions"][:, 7], atol=1e-6):
            raise RuntimeError(f"Gripper transform mismatch for {sample['episode_id']}")
        transformed_samples.append({**sample, "transformed": transform(copy.deepcopy(sample["raw"]))})

    load_started = time.perf_counter()
    params = _model.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    predict_velocity = nnx_utils.module_jit(types.MethodType(fixed_velocity_prediction, model))
    model_load_seconds = time.perf_counter() - load_started

    records = []
    forward_seconds = 0.0
    level_count = len(args.noise_levels)
    for sample_index, sample in enumerate(transformed_samples):
        observation = numpy_to_jax_observation([sample["transformed"]] * level_count)
        actions = jnp.asarray(np.stack([sample["transformed"]["actions"]] * level_count).astype(np.float32))
        for noise_index in range(args.noise_samples):
            rng = np.random.default_rng(args.seed + sample_index * 100_003 + noise_index)
            base_noise = rng.standard_normal(actions.shape[1:], dtype=np.float32)
            noise = jnp.asarray(np.stack([base_noise] * level_count))
            times = jnp.asarray(args.noise_levels, dtype=jnp.float32)
            started = time.perf_counter()
            prediction, target = predict_velocity(observation, actions, noise, times)
            prediction, target = np.asarray(prediction), np.asarray(target)
            forward_seconds += time.perf_counter() - started
            all_mse, all_cosine = metrics(prediction, target, slice(0, 8))
            continuous_mse, continuous_cosine = metrics(prediction, target, slice(0, 7))
            for level_index, noise_level in enumerate(args.noise_levels):
                records.append(
                    {
                        "episode_id": sample["episode_id"],
                        "direction": sample["direction"],
                        "house_index": sample["house_index"],
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
    episode_ids = [sample["episode_id"] for sample in samples]
    for noise_level in args.noise_levels:
        level_records = [record for record in records if record["noise_level"] == noise_level]
        row = {"noise_level": noise_level}
        for group in ("all_action", "continuous_action"):
            row[group] = {}
            for metric_name in ("mse", "cosine_error"):
                values = np.asarray(
                    [
                        np.mean(
                            [
                                record[group][metric_name]
                                for record in level_records
                                if record["episode_id"] == episode_id
                            ]
                        )
                        for episode_id in episode_ids
                    ]
                )
                row[group][metric_name] = {
                    "episode_macro_mean": float(values.mean()),
                    "episode_bootstrap_95ci": bootstrap_interval(values, args.bootstrap_samples, bootstrap_rng),
                }
        aggregate.append(row)

    result = {
        "schema_version": "openpi.external_molmospaces_velocity_probe.v1",
        "checkpoint": str(args.checkpoint_dir),
        "config_name": args.config_name,
        "dataset_repo": manifest["repo_id"],
        "dataset_revision": manifest["revision"],
        "protocol": {
            "split": "house-disjoint validation",
            "condition": "shoulder RGB, wrist RGB, 8D Franka joint/gripper state, exact task description",
            "target": "H15 x 8 absolute joint-position/gripper chunks converted to delta-joint by checkpoint transform",
            "normalization": "checkpoint droid norm_stats; quantile normalization after joint-position delta transform",
            "normalization_gate": "physical first-7 action deltas and unchanged [0,1] gripper asserted sample-wise",
            "gripper_mapping": "action 0/255 mapped to 0/1; state uses mean of two simulator gripper qpos values",
            "selection": "one midpoint anchor per trajectory; balanced left/right validation episodes",
            "time_convention": "noise_level t uses x_t=t*noise+(1-t)*action and target noise-action",
            "noise_levels": args.noise_levels,
            "noise_samples_per_anchor": args.noise_samples,
            "episodes": len(samples),
            "directions": {
                direction: sum(s["direction"] == direction for s in samples) for direction in ("left", "right")
            },
            "seed": args.seed,
        },
        "runtime": {
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "model_load_seconds": model_load_seconds,
            "forward_seconds_total": forward_seconds,
        },
        "aggregate": aggregate,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"protocol": result["protocol"], "runtime": result["runtime"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
