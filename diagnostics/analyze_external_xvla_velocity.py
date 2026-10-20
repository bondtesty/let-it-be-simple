#!/usr/bin/env python3
"""Probe public X-VLA target-prediction checkpoints at fixed flow times.

X-VLA trains with ``x_t = t * noise + (1 - t) * action`` but predicts the
clean action rather than velocity.  For continuous action dimensions, the
equivalent velocity estimate is

    v_hat = (x_t - action_hat) / t,

whose reference is ``noise - action``.  The script reports both the implied
velocity loss and the native clean-action prediction loss so the
parameterization-induced ``1 / t**2`` scaling remains explicit.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

from analyze_external_starvla_velocity import decode_video_frame
from analyze_external_starvla_velocity import git_revision
from analyze_external_starvla_velocity import read_jsonl
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import transformers

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
DOMAIN_IDS = {"bridge": 0, "rt1": 1, "calvin": 2}
ACTIVE_CONTINUOUS_INDICES = tuple(range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=sorted(DOMAIN_IDS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument(
        "--calvin-model-fps",
        type=float,
        default=30.0,
        help="Row-time convention for CALVIN. X-VLA's official CalvinHandler uses 30 Hz; use 10 only as a wrapper sensitivity.",
    )
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


def dataset_provenance(dataset_root: Path, dataset_kind: str) -> dict[str, str]:
    if dataset_kind in {"bridge", "rt1"}:
        manifest_path = dataset_root / "probe_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        return {
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
            "manifest": str(manifest_path),
        }
    metadata_path = dataset_root / ".cache" / "huggingface" / "download" / "README.md.metadata"
    revision = metadata_path.read_text().splitlines()[0]
    return {
        "repo_id": "ducido/calvin_task_D_D_lerobo_format",
        "revision": revision,
        "manifest": str(dataset_root / "meta" / "episodes.jsonl"),
    }


def decode_inline_image(cell: dict) -> Image.Image:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise TypeError(f"Expected inline LeRobot image bytes, got {type(cell)}")
    return Image.open(BytesIO(cell["bytes"])).convert("RGB")


def euler_to_rotation_6d(euler: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_euler("xyz", euler, degrees=False).as_matrix()
    return matrices[..., :, :2].reshape(euler.shape[:-1] + (6,)).astype(np.float32)


def quaternion_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_quat(quaternion, scalar_first=False).as_matrix()
    return matrices[..., :, :2].reshape(quaternion.shape[:-1] + (6,)).astype(np.float32)


def interpolate_trajectory(values: np.ndarray, timestamps: np.ndarray, start: int, duration: float) -> np.ndarray:
    start_time = float(timestamps[start])
    query = np.linspace(start_time, min(start_time + duration, float(timestamps[-1])), 31, dtype=np.float64)
    return np.stack([np.interp(query, timestamps, values[:, dim]) for dim in range(values.shape[1])], axis=-1).astype(
        np.float32
    )


def build_single_arm_trajectory(
    state: np.ndarray,
    action: np.ndarray | None,
    timestamps: np.ndarray,
    frame_index: int,
    dataset_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset_kind == "bridge":
        if action is None:
            raise RuntimeError("Bridge action array is required for gripper conversion")
        left = np.concatenate([state[:, :3], euler_to_rotation_6d(state[:, 3:6]), 1.0 - action[:, -1:]], axis=-1)
        duration = 5.0
    elif dataset_kind == "rt1":
        left = np.concatenate([state[:, :3], quaternion_to_rotation_6d(state[:, 3:7]), state[:, -1:]], axis=-1)
        duration = 10.0
    elif dataset_kind == "calvin":
        left = np.concatenate(
            [state[:, :3], euler_to_rotation_6d(state[:, 3:6]), (state[:, -1:] < 0.0).astype(np.float32)],
            axis=-1,
        )
        duration = 1.0
    else:
        raise ValueError(dataset_kind)
    right = np.zeros_like(left)
    absolute = np.concatenate([left, right], axis=-1)
    trajectory = interpolate_trajectory(absolute, timestamps, frame_index, duration)
    return trajectory[0], trajectory[1:]


def select_bridge_anchors(dataset_root: Path, max_samples: int, seed: int) -> list[dict]:
    manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
    rng = np.random.default_rng(seed)
    selected = []
    for task_index, task_entry in enumerate(manifest["tasks"]):
        candidates = [
            episode for episode in task_entry["episodes"] if int(episode["length"]) > 10 and episode.get("video")
        ]
        rng.shuffle(candidates)
        if not candidates:
            raise RuntimeError(f"No X-VLA-valid Bridge episode for {task_entry['task']!r}")
        episode = candidates[0]
        candidate_count = int(episode["length"]) - 10
        low = min(2, candidate_count - 1)
        frame_index = int(rng.integers(low, candidate_count)) if candidate_count > low else low
        selected.append(
            {
                "task": task_entry["task"],
                "task_index": task_index,
                "episode_index": int(episode["episode_index"]),
                "episode_length": int(episode["length"]),
                "frame_index": frame_index,
                "parquet": episode["parquet"],
                "video": episode["video"],
            }
        )
        if len(selected) == max_samples:
            break
    return selected


def select_rt1_anchors(dataset_root: Path, max_samples: int, seed: int) -> list[dict]:
    manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
    rng = np.random.default_rng(seed)
    selected = []
    for task_index, task_entry in enumerate(manifest["tasks"]):
        candidates = [
            episode for episode in task_entry["episodes"] if int(episode["length"]) > 6 and episode.get("video")
        ]
        rng.shuffle(candidates)
        if not candidates:
            raise RuntimeError(f"No X-VLA-valid RT1 episode for {task_entry['task']!r}")
        episode = candidates[0]
        candidate_count = int(episode["length"]) - 6
        table = pd.read_parquet(dataset_root / episode["parquet"], columns=["observation.state", "timestamp"])
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        timestamps = table["timestamp"].to_numpy(dtype=np.float64)
        candidate_indices = np.arange(candidate_count, dtype=np.int64)
        rng.shuffle(candidate_indices)
        frame_index = None
        for candidate_index in candidate_indices:
            proprio, target = build_single_arm_trajectory(state, None, timestamps, int(candidate_index), "rt1")
            if float(np.max(np.abs(target[0] - proprio))) >= 1e-5:
                frame_index = int(candidate_index)
                break
        if frame_index is None:
            raise RuntimeError(f"No nonstatic X-VLA-valid RT1 anchor for {task_entry['task']!r}")
        selected.append(
            {
                "task": task_entry["task"],
                "task_index": task_index,
                "episode_index": int(episode["episode_index"]),
                "episode_length": int(episode["length"]),
                "frame_index": frame_index,
                "parquet": episode["parquet"],
                "video": episode["video"],
            }
        )
        if len(selected) == max_samples:
            break
    return selected


def select_calvin_anchors(dataset_root: Path, max_samples: int, seed: int) -> list[dict]:
    available = {
        int(path.stem.removeprefix("episode_")): path for path in dataset_root.glob("data/*/episode_*.parquet")
    }
    episodes = [
        episode
        for episode in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
        if int(episode["episode_index"]) in available and int(episode["length"]) > 20
    ]
    tasks = {int(row["task_index"]): row["task"] for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")}
    task_ids = {task: index for index, task in tasks.items()}
    rng = np.random.default_rng(seed)
    selected = []
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        candidate_count = int(episode["length"]) - 20
        low = min(2, candidate_count - 1)
        frame_index = int(rng.integers(low, candidate_count)) if candidate_count > low else low
        task = episode["tasks"][0]
        selected.append(
            {
                "task": task,
                "task_index": task_ids[task],
                "episode_index": int(episode["episode_index"]),
                "episode_length": int(episode["length"]),
                "frame_index": frame_index,
                "parquet": str(available[int(episode["episode_index"])].relative_to(dataset_root)),
            }
        )
        if len(selected) == max_samples:
            break
    return selected


def select_anchors(args: argparse.Namespace) -> list[dict]:
    if args.dataset_kind == "bridge":
        selected = select_bridge_anchors(args.dataset_root, args.max_samples, args.selection_seed)
    elif args.dataset_kind == "rt1":
        selected = select_rt1_anchors(args.dataset_root, args.max_samples, args.selection_seed)
    else:
        selected = select_calvin_anchors(args.dataset_root, args.max_samples, args.selection_seed)
    if len(selected) != min(args.max_samples, 40):
        raise RuntimeError(f"Expected {min(args.max_samples, 40)} anchors, selected {len(selected)}")
    return selected


def timestamp_grid_audit(dataset_root: Path, selected: list[dict], dataset_kind: str, calvin_model_fps: float) -> dict:
    expected_fps = {"bridge": 5.0, "rt1": 3.0, "calvin": 10.0}[dataset_kind]
    errors = []
    for parquet in sorted({anchor["parquet"] for anchor in selected}):
        timestamps = pd.read_parquet(dataset_root / parquet, columns=["timestamp"])["timestamp"].to_numpy(
            dtype=np.float64
        )
        reference = np.arange(timestamps.size, dtype=np.float64) / expected_fps
        errors.append(float(np.max(np.abs((timestamps - timestamps[0]) - reference))))
    maximum = max(errors)
    if maximum > 1e-5:
        raise RuntimeError(f"LeRobot timestamp-grid mismatch: {maximum:.3e}")
    audit = {
        "expected_fps": expected_fps,
        "episodes": len(errors),
        "maximum_abs_seconds": maximum,
        "passed_threshold": 1e-5,
    }
    if dataset_kind == "calvin":
        audit.update(
            {
                "lerobot_wrapper_fps": expected_fps,
                "xvla_official_handler_fps": 30.0,
                "model_row_time_fps": calvin_model_fps,
                "timebase_override": not math.isclose(calvin_model_fps, expected_fps),
            }
        )
    return audit


def load_bridge_anchor(dataset_root: Path, anchor: dict) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    table = pd.read_parquet(
        dataset_root / anchor["parquet"], columns=["observation.state", "action", "timestamp", "frame_index"]
    )
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(table["action"].to_numpy()).astype(np.float32)
    timestamps = table["timestamp"].to_numpy(dtype=np.float64)
    frame_index = anchor["frame_index"]
    if int(table.iloc[frame_index]["frame_index"]) != frame_index:
        raise RuntimeError("Bridge frame index mismatch")
    proprio, target = build_single_arm_trajectory(state, action, timestamps, frame_index, "bridge")
    image = decode_video_frame(dataset_root / anchor["video"], frame_index)
    return [image], proprio, target


def load_calvin_anchor(
    dataset_root: Path, anchor: dict, model_fps: float
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    table = pd.read_parquet(
        dataset_root / anchor["parquet"],
        columns=["image", "wrist_image", "state", "timestamp", "frame_index", "task_index"],
    )
    state = np.stack(table["state"].to_numpy()).astype(np.float32)
    # The selective LeRobot wrapper declares 10 Hz, whereas the released
    # X-VLA CalvinHandler constructs row times at 30 Hz.  Primary runs use the
    # official model convention; 10 Hz is available as an explicit sensitivity.
    timestamps = np.arange(len(table), dtype=np.float64) / model_fps
    frame_index = anchor["frame_index"]
    row = table.iloc[frame_index]
    if int(row["frame_index"]) != frame_index or int(row["task_index"]) != anchor["task_index"]:
        raise RuntimeError("CALVIN frame/task index mismatch")
    proprio, target = build_single_arm_trajectory(state, None, timestamps, frame_index, "calvin")
    return [decode_inline_image(row["image"]), decode_inline_image(row["wrist_image"])], proprio, target


def load_rt1_anchor(dataset_root: Path, anchor: dict) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    table = pd.read_parquet(dataset_root / anchor["parquet"], columns=["observation.state", "timestamp", "frame_index"])
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    timestamps = table["timestamp"].to_numpy(dtype=np.float64)
    frame_index = anchor["frame_index"]
    if int(table.iloc[frame_index]["frame_index"]) != frame_index:
        raise RuntimeError("RT1 frame index mismatch")
    proprio, target = build_single_arm_trajectory(state, None, timestamps, frame_index, "rt1")
    image = decode_video_frame(dataset_root / anchor["video"], frame_index)
    return [image], proprio, target


def load_anchor(args: argparse.Namespace, anchor: dict) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    if args.dataset_kind == "bridge":
        return load_bridge_anchor(args.dataset_root, anchor)
    if args.dataset_kind == "rt1":
        return load_rt1_anchor(args.dataset_root, anchor)
    return load_calvin_anchor(args.dataset_root, anchor, args.calvin_model_fps)


def load_model(checkpoint_dir: Path, device: torch.device):
    from transformers import AutoConfig
    from transformers import AutoModel
    from transformers import AutoProcessor

    # X-VLA pins transformers<=4.51.3.  With that release, leaving ``use_fast``
    # unset reproduces the checkpoint's saved pairing: a fast BART tokenizer
    # (the repo ships tokenizer.json but no merges.txt) and the slow CLIP image
    # processor declared by preprocessor_config.json.
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True, local_files_only=True)
    config.florence_config._attn_implementation_internal = "eager"  # noqa: SLF001
    model, loading = AutoModel.from_pretrained(
        checkpoint_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    problems = {
        key: loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading.get(key)
    }
    if problems:
        raise RuntimeError(f"Non-strict X-VLA checkpoint load: {problems}")
    return model.to(device=device, dtype=torch.float32).eval(), processor, loading


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    mse = torch.mean((prediction - target) ** 2)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=0)
    return {
        "mse": float(mse.item()),
        "cosine_error": float((1.0 - cosine).item()),
        "prediction_norm": float(torch.linalg.vector_norm(prediction).item()),
        "target_norm": float(torch.linalg.vector_norm(target).item()),
    }


def aggregate(records: list[dict], noise_levels: list[float]) -> list[dict]:
    rows = []
    for noise_level in noise_levels:
        subset = [row for row in records if row["noise_level"] == noise_level]
        result = {"noise_level": noise_level, "model_time": noise_level}
        for key in ("continuous_action", "target_prediction"):
            values = np.asarray([row[key]["mse"] for row in subset], dtype=np.float64)
            result[key] = {
                "mse_mean": float(values.mean()),
                "mse_se": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
            }
        rows.append(result)
    return rows


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.noise_samples <= 0:
        raise ValueError("sample counts must be positive")
    if any(level <= 0 or level > 1 for level in args.noise_levels):
        raise ValueError("noise levels must lie in (0, 1]")
    if args.dataset_kind == "calvin" and args.calvin_model_fps not in {10.0, 30.0}:
        raise ValueError("CALVIN model row-time must be the official 30 Hz or the explicit 10 Hz sensitivity")
    checkpoint_file = args.checkpoint_dir / "model.safetensors"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    actual_sha256 = file_sha256(checkpoint_file)
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}")

    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    if config.get("action_mode") != "ee6d" or int(config.get("num_actions", -1)) != 30:
        raise RuntimeError(
            f"Unexpected X-VLA action contract: {config.get('action_mode')}, H{config.get('num_actions')}"
        )
    selected = select_anchors(args)
    timing_audit = timestamp_grid_audit(args.dataset_root, selected, args.dataset_kind, args.calvin_model_fps)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    model, processor, loading = load_model(args.checkpoint_dir, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    domain_id = torch.tensor([DOMAIN_IDS[args.dataset_kind]], dtype=torch.long, device=device)
    active = torch.tensor(ACTIVE_CONTINUOUS_INDICES, dtype=torch.long, device=device)

    records = []
    condition_seconds = 0.0
    head_seconds = 0.0
    for sample_index, anchor in enumerate(selected):
        images, proprio_np, action_np = load_anchor(args, anchor)
        if float(np.max(np.abs(action_np[0] - proprio_np))) < 1e-5:
            raise RuntimeError(f"Static X-VLA anchor: {anchor}")
        inputs = processor(images=images, language_instruction=anchor["task"])
        inputs = {
            key: value.to(device=device, dtype=torch.float32) if value.is_floating_point() else value.to(device)
            for key, value in inputs.items()
        }
        proprio = torch.from_numpy(proprio_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        action = torch.from_numpy(action_np).unsqueeze(0).to(device=device, dtype=torch.float32)

        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            encoded = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        synchronize(device)
        condition_seconds += time.perf_counter() - started

        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(
                args.noise_seed + sample_index * args.noise_samples + noise_index
            )
            noise = torch.randn(action.shape, generator=generator, device=device, dtype=action.dtype)
            target_velocity = noise - action
            for noise_level in args.noise_levels:
                timestep = torch.full((1,), noise_level, device=device, dtype=torch.float32)
                x_t = noise_level * noise + (1.0 - noise_level) * action
                proprio_model, x_t_model = model.action_space.preprocess(proprio, x_t)
                synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction_x0 = model.transformer(
                        domain_id=domain_id,
                        action_with_noise=x_t_model,
                        proprio=proprio_model,
                        t=timestep,
                        **encoded,
                    )
                synchronize(device)
                head_seconds += time.perf_counter() - started
                implied_velocity = (x_t - prediction_x0) / noise_level
                records.append(
                    {
                        **anchor,
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "model_time": noise_level,
                        "continuous_action": tensor_metrics(
                            implied_velocity.index_select(-1, active), target_velocity.index_select(-1, active)
                        ),
                        "target_prediction": tensor_metrics(
                            prediction_x0.index_select(-1, active), action.index_select(-1, active)
                        ),
                    }
                )

        del inputs, encoded, proprio, action, noise, target_velocity, x_t, prediction_x0, implied_velocity
        if device.type == "cuda":
            torch.cuda.empty_cache()

    algebra_identity_max_relative_error = max(
        abs(record["continuous_action"]["mse"] - record["target_prediction"]["mse"] / record["noise_level"] ** 2)
        / max(record["continuous_action"]["mse"], 1e-30)
        for record in records
    )
    algebra_identity_threshold = 1e-4
    if algebra_identity_max_relative_error > algebra_identity_threshold:
        raise RuntimeError(
            "X-VLA target-to-velocity algebra audit failed: "
            f"{algebra_identity_max_relative_error:.3e} > {algebra_identity_threshold:.3e}"
        )

    result = {
        "schema_version": "openpi.external_xvla_velocity.v1",
        "checkpoint": {
            "directory": str(args.checkpoint_dir),
            "file": str(checkpoint_file),
            "repo": args.checkpoint_repo,
            "sha256": actual_sha256,
            "framework": "X-VLA clean-action-prediction flow model",
            "source_revision": git_revision(args.xvla_root),
            "strict_load": True,
        },
        "dataset": {
            "kind": args.dataset_kind,
            "root": str(args.dataset_root),
            "provenance": dataset_provenance(args.dataset_root, args.dataset_kind),
            "source_relation": (
                "official Bridge LeRobot source and X-VLA Bridge conversion"
                if args.dataset_kind == "bridge"
                else (
                    "official Fractal/RT1 LeRobot source and X-VLA RT1 conversion"
                    if args.dataset_kind == "rt1"
                    else "CALVIN D-domain episodes; held out from the ABC-domain fine-tuning split"
                )
            ),
            "timestamp_grid_audit": timing_audit,
            "representation_audit": (
                "Official X-VLA single-arm D20 absolute xyz + rotation-6D + gripper conversion; inactive right "
                "arm is zero and excluded from the primary D9 continuous metric."
            ),
            "temporal_alignment": (
                (
                    f"CALVIN rows are interpreted at {args.calvin_model_fps:g} Hz; the official X-VLA "
                    "CalvinHandler uses 30 Hz while the selective LeRobot wrapper declares 10 Hz."
                )
                if args.dataset_kind == "calvin"
                else "Dataset timestamps match the official X-VLA handler frequency."
            ),
        },
        "protocol": {
            "sample_count": len(selected),
            "selection": (
                "task-balanced deterministic official-X-VLA-valid anchors"
                if args.dataset_kind in {"bridge", "rt1"}
                else "deterministic one anchor per locally available CALVIN D episode"
            ),
            "selection_seed": args.selection_seed,
            "noise_seed": args.noise_seed,
            "noise_samples_per_anchor": args.noise_samples,
            "noise_levels": args.noise_levels,
            "fixed_noise_per_anchor_across_times": True,
            "horizon": 30,
            "trajectory_seconds": {"bridge": 5.0, "rt1": 10.0, "calvin": 1.0}[args.dataset_kind],
            "model_action_dim": 20,
            "reported_continuous_action_dim": 9,
            "reported_indices": list(ACTIVE_CONTINUOUS_INDICES),
            "condition": "task text plus official processor images plus absolute D20 proprio",
            "normalization": "none; exact X-VLA absolute xyz + rotation-6D + gripper representation",
            "time_convention": "X-VLA t=0 data, t=1 Gaussian noise; reported noise_level=t",
            "parameterization": "model predicts clean action; v_hat=(x_t-action_hat)/t",
            "algebra_identity_max_relative_error": algebra_identity_max_relative_error,
            "algebra_identity_fail_threshold": algebra_identity_threshold,
            "interpretation_warning": (
                "Implied v-MSE equals clean-action MSE divided by t^2. Native target-prediction MSE is reported "
                "alongside it and X-VLA should not be treated as independent native-velocity evidence."
            ),
        },
        "runtime": {
            "transformers_version": transformers.__version__,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "image_processor_class": type(processor.image_processor).__name__,
            "model_load_seconds": model_load_seconds,
            "condition_seconds": condition_seconds,
            "action_head_seconds": head_seconds,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
            ),
        },
        "aggregate": aggregate(records, args.noise_levels),
        "selected_anchors": selected,
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint": result["checkpoint"],
                "dataset": result["dataset"],
                "protocol": result["protocol"],
                "runtime": result["runtime"],
                "aggregate": result["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
