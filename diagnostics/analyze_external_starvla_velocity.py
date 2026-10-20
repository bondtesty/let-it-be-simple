#!/usr/bin/env python3
"""Probe a public StarVLA GR00T-head checkpoint at fixed flow times.

The probe follows StarVLA's native linear path exactly::

    x_t = (1 - t) * noise + t * action
    velocity = action - noise

StarVLA model time therefore runs from pure Gaussian noise (t=0) to data
(t=1).  Results are reported on the common ``noise_level = 1 - t`` axis used
by the other external VLA probes in this repository.

The supported public checkpoints use a Qwen2.5-VL, Qwen3-VL, or Florence-2
condition encoder with StarVLA's shared GR00T flow head.  Every run audits its
input and normalization contract against checkpoint and dataset metadata.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import time

import av
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F  # noqa: N812

DEFAULT_NOISE_LEVELS = (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-config",
        type=Path,
        help="Explicit released JSON/YAML config when multiple checkpoint metadata files disagree.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind", choices=("libero", "vla_arena", "bridge", "calvin", "robocasa_gr1"), default="libero"
    )
    parser.add_argument("--stats-key", help="Auto-resolved when dataset_statistics.json contains one key.")
    parser.add_argument(
        "--normalization-reference",
        type=Path,
        help=(
            "Optional audited dataset_statistics.json used when a mixture checkpoint's descriptive mean/std are "
            "weighted but its actual q01/q99 normalization should match a single-dataset checkpoint."
        ),
    )
    parser.add_argument("--normalization-reference-key", help="Defaults to --stats-key.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--base-vlm",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base config/processor path; full base weights are not required.",
    )
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help="Qwen attention backend. Eager is the conservative FlashAttention-free reproduction path.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        help="Defaults to the lexicographically last .pt/.safetensors under checkpoint-dir/checkpoints.",
    )
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
        path = args.checkpoint_file
    else:
        candidates = sorted((args.checkpoint_dir / "checkpoints").glob("*.pt"))
        candidates += sorted((args.checkpoint_dir / "checkpoints").glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint under {args.checkpoint_dir / 'checkpoints'}")
        path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_checkpoint_config(checkpoint_dir: Path, config_file: Path | None = None) -> dict:
    from omegaconf import OmegaConf

    if config_file is not None:
        path = config_file if config_file.is_absolute() else checkpoint_dir / config_file
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    json_path = checkpoint_dir / "config.json"
    if json_path.is_file():
        return json.loads(json_path.read_text())
    for name in ("config.full.yaml", "config.yaml"):
        path = checkpoint_dir / name
        if path.is_file():
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raise FileNotFoundError(f"No config.json/config.full.yaml/config.yaml under {checkpoint_dir}")


def select_anchors(
    dataset_root: Path,
    *,
    dataset_kind: str,
    horizon: int,
    max_samples: int,
    samples_per_task: int,
    seed: int,
) -> list[dict]:
    """Select deterministic valid-horizon anchors balanced over tasks."""
    if dataset_kind == "robocasa_gr1":
        rng = np.random.default_rng(seed)
        selected = []
        for task_index, task_dir in enumerate(sorted(dataset_root.glob("gr1_unified.*"))):
            episode = read_jsonl(task_dir / "meta" / "episodes.jsonl")[0]
            episode_length = int(episode["length"])
            if episode_length < horizon:
                continue
            valid_count = episode_length - horizon + 1
            low = min(10, valid_count - 1)
            high = max(low + 1, valid_count - 10)
            frame_index = int(rng.integers(low, high)) if high > low else low
            table = pd.read_parquet(
                task_dir / "data" / "chunk-000" / "episode_000000.parquet",
                columns=["annotation.human.coarse_action"],
            )
            annotation_index = int(table.iloc[frame_index]["annotation.human.coarse_action"])
            tasks = {int(row["task_index"]): row["task"] for row in read_jsonl(task_dir / "meta" / "tasks.jsonl")}
            task = tasks[annotation_index]
            if not task:
                raise RuntimeError(f"Empty coarse-action annotation at {task_dir.name}:{frame_index}")
            selected.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "task_directory": task_dir.name,
                    "annotation_index": annotation_index,
                    "episode_index": 0,
                    "episode_length": episode_length,
                    "frame_index": frame_index,
                    "parquet": f"{task_dir.name}/data/chunk-000/episode_000000.parquet",
                    "video": (f"{task_dir.name}/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"),
                }
            )
            if len(selected) == max_samples:
                return selected
        return selected

    if dataset_kind == "bridge":
        manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
        rng = np.random.default_rng(seed)
        selected = []
        for task_index, task_entry in enumerate(manifest["tasks"]):
            candidates = [episode for episode in task_entry["episodes"] if int(episode["length"]) >= horizon]
            rng.shuffle(candidates)
            for episode in candidates[:samples_per_task]:
                valid_count = int(episode["length"]) - horizon + 1
                low = min(2, valid_count - 1)
                high = max(low + 1, valid_count - 2)
                frame_index = int(rng.integers(low, high)) if high > low else low
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
                    return selected
        return selected

    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = {row["task"]: int(row["task_index"]) for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")}
    if dataset_kind == "calvin":
        # The public probe is an intentionally selective download: retain only
        # episodes whose parquet payload is present locally instead of sampling
        # from the complete metadata index and failing later in load_anchor.
        available_episode_indices = {
            int(path.stem.removeprefix("episode_")) for path in dataset_root.glob("data/*/episode_*.parquet")
        }
        episodes = [episode for episode in episodes if int(episode["episode_index"]) in available_episode_indices]
        rng = np.random.default_rng(seed)
        selected = []
        for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
            valid_count = int(episode["length"]) - horizon + 1
            low = min(10, valid_count - 1)
            high = max(low + 1, valid_count - 10)
            frame_index = int(rng.integers(low, high)) if high > low else low
            task = episode["tasks"][0]
            selected.append(
                {
                    "task": task,
                    "task_index": tasks[task],
                    "episode_index": int(episode["episode_index"]),
                    "episode_length": int(episode["length"]),
                    "frame_index": frame_index,
                }
            )
            if len(selected) == max_samples:
                return selected
        return selected

    by_task: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        if int(episode["length"]) >= horizon:
            by_task[episode["tasks"][0]].append(episode)

    missing_tasks = sorted(set(tasks) - set(by_task))
    if missing_tasks:
        raise RuntimeError(f"No H{horizon} episode for tasks: {missing_tasks}")

    rng = np.random.default_rng(seed)
    selected = []
    for task in sorted(tasks, key=tasks.get):
        candidates = by_task[task].copy()
        rng.shuffle(candidates)
        for episode in candidates[:samples_per_task]:
            valid_count = int(episode["length"]) - horizon + 1
            # Avoid both the reset transient and the final padding boundary.
            low = min(10, valid_count - 1)
            high = max(low + 1, valid_count - 10)
            frame_index = int(rng.integers(low, high)) if high > low else low
            selected.append(
                {
                    "task": task,
                    "task_index": tasks[task],
                    "episode_index": int(episode["episode_index"]),
                    "episode_length": int(episode["length"]),
                    "frame_index": frame_index,
                }
            )
            if len(selected) == max_samples:
                return selected
    return selected


def episode_path(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def decode_image(cell: dict) -> Image.Image:
    if not isinstance(cell, dict) or not cell.get("bytes"):
        raise TypeError(f"Expected inline LeRobot image bytes, got {type(cell)}")
    # StarVLA's training dataloader calls PIL.resize((224, 224)) without an
    # explicit resampling mode.  Match that exact behavior here.
    return Image.open(BytesIO(cell["bytes"])).convert("RGB").resize((224, 224))


def decode_video_frame(path: Path, frame_index: int) -> Image.Image:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24")).resize((224, 224))
    raise RuntimeError(f"Could not decode frame {frame_index} from {path}")


def load_anchor(
    dataset_root: Path,
    anchor: dict,
    action_stats: dict,
    *,
    dataset_kind: str,
    horizon: int,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray | None]:
    if dataset_kind == "robocasa_gr1":
        table = pd.read_parquet(
            dataset_root / anchor["parquet"],
            columns=["observation.state", "action", "annotation.human.coarse_action"],
        )
        frame_index = anchor["frame_index"]
        if int(table.iloc[frame_index]["annotation.human.coarse_action"]) != anchor["annotation_index"]:
            raise RuntimeError("RoboCasa annotation/parquet mismatch")
        images = [decode_video_frame(dataset_root / anchor["video"], frame_index)]
        raw_action_44 = np.stack(table.iloc[frame_index : frame_index + horizon]["action"].to_numpy()).astype(
            np.float32
        )
        raw_state_44 = np.asarray(table.iloc[frame_index]["observation.state"], dtype=np.float32)
        slices = (slice(0, 7), slice(22, 29), slice(7, 13), slice(29, 35), slice(41, 44))
        raw = np.concatenate([raw_action_44[:, action_slice] for action_slice in slices], axis=-1)
        state_groups = [raw_state_44[state_slice] for state_slice in slices]
        state = np.concatenate(
            [component for group in state_groups for component in (np.sin(group), np.cos(group))], axis=-1
        ).astype(np.float32)
        if raw.shape != (horizon, 29) or state.shape != (58,):
            raise RuntimeError(f"Expected H{horizon}x29 action and D58 state, got {raw.shape} and {state.shape}")
        low = np.asarray(action_stats["min"], dtype=np.float32)
        high = np.asarray(action_stats["max"], dtype=np.float32)
        nonconstant = low != high
        normalized = np.zeros_like(raw)
        normalized[:, nonconstant] = (
            2.0 * (raw[:, nonconstant] - low[nonconstant]) / (high[nonconstant] - low[nonconstant]) - 1.0
        )
        return images, normalized, state

    if dataset_kind in {"libero", "vla_arena", "calvin"}:
        columns = ["image", "wrist_image", "frame_index", "task_index"]
        columns.append("rel_actions" if dataset_kind == "calvin" else "actions")
        table = pd.read_parquet(
            episode_path(dataset_root, anchor["episode_index"]),
            columns=columns,
        )
        action_column = "rel_actions" if dataset_kind == "calvin" else "actions"
    else:
        table = pd.read_parquet(dataset_root / anchor["parquet"], columns=["action", "frame_index"])
        action_column = "action"
    frame_index = anchor["frame_index"]
    if int(table.iloc[frame_index]["frame_index"]) != frame_index:
        raise RuntimeError("Parquet row/frame index mismatch")
    if (
        dataset_kind in {"libero", "vla_arena", "calvin"}
        and int(table.iloc[frame_index]["task_index"]) != anchor["task_index"]
    ):
        raise RuntimeError("Task metadata/parquet mismatch")

    if dataset_kind in {"libero", "vla_arena", "calvin"}:
        images = [
            decode_image(table.iloc[frame_index]["image"]),
            decode_image(table.iloc[frame_index]["wrist_image"]),
        ]
    else:
        images = [decode_video_frame(dataset_root / anchor["video"], frame_index)]
    raw = np.stack(table.iloc[frame_index : frame_index + horizon][action_column].to_numpy()).astype(np.float32)
    if raw.shape != (horizon, 7):
        raise RuntimeError(f"Expected H{horizon}x7 action chunk, got {raw.shape}")

    if dataset_kind == "libero":
        # The public physical-intelligence/libero data stores gripper in
        # {-1,+1}; the checkpoint's source stores the same signal in {1,0}.
        raw[:, 6] = (1.0 - raw[:, 6]) / 2.0
        low_key, high_key = "min", "max"
    elif dataset_kind == "vla_arena":
        # VLA-Arena's released LeRobot actions already use the StarVLA
        # checkpoint's {-1,+1} gripper convention.  Its rotation dimensions
        # are normalized with the released min/max contract.
        low_key, high_key = "min", "max"
    elif dataset_kind == "bridge":
        # StarVLA Bridge trains all six continuous dimensions with q99 and the
        # gripper with the registry's binary transform.
        raw[:, 6] = (raw[:, 6] > 0.5).astype(np.float32)
        low_key, high_key = "q01", "q99"
    else:
        # The public 10% CALVIN D->D probe stores relative gripper actions in
        # the opposite sign convention from the checkpoint statistics.
        raw[:, 6] *= -1.0
        low_key, high_key = "q01", "q99"
    low = np.asarray(action_stats[low_key], dtype=np.float32)
    high = np.asarray(action_stats[high_key], dtype=np.float32)
    mask = np.asarray(action_stats["mask"], dtype=bool)
    normalized = raw.copy()
    normalized[:, mask] = 2.0 * (raw[:, mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return images, normalized, None


def load_model(args: argparse.Namespace, checkpoint_file: Path, device: torch.device):
    """Construct a supported StarVLA flow model from its complete checkpoint.

    Released StarVLA files contain the full VLM and action-head state dict.  We
    therefore need the base model's config and processor, but not a redundant
    multi-gigabyte copy of its weights.
    """
    import sys

    sys.path.insert(0, str(args.starvla_root))
    from accelerate import init_empty_weights
    from omegaconf import OmegaConf
    from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
    import starVLA.model.modules.action_model.GR00T_ActionHeader as groot_head_module
    import starVLA.model.modules.action_model.LayerwiseFM_ActionHeader as layerwise_head_module
    from transformers import AutoConfig

    config = OmegaConf.create(read_checkpoint_config(args.checkpoint_dir, args.checkpoint_config))
    framework = str(config.framework.name)
    if framework not in {"QwenGR00T", "QwenPI"}:
        raise ValueError(f"This probe expects QwenGR00T or QwenPI, got {framework}")
    if framework == "QwenPI" and "florence" in args.base_vlm.lower():
        raise ValueError("QwenPI requires a layer-wise Qwen condition encoder")
    config.framework.qwenvl.base_vlm = args.base_vlm
    config.framework.qwenvl.attn_implementation = args.attention_implementation
    config.trainer.pretrained_checkpoint = None

    base_config_probe = AutoConfig.from_pretrained(args.base_vlm, trust_remote_code=True)
    base_config_class = type(base_config_probe).__name__
    is_florence = "Florence2" in base_config_class
    is_qwen3 = "Qwen3VL" in base_config_class
    is_qwen25 = "Qwen2_5_VL" in base_config_class
    if is_florence:
        from transformers import AutoModelForCausalLM

        original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__
        original_beta = groot_head_module.Beta

        @classmethod
        def empty_auto_from_pretrained(cls, model_id, *unused_args, **unused_kwargs):
            base_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return cls.from_config(base_config, trust_remote_code=True, attn_implementation="eager")

        def placeholder_beta(*unused_args, **unused_kwargs):
            return None

        AutoModelForCausalLM.from_pretrained = empty_auto_from_pretrained
        groot_head_module.Beta = placeholder_beta
        try:
            with init_empty_weights(include_buffers=False):
                model = Qwen_GR00T(config)
        finally:
            AutoModelForCausalLM.from_pretrained = classmethod(original_from_pretrained)
            groot_head_module.Beta = original_beta
        model.action_model.beta_dist = original_beta(
            config.framework.action_model.noise_beta_alpha,
            config.framework.action_model.noise_beta_beta,
        )
    else:
        if is_qwen3:
            from transformers import Qwen3VLForConditionalGeneration as QwenVLClass
        elif is_qwen25:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLClass
        else:
            raise ValueError(f"Unsupported base VLM config for meta construction: {base_config_class}")

        original_from_pretrained = QwenVLClass.from_pretrained.__func__
        original_beta = groot_head_module.Beta
        original_layerwise_beta = layerwise_head_module.Beta

        @classmethod
        def empty_from_pretrained(cls, model_id, *unused_args, **unused_kwargs):
            base_config = AutoConfig.from_pretrained(model_id)
            if is_qwen25:
                # StarVLA's public Bridge checkpoint uses its Action tokenizer,
                # whose added robot-action tokens extend the text embedding.
                # The upstream Qwen config retains the base vocab size, so use
                # the tokenizer length exactly as official from_pretrained does.
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(model_id)
                base_config.text_config.vocab_size = max(
                    int(base_config.text_config.vocab_size), len(processor.tokenizer)
                )
            import os
            if "STARVLA_BASE_VOCAB_SIZE" in os.environ:
                base_config.text_config.vocab_size = int(os.environ["STARVLA_BASE_VOCAB_SIZE"])
            base_config._attn_implementation = args.attention_implementation  # noqa: SLF001
            return cls(base_config)

        def placeholder_beta(*unused_args, **unused_kwargs):
            # Beta validates values with Tensor.item(), which meta tensors do
            # not support. The fixed-time probe never samples this distribution.
            return None

        QwenVLClass.from_pretrained = empty_from_pretrained
        groot_head_module.Beta = placeholder_beta
        layerwise_head_module.Beta = placeholder_beta
        try:
            # Keep tiny non-persistent buffers (for example rotary inv_freq) on
            # CPU: Hugging Face intentionally omits them from saved state dicts.
            with init_empty_weights(include_buffers=False):
                model = Qwen_GR00T(config) if framework == "QwenGR00T" else Qwen_PI(config)
        finally:
            QwenVLClass.from_pretrained = classmethod(original_from_pretrained)
            groot_head_module.Beta = original_beta
            layerwise_head_module.Beta = original_layerwise_beta
        beta_class = original_beta if framework == "QwenGR00T" else original_layerwise_beta
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
        raise RuntimeError(f"Checkpoint did not materialize all model tensors: {meta_tensors[:20]}")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def encode_condition(
    model,
    images: list[Image.Image],
    task: str,
) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=[images], instructions=[task])
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
    if str(model.config.framework.name) == "QwenPI":
        expected_layers = len(model.action_model.model.transformer_blocks)
        condition = list(output.hidden_states[-expected_layers:])
        condition_length = condition[-1].shape[1]
    else:
        condition = output.hidden_states[-1]
        condition_length = condition.shape[1]
    if attention_mask is not None and attention_mask.shape[1] != condition_length:
        # Florence expands image tokens inside its encoder but returns the
        # processor's pre-merge text mask. Batch-one probes contain no padding,
        # so an all-valid mask is exactly equivalent and avoids a length mismatch.
        attention_mask = None
    return condition, attention_mask


def fixed_velocity_prediction(
    head: torch.nn.Module,
    condition: torch.Tensor | list[torch.Tensor],
    attention_mask: torch.Tensor | None,
    actions: torch.Tensor,
    noise: torch.Tensor,
    model_time: float,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the checkpoint's native action-head path at explicit noise/time."""
    batch_size = actions.shape[0]
    t = torch.full((batch_size,), model_time, device=actions.device, dtype=actions.dtype)
    noisy_trajectory = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    target_velocity = actions - noise
    time_bucket = (t * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy_trajectory, time_bucket)
    if head.config.add_pos_embed:
        position_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
        action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
    future_tokens = head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
    state_features = head.state_encoder(state) if state is not None else None
    state_action_features = (
        torch.cat((state_features, future_tokens, action_features), dim=1)
        if state_features is not None
        else torch.cat((future_tokens, action_features), dim=1)
    )
    model_kwargs = {
        "hidden_states": state_action_features,
        "encoder_hidden_states": condition,
        "encoder_attention_mask": attention_mask,
        "timestep": time_bucket,
    }
    if isinstance(condition, list):
        temb = head.model.timestep_encoder(time_bucket)
        model_output = state_action_features
        for layer_idx, block in enumerate(head.model.transformer_blocks):
            model_output = block(
                model_output,
                attention_mask=None,
                encoder_hidden_states=condition[layer_idx],
                encoder_attention_mask=attention_mask,
                temb=temb,
            )
    else:
        model_output = head.model(**model_kwargs, return_all_hidden_states=False)
    prediction = head.action_decoder(model_output)[:, -actions.shape[1] :]
    return prediction, target_velocity


def metric_rows(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, dict[str, float]]:
    groups = {"all_action": (prediction, target)}
    if prediction.shape[-1] == 7:
        groups["continuous_action"] = (prediction[..., :6], target[..., :6])
        groups["gripper"] = (prediction[..., 6:], target[..., 6:])
    else:
        groups["continuous_action"] = (prediction, target)
    result = {}
    for name, (raw_prediction, raw_target) in groups.items():
        prediction_values = raw_prediction.float()
        target_values = raw_target.float()
        mse = (prediction_values - target_values).square().mean()
        cosine_error = 1.0 - F.cosine_similarity(prediction_values.flatten(), target_values.flatten(), dim=0)
        result[name] = {
            "mse": float(mse.cpu()),
            "cosine_error": float(cosine_error.cpu()),
            "prediction_norm": float(prediction_values.norm().cpu()),
            "target_norm": float(target_values.norm().cpu()),
        }
    return result


def aggregate(records: list[dict], noise_levels: list[float]) -> list[dict]:
    rows = []
    for noise_level in noise_levels:
        selected = [record for record in records if math.isclose(record["noise_level"], noise_level)]
        row = {"noise_level": noise_level, "model_time": 1.0 - noise_level}
        metric_groups = [key for key in records[0] if key in {"all_action", "continuous_action", "gripper"}]
        for group in metric_groups:
            values = np.asarray([record[group]["mse"] for record in selected], dtype=np.float64)
            row[group] = {
                "mse_mean": float(values.mean()),
                "mse_se": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.samples_per_task <= 0 or args.noise_samples <= 0:
        raise ValueError("sample counts must be positive")
    if any(level <= 0 or level > 1 for level in args.noise_levels):
        raise ValueError("noise levels must lie in (0, 1]")
    if not args.starvla_root.is_dir():
        raise FileNotFoundError(args.starvla_root)

    checkpoint_file = resolve_checkpoint(args)
    checkpoint_config = read_checkpoint_config(args.checkpoint_dir, args.checkpoint_config)
    action_config = checkpoint_config["framework"]["action_model"]
    horizon = int(action_config["action_horizon"])
    action_dim = int(action_config["action_dim"])
    expected_action_dim = 29 if args.dataset_kind == "robocasa_gr1" else 7
    if action_dim != expected_action_dim:
        raise ValueError(f"Expected a {expected_action_dim}D action for {args.dataset_kind}, got {action_dim}")
    metadata = json.loads((args.checkpoint_dir / "dataset_statistics.json").read_text())
    stats_key = args.stats_key
    if stats_key is None:
        if len(metadata) != 1:
            raise ValueError(f"Pass --stats-key from {list(metadata)}")
        stats_key = next(iter(metadata))
    if stats_key not in metadata:
        raise ValueError(f"stats-key {stats_key!r} not in {list(metadata)}")
    stats = metadata[stats_key]
    action_stats = stats["action"]
    expected_mask = [True] * 29 if args.dataset_kind == "robocasa_gr1" else [True] * 6 + [False]
    if action_stats["mask"] != expected_mask:
        raise ValueError(f"Unexpected action normalization mask: {action_stats['mask']}")

    selected = select_anchors(
        args.dataset_root,
        dataset_kind=args.dataset_kind,
        horizon=horizon,
        max_samples=args.max_samples,
        samples_per_task=args.samples_per_task,
        seed=args.selection_seed,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = load_model(args, checkpoint_file, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    records = []
    condition_seconds = 0.0
    head_seconds = 0.0
    for sample_index, anchor in enumerate(selected):
        images, action_values, state_values = load_anchor(
            args.dataset_root,
            anchor,
            action_stats,
            dataset_kind=args.dataset_kind,
            horizon=horizon,
        )
        synchronize(device)
        started = time.perf_counter()
        condition, attention_mask = encode_condition(model, images, anchor["task"])
        synchronize(device)
        condition_seconds += time.perf_counter() - started
        actions = torch.from_numpy(action_values).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        state = (
            torch.from_numpy(state_values).reshape(1, 1, -1).to(device=device, dtype=torch.bfloat16)
            if state_values is not None
            else None
        )

        for noise_index in range(args.noise_samples):
            generator = torch.Generator(device=device).manual_seed(
                args.noise_seed + sample_index * args.noise_samples + noise_index
            )
            noise = torch.randn(actions.shape, generator=generator, device=device, dtype=actions.dtype)
            for noise_level in args.noise_levels:
                model_time = 1.0 - noise_level
                synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                    prediction, target = fixed_velocity_prediction(
                        model.action_model,
                        condition,
                        attention_mask,
                        actions,
                        noise,
                        model_time,
                        state,
                    )
                synchronize(device)
                head_seconds += time.perf_counter() - started
                records.append(
                    {
                        **anchor,
                        "noise_index": noise_index,
                        "noise_level": noise_level,
                        "model_time": model_time,
                        **metric_rows(prediction, target),
                    }
                )

        # QwenPI retains one condition tensor per decoder layer.  Sequence
        # lengths vary across task prompts, so the CUDA caching allocator can
        # otherwise accumulate several differently sized attention workspaces
        # over a task-balanced sweep and exhaust a 32 GiB card even though a
        # single anchor peaks well below that limit.
        del condition, attention_mask, actions, state, noise, prediction, target
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.dataset_kind == "libero":
        dataset_summary = {
            "kind": "libero",
            "root": str(args.dataset_root),
            "episodes": 1693,
            "frames": 273465,
            "checkpoint_stats_transitions": int(stats["num_transitions"]),
            "checkpoint_stats_trajectories": int(stats["num_trajectories"]),
            "transition_count_difference": 273465 - int(stats["num_transitions"]),
            "same_source_evidence": (
                "Trajectory count and first-six action extrema match exactly; gripper extrema/mean match after "
                "mapping local {-1,+1} to checkpoint {1,0}. The 1361-transition count difference is not explained "
                "by all-zero-action removal and remains an explicit audit limitation."
            ),
            "condition_contract_evidence": (
                "StarVLA's LIBERO registry lists both video.primary_image and video.wrist_image; its official "
                "LIBERO evaluator also passes the primary and wrist views."
            ),
        }
        condition_description = "two RGB cameras resized to 224x224 plus task text; no proprioceptive state"
        normalization_description = (
            "first six action dims checkpoint min-max to [-1,1]; gripper mapped to checkpoint 0/1 and unscaled"
        )
    elif args.dataset_kind == "vla_arena":
        info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
        local_stats_rows = [
            json.loads(line)
            for line in (args.dataset_root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
            if line.strip()
        ]
        count = 0
        first_moment = np.zeros(7, dtype=np.float64)
        second_moment = np.zeros(7, dtype=np.float64)
        global_min = np.full(7, np.inf, dtype=np.float64)
        global_max = np.full(7, -np.inf, dtype=np.float64)
        for row in local_stats_rows:
            local = row["stats"]["actions"]
            local_count = int(local["count"][0])
            local_mean = np.asarray(local["mean"], dtype=np.float64)
            local_std = np.asarray(local["std"], dtype=np.float64)
            count += local_count
            first_moment += local_count * local_mean
            second_moment += local_count * (local_std**2 + local_mean**2)
            global_min = np.minimum(global_min, local["min"])
            global_max = np.maximum(global_max, local["max"])
        local_mean = first_moment / count
        local_std = np.sqrt(np.maximum(second_moment / count - local_mean**2, 0.0))
        checkpoint_mean = np.asarray(action_stats["mean"], dtype=np.float64)
        checkpoint_std = np.asarray(action_stats["std"], dtype=np.float64)
        mean_max_abs = float(np.max(np.abs(local_mean - checkpoint_mean)))
        std_max_abs = float(np.max(np.abs(local_std - checkpoint_std)))
        min_max_abs = float(
            max(
                np.max(np.abs(global_min - np.asarray(action_stats["min"], dtype=np.float64))),
                np.max(np.abs(global_max - np.asarray(action_stats["max"], dtype=np.float64))),
            )
        )
        if count != int(stats["num_transitions"]) or int(info["total_episodes"]) != int(stats["num_trajectories"]):
            raise RuntimeError("VLA-Arena source counts do not match checkpoint statistics")
        if mean_max_abs > 1e-5 or std_max_abs > 1e-5 or min_max_abs > 1e-6:
            raise RuntimeError("VLA-Arena action statistics do not match the checkpoint contract")
        dataset_summary = {
            "kind": "vla_arena",
            "root": str(args.dataset_root),
            "full_source_episodes": int(info["total_episodes"]),
            "full_source_frames": int(info["total_frames"]),
            "full_source_tasks": int(info["total_tasks"]),
            "probe_episode_files": len(list(args.dataset_root.glob("data/chunk-*/episode_*.parquet"))),
            "checkpoint_stats_transitions": int(stats["num_transitions"]),
            "checkpoint_stats_trajectories": int(stats["num_trajectories"]),
            "action_mean_max_abs_difference": mean_max_abs,
            "action_std_max_abs_difference": std_max_abs,
            "action_extrema_max_abs_difference": min_max_abs,
            "same_source_evidence": (
                "The VLA-Arena L0-L LeRobot v2.1 metadata has 60 tasks, 3018 trajectories, and 361736 transitions. "
                "Aggregated per-episode action moments/extrema match the released StarVLA checkpoint statistics."
            ),
        }
        condition_description = "static and wrist inline RGB images resized to 224x224 plus task text; no state"
        normalization_description = "continuous action dims checkpoint min/max to [-1,1]; VLA-Arena gripper remains in the released {-1,+1} convention"
    elif args.dataset_kind == "bridge":
        info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
        manifest = json.loads((args.dataset_root / "probe_manifest.json").read_text())
        local_stats = json.loads((args.dataset_root / "meta" / "stats.json").read_text())["action"]
        extrema_match = np.allclose(local_stats["min"], action_stats["min"], rtol=0, atol=1e-8) and np.allclose(
            local_stats["max"], action_stats["max"], rtol=0, atol=1e-8
        )
        mean_match = np.allclose(local_stats["mean"], action_stats["mean"], rtol=0, atol=5e-5)
        normalization_reference_summary = None
        if not extrema_match:
            raise RuntimeError("Bridge source extrema do not match the checkpoint contract")
        if not mean_match:
            if args.normalization_reference is None:
                raise RuntimeError(
                    "Bridge descriptive means do not match; pass an audited --normalization-reference only when "
                    "the checkpoint's actual q01/q99 transform is known to match"
                )
            reference_payload = json.loads(args.normalization_reference.read_text())
            reference_key = args.normalization_reference_key or stats_key
            reference_stats = reference_payload[reference_key]["action"]
            reference_mean_match = np.allclose(local_stats["mean"], reference_stats["mean"], rtol=0, atol=5e-5)
            normalization_match = (
                all(
                    np.allclose(action_stats[key], reference_stats[key], rtol=0, atol=1e-8)
                    for key in ("min", "max", "q01", "q99")
                )
                and action_stats["mask"] == reference_stats["mask"]
            )
            if not reference_mean_match or not normalization_match:
                raise RuntimeError("Bridge checkpoint failed the explicit normalization-reference gate")
            normalization_reference_summary = {
                "path": str(args.normalization_reference),
                "stats_key": reference_key,
                "gate": "min/max/q01/q99/mask exact within 1e-8; reference mean matches local source within 5e-5",
                "mixture_descriptive_stats_note": (
                    "The mixture checkpoint's oxe_bridge mean/std are weighted and do not represent standalone "
                    "Bridge moments; they are not used by the q01/q99 action transform."
                ),
            }
        dataset_summary = {
            "kind": "bridge",
            "root": str(args.dataset_root),
            "full_source_episodes": int(info["total_episodes"]),
            "full_source_frames": int(info["total_frames"]),
            "probe_subset_tasks": int(manifest["selection"]["tasks"]),
            "probe_subset_episodes_per_task": int(manifest["selection"]["episodes_per_task"]),
            "source_revision": manifest["revision"],
            "checkpoint_stats_transitions": int(stats["num_transitions"]),
            "checkpoint_stats_trajectories": int(stats["num_trajectories"]),
            "normalization_reference": normalization_reference_summary,
            "same_source_evidence": (
                "IPEC-COMMUNITY/bridge_orig_lerobot trajectory count and all action extrema match exactly. For a "
                "single-dataset checkpoint, action means also match within 5e-5. A mixture checkpoint must instead "
                "pass the explicit q01/q99 normalization-reference gate. The selective probe contains 8 episodes "
                "for each of 40 tasks."
            ),
        }
        condition_description = "one RGB camera resized to 224x224 plus task text; no proprioceptive state"
        normalization_description = "first six action dims checkpoint q01/q99 to [-1,1]; gripper binary 0/1"
    elif args.dataset_kind == "calvin":
        info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
        local_stats = json.loads((args.dataset_root / "meta" / "stats.json").read_text())["rel_actions"]
        local_mean = np.asarray(local_stats["mean"], dtype=np.float64)
        local_std = np.asarray(local_stats["std"], dtype=np.float64)
        local_mean[6] *= -1.0
        checkpoint_mean = np.asarray(action_stats["mean"], dtype=np.float64)
        checkpoint_std = np.asarray(action_stats["std"], dtype=np.float64)
        continuous_mean_max_abs = float(np.max(np.abs(local_mean[:6] - checkpoint_mean[:6])))
        continuous_std_max_relative = float(np.max(np.abs(local_std[:6] / checkpoint_std[:6] - 1.0)))
        trajectory_scale = int(stats["num_trajectories"]) / int(info["total_episodes"])
        transition_scale = int(stats["num_transitions"]) / int(info["total_frames"])
        if continuous_mean_max_abs > 0.01 or continuous_std_max_relative > 0.1:
            raise RuntimeError("CALVIN 10% probe continuous-action statistics do not align with checkpoint source")
        if not (9.0 <= trajectory_scale <= 11.0 and 9.0 <= transition_scale <= 11.0):
            raise RuntimeError("CALVIN probe is not an approximately 10% source subset")
        dataset_summary = {
            "kind": "calvin_d_d",
            "root": str(args.dataset_root),
            "probe_source_episodes": int(info["total_episodes"]),
            "probe_source_frames": int(info["total_frames"]),
            "downloaded_episode_files": len(list((args.dataset_root / "data").glob("chunk-*/episode_*.parquet"))),
            "checkpoint_stats_transitions": int(stats["num_transitions"]),
            "checkpoint_stats_trajectories": int(stats["num_trajectories"]),
            "trajectory_count_scale": trajectory_scale,
            "transition_count_scale": transition_scale,
            "continuous_mean_max_abs_difference": continuous_mean_max_abs,
            "continuous_std_max_relative_difference": continuous_std_max_relative,
            "same_source_evidence": (
                "The checkpoint and public LeRobot probe are both CALVIN task D->D. Checkpoint counts are about "
                "10x the public 10% source subset, and continuous relative-action means/std pass explicit 0.01/10% "
                "alignment gates. This is a source-aligned subset, not an exact full-dataset reproduction."
            ),
            "condition_contract_evidence": (
                "The current official StarVLA CALVIN mixture routes through the two-camera Franka registry; the "
                "probe uses static and wrist inline images plus task text."
            ),
        }
        condition_description = "static and wrist RGB images resized to 224x224 plus task text; no proprioceptive state"
        normalization_description = (
            "first six relative-action dims checkpoint q01/q99 to [-1,1]; public gripper sign inverted and unscaled"
        )
    else:
        task_dirs = sorted(args.dataset_root.glob("gr1_unified.*"))
        if len(task_dirs) != 24:
            raise RuntimeError(f"Expected all 24 RoboCasa-GR1 task metadata directories, got {len(task_dirs)}")
        raw_indices = np.asarray([*range(7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44)])
        total_frames = 0
        total_episodes = 0
        weighted_first = np.zeros(29, dtype=np.float64)
        weighted_second = np.zeros(29, dtype=np.float64)
        global_min = np.full(29, np.inf, dtype=np.float64)
        global_max = np.full(29, -np.inf, dtype=np.float64)
        for task_dir in task_dirs:
            info = json.loads((task_dir / "meta" / "info.json").read_text())
            task_stats = json.loads((task_dir / "meta" / "stats.json").read_text())["action"]
            frames = int(info["total_frames"])
            task_mean = np.asarray(task_stats["mean"], dtype=np.float64)[raw_indices]
            task_std = np.asarray(task_stats["std"], dtype=np.float64)[raw_indices]
            task_min = np.asarray(task_stats["min"], dtype=np.float64)[raw_indices]
            task_max = np.asarray(task_stats["max"], dtype=np.float64)[raw_indices]
            total_frames += frames
            total_episodes += int(info["total_episodes"])
            # The released checkpoint stores the mixture's task-balanced
            # moments (equal weight per 1000-episode task directory), not a
            # frame-weighted moment over variable-duration trajectories.
            weighted_first += task_mean
            weighted_second += task_std**2 + task_mean**2
            global_min = np.minimum(global_min, task_min)
            global_max = np.maximum(global_max, task_max)
        source_mean = weighted_first / len(task_dirs)
        source_std = np.sqrt(np.maximum(weighted_second / len(task_dirs) - source_mean**2, 0.0))
        mean_max_abs = float(np.max(np.abs(source_mean - np.asarray(action_stats["mean"]))))
        std_max_abs = float(np.max(np.abs(source_std - np.asarray(action_stats["std"]))))
        extrema_max_abs = float(
            max(
                np.max(np.abs(global_min - np.asarray(action_stats["min"]))),
                np.max(np.abs(global_max - np.asarray(action_stats["max"]))),
            )
        )
        if total_episodes != int(stats["num_trajectories"]) or total_frames != int(stats["num_transitions"]):
            raise RuntimeError("RoboCasa-GR1 source counts do not match checkpoint statistics")
        if mean_max_abs > 1e-5 or std_max_abs > 1e-5 or extrema_max_abs > 1e-6:
            raise RuntimeError("RoboCasa-GR1 reordered action statistics do not match the checkpoint")
        dataset_summary = {
            "kind": "robocasa_gr1",
            "root": str(args.dataset_root),
            "official_tasks": len(task_dirs),
            "official_episodes": total_episodes,
            "official_frames": total_frames,
            "probe_episode_files": len(
                list(args.dataset_root.glob("gr1_unified.*/data/chunk-000/episode_000000.parquet"))
            ),
            "probe_video_files": len(
                list(args.dataset_root.glob("gr1_unified.*/videos/chunk-000/*/episode_000000.mp4"))
            ),
            "action_mean_max_abs_difference": mean_max_abs,
            "action_std_max_abs_difference": std_max_abs,
            "action_extrema_max_abs_difference": extrema_max_abs,
            "same_source_evidence": (
                "All 24 official gr1_unified task folders contribute 1000 episodes each. Their summed "
                "trajectory/frame counts and reordered 29D action statistics match the checkpoint. The probe "
                "downloads one episode and one ego-view video per task."
            ),
            "condition_contract_evidence": (
                "StarVLA's Fourier-GR1 registry uses ego_view, 29D proprioception expanded to 58D by per-group "
                "sin/cos features, and annotation.human.coarse_action."
            ),
        }
        condition_description = "one ego-view RGB image, 58D sin/cos state, and coarse-action text"
        normalization_description = "reordered 29D action uses checkpoint min/max to [-1,1]"

    result = {
        "schema_version": "openpi.external_starvla_velocity.v2",
        "checkpoint": {
            "directory": str(args.checkpoint_dir),
            "file": str(checkpoint_file),
            "config_file": str(args.checkpoint_config) if args.checkpoint_config is not None else "auto",
            "framework": checkpoint_config["framework"]["name"],
            "base_vlm": args.base_vlm,
            "source_revision": git_revision(args.starvla_root),
        },
        "dataset": dataset_summary,
        "protocol": {
            "sample_count": len(selected),
            "samples_per_task": None if args.dataset_kind == "calvin" else args.samples_per_task,
            "selection_seed": args.selection_seed,
            "selection": (
                f"one deterministic nonboundary valid-H{horizon} anchor per locally downloaded episode"
                if args.dataset_kind == "calvin"
                else (
                    f"task-balanced deterministic nonboundary valid-H{horizon} anchors; checkpoint training data, "
                    "not held out"
                )
            ),
            "noise_samples_per_anchor": args.noise_samples,
            "noise_seed": args.noise_seed,
            "noise_levels": args.noise_levels,
            "time_convention": "StarVLA t=0 pure Gaussian noise, t=1 data; reported noise_level=1-t",
            "fixed_noise_per_anchor_across_times": True,
            "horizon": horizon,
            "action_dim": action_dim,
            "stats_key": stats_key,
            "condition": condition_description,
            "normalization": normalization_description,
            "attention_implementation": (
                "eager (hardcoded by StarVLA Florence interface)"
                if "florence" in args.base_vlm.lower()
                else (
                    "sdpa (hardcoded by current StarVLA Qwen3 interface)"
                    if "qwen3-vl" in args.base_vlm.lower()
                    else args.attention_implementation
                )
            ),
        },
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "condition_seconds": condition_seconds,
            "action_head_seconds": head_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        },
        "selected_anchors": selected,
        "aggregate": aggregate(records, args.noise_levels),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
