#!/usr/bin/env python3
"""Estimate LIBERO action spread under a lattice of task, state, and image conditions.

This is a model-free diagnostic. Held-out episodes (episode_index % split_modulus
== holdout_remainder) are used only as query anchors. Neighbors come from the
remaining episodes of the same task, with at most one neighbor per episode.

The primary comparison is a task-only action mean versus cross-episode kNN under
task + proprio. With cached visual features, the optional condition lattice also
measures no-wrist, no-state, full, and globally matched no-prompt information
sets. An oracle progress coordinate is a diagnostic only; it is not a policy
input and must not be reported as such.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

SUITES = {
    **dict.fromkeys(range(10), "libero_10"),
    **dict.fromkeys(range(10, 20), "libero_goal"),
    **dict.fromkeys(range(20, 30), "libero_object"),
    **dict.fromkeys(range(30, 40), "libero_spatial"),
}

LATTICE_EDGES = {
    "state_given_task": ("task_state", "task_only"),
    "base_given_task_state": ("task_state_base", "task_state"),
    "wrist_given_task_state_base": ("task_state_base_wrist", "task_state_base"),
    "both_images_given_task_state": ("task_state_base_wrist", "task_state"),
    "state_given_task_images": ("task_state_base_wrist", "task_base_wrist"),
    "task_given_state_images": ("task_state_base_wrist", "state_base_wrist"),
    "both_images_given_task": ("task_base_wrist", "task_only"),
}


@dataclasses.dataclass
class Episode:
    episode_index: int
    task_index: int
    states: np.ndarray
    action_chunks: np.ndarray
    progress: np.ndarray
    visual_features: np.ndarray | None = None


@dataclasses.dataclass
class GlobalCandidatePool:
    state_features: np.ndarray
    visual_features: np.ndarray
    action_chunks: np.ndarray
    episode_indices: np.ndarray
    task_indices: np.ndarray
    state_tree: cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ubuntu/.cache/huggingface/lerobot/physical-intelligence/libero"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results/libero_conditional_spread"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--anchor-validity-horizon",
        type=int,
        default=None,
        help="Require this many remaining actions for every anchor; useful for matched horizon sweeps.",
    )
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--max-anchors-per-task", type=int, default=500)
    parser.add_argument("--cluster-anchors-per-task", type=int, default=100)
    parser.add_argument("--split-modulus", type=int, default=10)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--max-train-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-heldout-episodes-per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=None, help="Smoke-test limit; omit for the full dataset.")
    parser.add_argument(
        "--norm-stats-path",
        "--norm-stats-json",
        dest="norm_stats_path",
        type=Path,
        default=Path(
            "/tmp/openpi_checkpoints/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4/"
            "libero_holdout_h10_a4_mask7_50k/49999/assets/physical-intelligence/libero/norm_stats.json"
        ),
        help="OpenPI norm_stats.json or a LeRobot normalizer .safetensors file.",
    )
    parser.add_argument("--norm-stats-source", choices=("checkpoint", "train_partition"), default="checkpoint")
    parser.add_argument(
        "--extra-delta-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match this paper checkpoint by subtracting current state[:6] from each raw action target.",
    )
    parser.add_argument("--visual-embedding-root", type=Path, default=None)
    parser.add_argument("--visual-weights", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--visual-state-shortlist", type=int, default=16)
    parser.add_argument("--condition-lattice", action="store_true")
    parser.add_argument("--lattice-visual-weight", type=float, default=0.5)
    parser.add_argument("--visual-only-shortlist", type=int, default=32)
    parser.add_argument("--visual-global-dims", type=int, default=32)
    parser.add_argument("--cross-task-state-shortlist", type=int, default=2048)
    parser.add_argument("--cross-task-batch-size", type=int, default=32)
    return parser.parse_args()


def fixed_list_to_numpy(column: Any) -> np.ndarray:
    array = column.combine_chunks()
    width = array.type.list_size
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values).reshape(len(array), width)


def load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    with (dataset_root / "meta" / "tasks.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def load_episode_indices(dataset_root: Path, max_episodes: int | None) -> list[int]:
    indices = []
    with (dataset_root / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            indices.append(int(json.loads(line)["episode_index"]))
            if max_episodes is not None and len(indices) >= max_episodes:
                break
    return indices


def load_episodes(
    dataset_root: Path,
    episode_indices: list[int],
    horizon: int,
    *,
    anchor_validity_horizon: int,
    extra_delta_transform: bool,
    visual_embedding_root: Path | None,
) -> list[Episode]:
    episodes = []
    offsets = np.arange(horizon)[None, :]
    for position, episode_index in enumerate(episode_indices, start=1):
        path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(path, columns=["state", "actions", "task_index"])
        states = fixed_list_to_numpy(table["state"]).astype(np.float32, copy=False)
        actions = fixed_list_to_numpy(table["actions"]).astype(np.float32, copy=False)
        task_values = table["task_index"].combine_chunks().to_numpy(zero_copy_only=False)
        unique_tasks = np.unique(task_values)
        if len(unique_tasks) != 1:
            raise ValueError(f"Episode {episode_index} has multiple task indices: {unique_tasks.tolist()}")
        valid_length = len(actions) - max(horizon, anchor_validity_horizon) + 1
        if valid_length <= 0:
            continue
        chunk_indices = np.arange(valid_length)[:, None] + offsets
        action_chunks = actions[chunk_indices]
        if extra_delta_transform:
            action_chunks[..., :6] -= states[:valid_length, None, :6]
        denominator = max(len(actions) - 1, 1)
        progress = np.arange(valid_length, dtype=np.float32) / denominator
        visual_features = None
        if visual_embedding_root is not None:
            visual_path = visual_embedding_root / f"episode_{episode_index:06d}.npy"
            if not visual_path.exists():
                raise FileNotFoundError(f"Missing visual embeddings: {visual_path}")
            visual_features = np.asarray(np.load(visual_path, mmap_mode="r")[:valid_length], dtype=np.float32)
            if visual_features.ndim != 3 or visual_features.shape[:2] != (valid_length, 2):
                raise ValueError(f"Unexpected visual feature shape at {visual_path}: {visual_features.shape}")
        episodes.append(
            Episode(
                episode_index=episode_index,
                task_index=int(unique_tasks[0]),
                states=states[:valid_length],
                action_chunks=action_chunks,
                progress=progress,
                visual_features=visual_features,
            )
        )
        if position % 200 == 0 or position == len(episode_indices):
            print(f"[load] {position}/{len(episode_indices)} episodes", flush=True)
    return episodes


def load_checkpoint_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="np") as handle:
            state_mean = np.asarray(handle.get_tensor("observation.state.mean")[:8], dtype=np.float32)
            state_std = np.asarray(handle.get_tensor("observation.state.std")[:8], dtype=np.float32)
            action_mean = np.asarray(handle.get_tensor("action.mean")[:7], dtype=np.float32)
            action_std = np.asarray(handle.get_tensor("action.std")[:7], dtype=np.float32)
        if np.any(state_std <= 0) or np.any(action_std <= 0):
            raise ValueError(f"Non-positive physical-dimension std in {path}")
        return state_mean, state_std, action_mean, action_std

    payload = json.loads(path.read_text())["norm_stats"]
    state_mean = np.asarray(payload["state"]["mean"][:8], dtype=np.float32)
    state_std = np.asarray(payload["state"]["std"][:8], dtype=np.float32)
    action_mean = np.asarray(payload["actions"]["mean"][:7], dtype=np.float32)
    action_std = np.asarray(payload["actions"]["std"][:7], dtype=np.float32)
    if np.any(state_std <= 0) or np.any(action_std <= 0):
        raise ValueError(f"Non-positive physical-dimension std in {path}")
    return state_mean, state_std, action_mean, action_std


def safe_std(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    std = np.std(values, axis=axis, dtype=np.float64).astype(np.float32)
    return np.maximum(std, 1e-6)


def feature_matrix(episode: Episode, state_mean: np.ndarray, state_std: np.ndarray, condition: str) -> np.ndarray:
    state = (episode.states - state_mean) / state_std
    if condition == "task_state":
        return state
    if condition == "task_state_oracle_progress":
        # Uniform [0, 1] has standard deviation 1 / sqrt(12). The coordinate is
        # standardized so it has the same nominal scale as each state dimension.
        progress = ((episode.progress - 0.5) * math.sqrt(12.0))[:, None]
        return np.concatenate([state, progress], axis=1)
    raise ValueError(f"Unknown condition: {condition}")


def compute_visual_stats(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    first = episodes[0].visual_features
    if first is None:
        raise ValueError("Visual embeddings are not loaded.")
    feature_sum = np.zeros(first.shape[1:], dtype=np.float64)
    feature_square_sum = np.zeros(first.shape[1:], dtype=np.float64)
    count = 0
    for episode in episodes:
        if episode.visual_features is None:
            raise ValueError(f"Episode {episode.episode_index} is missing visual embeddings.")
        values = episode.visual_features.astype(np.float64)
        feature_sum += np.sum(values, axis=0)
        feature_square_sum += np.sum(np.square(values), axis=0)
        count += len(values)
    mean = feature_sum / count
    variance = np.maximum(feature_square_sum / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalized_visual_features(
    episode: Episode,
    mean: np.ndarray,
    std: np.ndarray,
    views: tuple[int, ...] = (0, 1),
    feature_end: int | None = None,
) -> np.ndarray:
    if episode.visual_features is None:
        raise ValueError("Visual embeddings are not loaded.")
    features = (episode.visual_features[:, views, :feature_end] - mean[None, views, :feature_end]) / std[
        None, views, :feature_end
    ]
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    return features.reshape(len(features), -1) / math.sqrt(features.shape[1])


def build_global_candidate_pool(
    episodes: list[Episode],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
) -> GlobalCandidatePool:
    state_features = np.concatenate(
        [feature_matrix(episode, state_mean, state_std, "task_state") for episode in episodes], axis=0
    )
    visual_features = np.concatenate(
        [normalized_visual_features(episode, visual_mean, visual_std) for episode in episodes], axis=0
    )
    action_chunks = np.concatenate([episode.action_chunks for episode in episodes], axis=0)
    episode_indices = np.concatenate(
        [np.full(len(episode.states), episode.episode_index, dtype=np.int32) for episode in episodes]
    )
    task_indices = np.concatenate(
        [np.full(len(episode.states), episode.task_index, dtype=np.int16) for episode in episodes]
    )
    return GlobalCandidatePool(
        state_features=state_features,
        visual_features=visual_features,
        action_chunks=action_chunks,
        episode_indices=episode_indices,
        task_indices=task_indices,
        state_tree=cKDTree(state_features),
    )


def select_anchors(episodes: list[Episode], maximum: int, rng: np.random.Generator) -> tuple[Episode, np.ndarray]:
    states = np.concatenate([episode.states for episode in episodes], axis=0)
    chunks = np.concatenate([episode.action_chunks for episode in episodes], axis=0)
    progress = np.concatenate([episode.progress for episode in episodes], axis=0)
    visual_features = None
    if episodes[0].visual_features is not None:
        visual_features = np.concatenate([episode.visual_features for episode in episodes], axis=0)
    episode_ids = np.concatenate(
        [np.full(len(episode.states), episode.episode_index, dtype=np.int32) for episode in episodes]
    )
    if len(states) > maximum:
        selected = np.sort(rng.choice(len(states), size=maximum, replace=False))
        states = states[selected]
        chunks = chunks[selected]
        progress = progress[selected]
        episode_ids = episode_ids[selected]
        if visual_features is not None:
            visual_features = visual_features[selected]
    # Use an Episode-shaped container so feature_matrix has one implementation.
    anchors = Episode(-1, episodes[0].task_index, states, chunks, progress, visual_features)
    return anchors, episode_ids


def limit_episode_subset(episodes: list[Episode], maximum: int | None, rng: np.random.Generator) -> list[Episode]:
    if maximum is None or len(episodes) <= maximum:
        return episodes
    selected = np.sort(rng.choice(len(episodes), size=maximum, replace=False))
    return [episodes[int(index)] for index in selected]


def group_mse(diff: np.ndarray) -> dict[str, float]:
    return {
        "chunk_all": float(np.mean(np.square(diff))),
        "chunk_continuous": float(np.mean(np.square(diff[..., :6]))),
        "chunk_gripper": float(np.mean(np.square(diff[..., 6:7]))),
        "first_all": float(np.mean(np.square(diff[..., 0, :]))),
        "first_continuous": float(np.mean(np.square(diff[..., 0, :6]))),
        "first_translation": float(np.mean(np.square(diff[..., 0, :3]))),
        "first_rotation": float(np.mean(np.square(diff[..., 0, 3:6]))),
        "first_gripper": float(np.mean(np.square(diff[..., 0, 6:7]))),
    }


def two_cluster_gain(samples: np.ndarray, iterations: int = 12) -> tuple[float, float]:
    """Return two-means SSE gain and smaller-cluster fraction for [K, F] samples."""
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    sse_one = float(np.sum(np.square(centered)))
    if sse_one <= 1e-12:
        return 0.0, 0.0
    pairwise = np.sum(np.square(samples[:, None, :] - samples[None, :, :]), axis=-1)
    first, second = np.unravel_index(int(np.argmax(pairwise)), pairwise.shape)
    centers = samples[[first, second]].copy()
    labels = np.zeros(len(samples), dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum(np.square(samples[:, None, :] - centers[None, :, :]), axis=-1)
        new_labels = np.argmin(distances, axis=1)
        if np.all(new_labels == new_labels[0]):
            return 0.0, 0.0
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        centers = np.stack([np.mean(samples[labels == cluster], axis=0) for cluster in range(2)])
    residual = samples - centers[labels]
    sse_two = float(np.sum(np.square(residual)))
    counts = np.bincount(labels, minlength=2)
    return max(0.0, 1.0 - sse_two / sse_one), float(np.min(counts) / len(samples))


def summarize_cluster_scores(scores: list[tuple[float, float]]) -> dict[str, float]:
    gains = np.asarray([score[0] for score in scores], dtype=np.float32)
    fractions = np.asarray([score[1] for score in scores], dtype=np.float32)
    balanced = fractions >= 0.25
    return {
        "anchors": len(scores),
        "gain_median": float(np.median(gains)),
        "gain_p90": float(np.quantile(gains, 0.9)),
        "balanced_gain_ge_0.5_fraction": float(np.mean(balanced & (gains >= 0.5))),
        "smaller_cluster_fraction_median": float(np.median(fractions)),
    }


def nearest_episode_neighbors(
    anchors: Episode,
    train_episodes: list[Episode],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    condition: str,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_features = feature_matrix(anchors, state_mean, state_std, condition)
    nearest_actions = []
    nearest_distances = []
    for episode in train_episodes:
        features = feature_matrix(episode, state_mean, state_std, condition)
        distances, indices = cKDTree(features).query(anchor_features, k=1, workers=1)
        nearest_distances.append(distances.astype(np.float32))
        nearest_actions.append(episode.action_chunks[indices])
    distances_by_episode = np.stack(nearest_distances, axis=1)
    actions_by_episode = np.stack(nearest_actions, axis=1)
    k = min(neighbors, len(train_episodes))
    selected = np.argpartition(distances_by_episode, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances_by_episode, selected, axis=1)
    order = np.argsort(selected_distances, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)
    selected_actions = actions_by_episode[np.arange(len(anchor_features))[:, None], selected]
    return selected_actions, selected_distances, actions_by_episode


def nearest_episode_neighbors_visual(
    anchors: Episode,
    train_episodes: list[Episode],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    visual_weight: float,
    state_shortlist: int,
    neighbors: int,
    visual_views: tuple[int, ...] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    anchor_state = feature_matrix(anchors, state_mean, state_std, "task_state")
    anchor_visual = normalized_visual_features(anchors, visual_mean, visual_std, visual_views)
    nearest_actions = []
    nearest_distances = []
    for episode in train_episodes:
        episode_state = feature_matrix(episode, state_mean, state_std, "task_state")
        episode_visual = normalized_visual_features(episode, visual_mean, visual_std, visual_views)
        shortlist = min(state_shortlist, len(episode_state))
        state_distances, indices = cKDTree(episode_state).query(anchor_state, k=shortlist, workers=1)
        if shortlist == 1:
            state_distances = state_distances[:, None]
            indices = indices[:, None]
        candidate_visual = episode_visual[indices]
        visual_distance_squared = np.sum(np.square(candidate_visual - anchor_visual[:, None, :]), axis=-1)
        combined_distance_squared = np.square(state_distances) / anchor_state.shape[1]
        combined_distance_squared += visual_weight**2 * visual_distance_squared
        best = np.argmin(combined_distance_squared, axis=1)
        best_indices = indices[np.arange(len(anchors.states)), best]
        nearest_actions.append(episode.action_chunks[best_indices])
        nearest_distances.append(
            np.sqrt(combined_distance_squared[np.arange(len(anchors.states)), best]).astype(np.float32)
        )
    distances_by_episode = np.stack(nearest_distances, axis=1)
    actions_by_episode = np.stack(nearest_actions, axis=1)
    k = min(neighbors, len(train_episodes))
    selected = np.argpartition(distances_by_episode, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances_by_episode, selected, axis=1)
    order = np.argsort(selected_distances, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)
    selected_actions = actions_by_episode[np.arange(len(anchor_state))[:, None], selected]
    return selected_actions, selected_distances


def nearest_episode_neighbors_visual_only(
    anchors: Episode,
    train_episodes: list[Episode],
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    visual_shortlist: int,
    visual_global_dims: int,
    neighbors: int,
    visual_views: tuple[int, ...] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    anchor_visual = normalized_visual_features(anchors, visual_mean, visual_std, visual_views)
    anchor_global = normalized_visual_features(
        anchors, visual_mean, visual_std, visual_views, feature_end=visual_global_dims
    )
    nearest_actions = []
    nearest_distances = []
    for episode in train_episodes:
        episode_visual = normalized_visual_features(episode, visual_mean, visual_std, visual_views)
        episode_global = normalized_visual_features(
            episode, visual_mean, visual_std, visual_views, feature_end=visual_global_dims
        )
        shortlist = min(visual_shortlist, len(episode_global))
        _, indices = cKDTree(episode_global).query(anchor_global, k=shortlist, workers=1)
        if shortlist == 1:
            indices = indices[:, None]
        candidate_visual = episode_visual[indices]
        distances_squared = np.sum(np.square(candidate_visual - anchor_visual[:, None, :]), axis=-1)
        best = np.argmin(distances_squared, axis=1)
        best_indices = indices[np.arange(len(anchors.states)), best]
        nearest_actions.append(episode.action_chunks[best_indices])
        nearest_distances.append(np.sqrt(distances_squared[np.arange(len(anchors.states)), best]).astype(np.float32))
    distances_by_episode = np.stack(nearest_distances, axis=1)
    actions_by_episode = np.stack(nearest_actions, axis=1)
    k = min(neighbors, len(train_episodes))
    selected = np.argpartition(distances_by_episode, kth=k - 1, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances_by_episode, selected, axis=1)
    order = np.argsort(selected_distances, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)
    selected_actions = actions_by_episode[np.arange(len(anchor_visual))[:, None], selected]
    return selected_actions, selected_distances


def nearest_global_episode_neighbors_state_visual(
    anchors: Episode,
    pool: GlobalCandidatePool,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    visual_weight: float,
    state_shortlist: int,
    batch_size: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_state = feature_matrix(anchors, state_mean, state_std, "task_state")
    anchor_visual = normalized_visual_features(anchors, visual_mean, visual_std)
    shortlist = min(state_shortlist, len(pool.state_features))
    selected_actions = []
    selected_distances = []
    selected_tasks = []
    for start in range(0, len(anchor_state), batch_size):
        stop = min(start + batch_size, len(anchor_state))
        state_distances, indices = pool.state_tree.query(anchor_state[start:stop], k=shortlist, workers=1)
        if shortlist == 1:
            state_distances = state_distances[:, None]
            indices = indices[:, None]
        candidate_visual = pool.visual_features[indices]
        visual_distance_squared = np.sum(np.square(candidate_visual - anchor_visual[start:stop, None, :]), axis=-1)
        combined_distance_squared = np.square(state_distances) / anchor_state.shape[1]
        combined_distance_squared += visual_weight**2 * visual_distance_squared
        orders = np.argsort(combined_distance_squared, axis=1)
        for row, order in enumerate(orders):
            seen_episodes: set[int] = set()
            chosen = []
            for position in order:
                candidate_index = int(indices[row, position])
                episode_index = int(pool.episode_indices[candidate_index])
                if episode_index in seen_episodes:
                    continue
                seen_episodes.add(episode_index)
                chosen.append((candidate_index, float(combined_distance_squared[row, position])))
                if len(chosen) == neighbors:
                    break
            if len(chosen) != neighbors:
                raise ValueError(
                    f"Global shortlist found only {len(chosen)} distinct episodes; "
                    f"increase --cross-task-state-shortlist above {state_shortlist}."
                )
            chosen_indices = np.asarray([item[0] for item in chosen], dtype=np.int64)
            selected_actions.append(pool.action_chunks[chosen_indices])
            selected_distances.append(np.sqrt([item[1] for item in chosen]))
            selected_tasks.append(pool.task_indices[chosen_indices])
    return (
        np.stack(selected_actions),
        np.asarray(selected_distances, dtype=np.float32),
        np.stack(selected_tasks),
    )


def random_task_neighborhoods(
    train_episodes: list[Episode], count: int, neighbors: int, rng: np.random.Generator
) -> np.ndarray:
    result = []
    k = min(neighbors, len(train_episodes))
    for _ in range(count):
        selected_episodes = rng.choice(len(train_episodes), size=k, replace=False)
        chunks = []
        for episode_position in selected_episodes:
            episode = train_episodes[int(episode_position)]
            chunks.append(episode.action_chunks[int(rng.integers(len(episode.action_chunks)))])
        result.append(np.stack(chunks))
    return np.stack(result)


def evaluate_condition(
    anchors: Episode,
    train_episodes: list[Episode],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    condition: str,
    neighbors: int,
    cluster_anchors: int,
    rng: np.random.Generator,
    *,
    visual_mean: np.ndarray | None = None,
    visual_std: np.ndarray | None = None,
    visual_weight: float | None = None,
    visual_state_shortlist: int = 16,
    visual_views: tuple[int, ...] = (0, 1),
    retrieval_mode: str = "state_visual",
    visual_only_shortlist: int = 32,
    visual_global_dims: int = 32,
    global_pool: GlobalCandidatePool | None = None,
    cross_task_state_shortlist: int = 2048,
    cross_task_batch_size: int = 32,
) -> dict[str, Any]:
    neighbor_task_indices = None
    if visual_weight is None:
        neighbor_actions, neighbor_distances, _ = nearest_episode_neighbors(
            anchors, train_episodes, state_mean, state_std, condition, neighbors
        )
    else:
        if visual_mean is None or visual_std is None:
            raise ValueError("Visual mean/std are required for a visual condition.")
        if retrieval_mode == "state_visual":
            neighbor_actions, neighbor_distances = nearest_episode_neighbors_visual(
                anchors,
                train_episodes,
                state_mean,
                state_std,
                visual_mean,
                visual_std,
                visual_weight,
                visual_state_shortlist,
                neighbors,
                visual_views,
            )
        elif retrieval_mode == "visual_only":
            neighbor_actions, neighbor_distances = nearest_episode_neighbors_visual_only(
                anchors,
                train_episodes,
                visual_mean,
                visual_std,
                visual_only_shortlist,
                visual_global_dims,
                neighbors,
                visual_views,
            )
        elif retrieval_mode == "global_state_visual":
            if global_pool is None:
                raise ValueError("global_pool is required for global_state_visual retrieval.")
            neighbor_actions, neighbor_distances, neighbor_task_indices = nearest_global_episode_neighbors_state_visual(
                anchors,
                global_pool,
                state_mean,
                state_std,
                visual_mean,
                visual_std,
                visual_weight,
                cross_task_state_shortlist,
                cross_task_batch_size,
                neighbors,
            )
        else:
            raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")
    normalized_targets = (anchors.action_chunks - action_mean) / action_std
    normalized_neighbors = (neighbor_actions - action_mean) / action_std
    prediction = np.mean(normalized_neighbors, axis=1)
    local_diff = normalized_neighbors - prediction[:, None]
    prediction_metrics = group_mse(normalized_targets - prediction)
    local_spread_metrics = group_mse(local_diff)
    if normalized_neighbors.shape[1] > 1:
        correction = normalized_neighbors.shape[1] / (normalized_neighbors.shape[1] - 1)
        local_spread_metrics = {key: value * correction for key, value in local_spread_metrics.items()}

    raw_prediction = np.mean(neighbor_actions, axis=1)
    raw_diff = anchors.action_chunks - raw_prediction
    support_distance = np.sqrt(np.min(np.mean(np.square(local_diff[..., :6]), axis=(2, 3)), axis=1))
    local_rms = np.sqrt(np.mean(np.square(local_diff[..., :6]), axis=(1, 2, 3)))
    support_ratio = support_distance / np.maximum(local_rms, 1e-8)

    cluster_count = min(cluster_anchors, len(anchors.states))
    cluster_indices = np.sort(rng.choice(len(anchors.states), size=cluster_count, replace=False))
    cluster_scores = [
        two_cluster_gain(normalized_neighbors[index, :, :, :6].reshape(normalized_neighbors.shape[1], -1))
        for index in cluster_indices
    ]
    random_neighborhoods = random_task_neighborhoods(train_episodes, cluster_count, neighbors, rng)
    normalized_random = (random_neighborhoods - action_mean) / action_std
    random_cluster_scores = [
        two_cluster_gain(normalized_random[index, :, :, :6].reshape(normalized_random.shape[1], -1))
        for index in range(cluster_count)
    ]

    neighbor_task_diagnostics = None
    if neighbor_task_indices is not None:
        unique_tasks_per_anchor = np.asarray(
            [len(np.unique(task_indices)) for task_indices in neighbor_task_indices], dtype=np.int32
        )
        neighbor_task_diagnostics = {
            "same_as_anchor_task_fraction": float(np.mean(neighbor_task_indices == anchors.task_index)),
            "unique_neighbor_tasks": len(np.unique(neighbor_task_indices)),
            "unique_tasks_per_anchor_mean": float(np.mean(unique_tasks_per_anchor)),
            "unique_tasks_per_anchor_median": float(np.median(unique_tasks_per_anchor)),
        }

    return {
        "condition": condition,
        "retrieval_mode": retrieval_mode if visual_weight is not None else "state_only",
        "visual_weight": visual_weight,
        "visual_state_shortlist": visual_state_shortlist if visual_weight is not None else None,
        "visual_views": list(visual_views) if visual_weight is not None else None,
        "neighbor_task_diagnostics": neighbor_task_diagnostics,
        "neighbors": int(normalized_neighbors.shape[1]),
        "normalized_knn_prediction_mse": prediction_metrics,
        "normalized_local_action_variance": local_spread_metrics,
        "target_space_knn_prediction_rmse": {
            "chunk_continuous": float(np.sqrt(np.mean(np.square(raw_diff[..., :6])))),
            "first_translation": float(np.sqrt(np.mean(np.square(raw_diff[..., 0, :3])))),
            "first_rotation": float(np.sqrt(np.mean(np.square(raw_diff[..., 0, 3:6])))),
            "first_gripper": float(np.sqrt(np.mean(np.square(raw_diff[..., 0, 6:7])))),
        },
        "condition_distance": {
            "nearest_median": float(np.median(neighbor_distances[:, 0])),
            "nearest_p90": float(np.quantile(neighbor_distances[:, 0], 0.9)),
            "kth_median": float(np.median(neighbor_distances[:, -1])),
            "kth_p90": float(np.quantile(neighbor_distances[:, -1], 0.9)),
        },
        "neighbor_mean_support": {
            "continuous_support_distance_median": float(np.median(support_distance)),
            "continuous_support_ratio_median": float(np.median(support_ratio)),
            "continuous_support_ratio_p90": float(np.quantile(support_ratio, 0.9)),
        },
        "two_cluster_screen_continuous_chunk": {
            "conditioned_neighborhood": summarize_cluster_scores(cluster_scores),
            "task_only_random_neighborhood": summarize_cluster_scores(random_cluster_scores),
        },
    }


def task_only_metrics(
    anchors: Episode, train_episodes: list[Episode], action_mean: np.ndarray, action_std: np.ndarray
) -> dict[str, Any]:
    train_chunks = np.concatenate([episode.action_chunks for episode in train_episodes], axis=0)
    normalized_train = (train_chunks - action_mean) / action_std
    normalized_targets = (anchors.action_chunks - action_mean) / action_std
    task_mean = np.mean(normalized_train, axis=0)
    return {
        "normalized_task_mean_prediction_mse": group_mse(normalized_targets - task_mean),
        "normalized_task_action_variance": group_mse(normalized_train - task_mean),
    }


def metric_ratios(task_only: dict[str, Any], condition: dict[str, Any]) -> dict[str, float]:
    baseline = task_only["normalized_task_mean_prediction_mse"]
    prediction = condition["normalized_knn_prediction_mse"]
    unconditional_variance = task_only["normalized_task_action_variance"]
    local_variance = condition["normalized_local_action_variance"]
    return {f"prediction_mse_ratio_{key}": prediction[key] / max(baseline[key], 1e-12) for key in baseline} | {
        f"local_variance_ratio_{key}": local_variance[key] / max(unconditional_variance[key], 1e-12)
        for key in unconditional_variance
    }


def aggregate_rows(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    output = {}
    for group, items in grouped.items():
        condition_names = list(items[0]["conditions"])
        ratios_by_condition = {
            condition_name: {
                key: float(np.mean([item["conditions"][condition_name]["ratios"][key] for item in items]))
                for key in items[0]["conditions"][condition_name]["ratios"]
            }
            for condition_name in condition_names
        }
        absolute_by_condition = {
            condition_name: {
                metric_family: {
                    key: float(np.mean([item["conditions"][condition_name][metric_family][key] for item in items]))
                    for key in items[0]["conditions"][condition_name][metric_family]
                }
                for metric_family in ("normalized_knn_prediction_mse", "normalized_local_action_variance")
            }
            for condition_name in condition_names
        }
        task_only_absolute = {
            metric_family: {
                key: float(np.mean([item["task_only"][metric_family][key] for item in items]))
                for key in items[0]["task_only"][metric_family]
            }
            for metric_family in ("normalized_task_mean_prediction_mse", "normalized_task_action_variance")
        }
        lattice_edge_ratios = {}
        for edge_name, (richer, weaker) in LATTICE_EDGES.items():
            if richer not in condition_names or (weaker != "task_only" and weaker not in condition_names):
                continue
            edge_metrics = {}
            for output_prefix, condition_family, task_only_family in (
                ("prediction_mse_ratio", "normalized_knn_prediction_mse", "normalized_task_mean_prediction_mse"),
                (
                    "local_variance_ratio",
                    "normalized_local_action_variance",
                    "normalized_task_action_variance",
                ),
            ):
                for key in items[0]["conditions"][richer][condition_family]:
                    task_ratios = []
                    for item in items:
                        numerator = item["conditions"][richer][condition_family][key]
                        if weaker == "task_only":
                            denominator = item["task_only"][task_only_family][key]
                        else:
                            denominator = item["conditions"][weaker][condition_family][key]
                        task_ratios.append(numerator / max(denominator, 1e-12))
                    edge_metrics[f"{output_prefix}_{key}"] = float(np.mean(task_ratios))
            lattice_edge_ratios[edge_name] = {
                "richer_condition": richer,
                "weaker_condition": weaker,
                "macro_mean_task_ratio": edge_metrics,
            }
        neighbor_task_diagnostics = {}
        for condition_name in condition_names:
            diagnostics = [
                item["conditions"][condition_name]["neighbor_task_diagnostics"]
                for item in items
                if item["conditions"][condition_name]["neighbor_task_diagnostics"] is not None
            ]
            if diagnostics:
                neighbor_task_diagnostics[condition_name] = {
                    key: float(np.mean([diagnostic[key] for diagnostic in diagnostics])) for key in diagnostics[0]
                }
        output[group] = {
            "tasks": len(items),
            "anchors": int(sum(item["anchors"] for item in items)),
            "macro_mean_ratios_by_condition": ratios_by_condition,
            "macro_mean_absolute_by_condition": absolute_by_condition,
            "macro_mean_task_only_absolute": task_only_absolute,
            "macro_mean_lattice_edge_ratios": lattice_edge_ratios,
            "macro_mean_neighbor_task_diagnostics": neighbor_task_diagnostics,
            "macro_mean_ratios": ratios_by_condition["task_state"],
            "macro_mean_oracle_progress_ratios": ratios_by_condition["task_state_oracle_progress"],
        }
    return output


def main() -> None:
    args = parse_args()
    anchor_validity_horizon = args.anchor_validity_horizon or args.horizon
    positive_values = {
        "horizon": args.horizon,
        "anchor-validity-horizon": anchor_validity_horizon,
        "neighbors": args.neighbors,
        "visual-state-shortlist": args.visual_state_shortlist,
        "visual-only-shortlist": args.visual_only_shortlist,
        "visual-global-dims": args.visual_global_dims,
        "cross-task-state-shortlist": args.cross_task_state_shortlist,
        "cross-task-batch-size": args.cross_task_batch_size,
    }
    if args.max_train_episodes_per_task is not None:
        positive_values["max-train-episodes-per-task"] = args.max_train_episodes_per_task
    if args.max_heldout_episodes_per_task is not None:
        positive_values["max-heldout-episodes-per-task"] = args.max_heldout_episodes_per_task
    if invalid := [name for name, value in positive_values.items() if value <= 0]:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if any(weight <= 0 for weight in args.visual_weights):
        raise ValueError("visual-weights must be positive")
    if args.lattice_visual_weight <= 0:
        raise ValueError("lattice-visual-weight must be positive")
    if args.condition_lattice and args.visual_embedding_root is None:
        raise ValueError("--condition-lattice requires --visual-embedding-root")
    tasks = load_tasks(args.dataset_root)
    episode_indices = load_episode_indices(args.dataset_root, args.max_episodes)
    episodes = load_episodes(
        args.dataset_root,
        episode_indices,
        args.horizon,
        anchor_validity_horizon=anchor_validity_horizon,
        extra_delta_transform=args.extra_delta_transform,
        visual_embedding_root=args.visual_embedding_root,
    )
    train_episodes = [
        episode for episode in episodes if episode.episode_index % args.split_modulus != args.holdout_remainder
    ]
    heldout_episodes = [
        episode for episode in episodes if episode.episode_index % args.split_modulus == args.holdout_remainder
    ]
    if not train_episodes or not heldout_episodes:
        raise ValueError("The selected episodes do not contain both train and held-out partitions.")

    if args.norm_stats_source == "checkpoint":
        state_mean, state_std, action_mean_vector, action_std_vector = load_checkpoint_norm_stats(args.norm_stats_path)
        normalization_description = "checkpoint state/action per-dimension mean/std, first 8/7 physical dimensions"
    else:
        all_train_states = np.concatenate([episode.states for episode in train_episodes], axis=0)
        all_train_actions = np.concatenate(
            [episode.action_chunks.reshape(-1, episode.action_chunks.shape[-1]) for episode in train_episodes], axis=0
        )
        state_mean = np.mean(all_train_states, axis=0, dtype=np.float64).astype(np.float32)
        state_std = safe_std(all_train_states, axis=0)
        action_mean_vector = np.mean(all_train_actions, axis=0, dtype=np.float64).astype(np.float32)
        action_std_vector = safe_std(all_train_actions, axis=0)
        normalization_description = "mean/std recomputed from episode_index training partition"
    action_mean = action_mean_vector[None, :]
    action_std = action_std_vector[None, :]
    visual_mean = visual_std = None
    if args.visual_embedding_root is not None:
        visual_mean, visual_std = compute_visual_stats(train_episodes)
    global_pool = None
    if args.condition_lattice:
        if visual_mean is None or visual_std is None:
            raise ValueError("Visual statistics are required for the condition lattice.")
        print("[global-pool] building cross-task state and visual candidate pool", flush=True)
        global_pool = build_global_candidate_pool(
            train_episodes,
            state_mean,
            state_std,
            visual_mean,
            visual_std,
        )
        print(f"[global-pool] frames={len(global_pool.state_features)}", flush=True)

    train_by_task: dict[int, list[Episode]] = defaultdict(list)
    heldout_by_task: dict[int, list[Episode]] = defaultdict(list)
    for episode in train_episodes:
        train_by_task[episode.task_index].append(episode)
    for episode in heldout_episodes:
        heldout_by_task[episode.task_index].append(episode)

    rows = []
    common_tasks = sorted(set(train_by_task) & set(heldout_by_task))
    for task_position, task_index in enumerate(common_tasks, start=1):
        train_episode_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_index, 901]))
        heldout_episode_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_index, 902]))
        task_train = limit_episode_subset(
            train_by_task[task_index], args.max_train_episodes_per_task, train_episode_rng
        )
        task_heldout = limit_episode_subset(
            heldout_by_task[task_index], args.max_heldout_episodes_per_task, heldout_episode_rng
        )
        if len(task_train) < 2:
            print(f"[skip] task={task_index}: only {len(task_train)} training episodes", flush=True)
            continue
        anchor_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_index, 0]))
        anchors, anchor_episode_ids = select_anchors(task_heldout, args.max_anchors_per_task, anchor_rng)
        baseline = task_only_metrics(anchors, task_train, action_mean, action_std)
        conditions = {}
        for condition_position, condition_name in enumerate(("task_state", "task_state_oracle_progress"), start=1):
            condition_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_index, condition_position]))
            metrics = evaluate_condition(
                anchors,
                task_train,
                state_mean,
                state_std,
                action_mean,
                action_std,
                condition_name,
                args.neighbors,
                args.cluster_anchors_per_task,
                condition_rng,
            )
            metrics["ratios"] = metric_ratios(baseline, metrics)
            conditions[condition_name] = metrics
        if args.condition_lattice:
            lattice_specs = (
                ("task_state_base", task_train, "state_visual", (0,), 101),
                ("task_state_base_wrist", task_train, "state_visual", (0, 1), 102),
                ("task_base_wrist", task_train, "visual_only", (0, 1), 103),
                ("state_base_wrist", train_episodes, "global_state_visual", (0, 1), 104),
            )
            for condition_name, candidates, retrieval_mode, visual_views, condition_seed in lattice_specs:
                condition_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_index, condition_seed]))
                metrics = evaluate_condition(
                    anchors,
                    candidates,
                    state_mean,
                    state_std,
                    action_mean,
                    action_std,
                    condition_name,
                    args.neighbors,
                    args.cluster_anchors_per_task,
                    condition_rng,
                    visual_mean=visual_mean,
                    visual_std=visual_std,
                    visual_weight=args.lattice_visual_weight,
                    visual_state_shortlist=args.visual_state_shortlist,
                    visual_views=visual_views,
                    retrieval_mode=retrieval_mode,
                    visual_only_shortlist=args.visual_only_shortlist,
                    visual_global_dims=args.visual_global_dims,
                    global_pool=global_pool,
                    cross_task_state_shortlist=args.cross_task_state_shortlist,
                    cross_task_batch_size=args.cross_task_batch_size,
                )
                metrics["ratios"] = metric_ratios(baseline, metrics)
                conditions[condition_name] = metrics
        elif args.visual_embedding_root is not None:
            for visual_position, visual_weight in enumerate(args.visual_weights, start=1):
                condition_name = f"task_state_tiny_siglip_w{visual_weight:g}".replace(".", "p")
                condition_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, task_index, 100 + visual_position])
                )
                metrics = evaluate_condition(
                    anchors,
                    task_train,
                    state_mean,
                    state_std,
                    action_mean,
                    action_std,
                    condition_name,
                    args.neighbors,
                    args.cluster_anchors_per_task,
                    condition_rng,
                    visual_mean=visual_mean,
                    visual_std=visual_std,
                    visual_weight=visual_weight,
                    visual_state_shortlist=args.visual_state_shortlist,
                )
                metrics["ratios"] = metric_ratios(baseline, metrics)
                conditions[condition_name] = metrics
        row = {
            "task_index": task_index,
            "task": tasks.get(task_index, ""),
            "suite": SUITES.get(task_index, "unknown"),
            "train_episodes": len(task_train),
            "heldout_episodes": len(task_heldout),
            "anchors": len(anchors.states),
            "anchor_episode_ids": sorted(np.unique(anchor_episode_ids).astype(int).tolist()),
            "task_only": baseline,
            "conditions": conditions,
        }
        rows.append(row)
        ratio = conditions["task_state"]["ratios"]["prediction_mse_ratio_chunk_continuous"]
        spread = conditions["task_state"]["ratios"]["local_variance_ratio_chunk_continuous"]
        print(
            f"[task] {task_position}/{len(common_tasks)} index={task_index:02d} "
            f"anchors={len(anchors.states)} train_eps={len(task_train)} "
            f"prediction_ratio={ratio:.3f} local_spread_ratio={spread:.3f}",
            flush=True,
        )

    if not rows:
        raise ValueError("No tasks had both train and held-out data.")
    result = {
        "schema_version": "openpi.libero_conditional_spread.v3",
        "protocol": {
            "dataset_root": str(args.dataset_root),
            "horizon": args.horizon,
            "neighbors": args.neighbors,
            "neighbor_constraint": (
                "same task except state_base_wrist/no_prompt; one nearest frame per distinct training episode"
                if args.condition_lattice
                else "same task, one nearest frame per distinct training episode"
            ),
            "split": (
                f"held-out iff episode_index % {args.split_modulus} == {args.holdout_remainder}; "
                "all other episodes are neighbor candidates"
            ),
            "anchor_validity_horizon": anchor_validity_horizon,
            "terminal_frames": f"last {anchor_validity_horizon - 1} frames excluded; no padded chunks",
            "max_anchors_per_task": args.max_anchors_per_task,
            "cluster_anchors_per_task": args.cluster_anchors_per_task,
            "max_train_episodes_per_task": args.max_train_episodes_per_task,
            "max_heldout_episodes_per_task": args.max_heldout_episodes_per_task,
            "normalization": normalization_description,
            "norm_stats_source": args.norm_stats_source,
            "norm_stats_path": str(args.norm_stats_path) if args.norm_stats_source == "checkpoint" else None,
            "target_transform": (
                "actions[..., :6] -= current_state[:6]" if args.extra_delta_transform else "raw parquet actions"
            ),
            "local_variance_estimator": (
                "sample variance across one matched frame per distinct training episode, Bessel corrected"
            ),
            "oracle_progress_warning": "task_state_oracle_progress includes normalized episode progress unavailable to policy",
            "visual_embedding_root": (
                str(args.visual_embedding_root) if args.visual_embedding_root is not None else None
            ),
            "visual_matching": (
                {
                    "weights": ([args.lattice_visual_weight] if args.condition_lattice else list(args.visual_weights)),
                    "state_shortlist_per_episode": args.visual_state_shortlist,
                    "feature_normalization": (
                        "training-split z-score per view/dimension, then L2 per view; two views equally weighted"
                    ),
                    "combined_distance": (
                        "mean squared checkpoint-normalized state distance + weight^2 * visual L2 distance"
                    ),
                }
                if args.visual_embedding_root is not None
                else None
            ),
            "condition_lattice": (
                {
                    "task_only": "T; exact task-specific action mean/variance baseline",
                    "task_state": "T+S; same-task state matching; offline analog of no_image",
                    "task_state_base": "T+S+B; same-task state+base-image matching; no_wrist",
                    "task_state_base_wrist": "T+S+B+W; same-task state+both-image matching; full",
                    "task_base_wrist": "T+B+W; same-task visual-only matching; no_state",
                    "state_base_wrist": (
                        "S+B+W; global cross-task state+both-image matching with no task restriction; no_prompt"
                    ),
                    "visual_only_shortlist_per_episode": args.visual_only_shortlist,
                    "visual_global_prefix_dimensions_per_view": args.visual_global_dims,
                    "cross_task_state_shortlist": args.cross_task_state_shortlist,
                    "cross_task_batch_size": args.cross_task_batch_size,
                    "no_prompt_leakage_check": (
                        "reports selected-neighbor same-task fraction and task diversity without using task in retrieval"
                    ),
                }
                if args.condition_lattice
                else None
            ),
            "seed": args.seed,
        },
        "coverage": {
            "episodes_loaded": len(episodes),
            "train_episodes": len(train_episodes),
            "heldout_episodes": len(heldout_episodes),
            "tasks": len(rows),
            "anchors": int(sum(row["anchors"] for row in rows)),
        },
        "normalization_parameters": {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "action_mean": action_mean_vector.tolist(),
            "action_std": action_std_vector.tolist(),
        },
        "suite_aggregates": aggregate_rows(rows, "suite"),
        "all_task_aggregate": aggregate_rows([{**row, "all": "all"} for row in rows], "all")["all"],
        "tasks": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.max_episodes is None else f"first{args.max_episodes}episodes"
    if anchor_validity_horizon != args.horizon:
        suffix += f"_anchors_valid_h{anchor_validity_horizon}"
    if args.max_train_episodes_per_task is not None:
        suffix += f"_train_eps{args.max_train_episodes_per_task}"
    if args.max_heldout_episodes_per_task is not None:
        suffix += f"_heldout_eps{args.max_heldout_episodes_per_task}"
    if args.visual_embedding_root is not None:
        suffix += f"_tiny_siglip_visual_ss{args.visual_state_shortlist}"
    if args.condition_lattice:
        suffix += (
            f"_condition_lattice_vos{args.visual_only_shortlist}"
            f"_vg{args.visual_global_dims}_xss{args.cross_task_state_shortlist}"
        )
    output_path = args.output_dir / f"conditional_spread_h{args.horizon}_k{args.neighbors}_{suffix}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
