#!/usr/bin/env python3
"""Check whether the shared LIBERO-Plus LeRobot snapshot is ready for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _episode_paths(root: Path, episode_index: int) -> list[Path]:
    chunk = f"chunk-{episode_index // 1000:03d}"
    episode = f"episode_{episode_index:06d}"
    return [
        root / "data" / chunk / f"{episode}.parquet",
        root / "videos" / chunk / "observation.images.front" / f"{episode}.mp4",
        root / "videos" / chunk / "observation.images.wrist" / f"{episode}.mp4",
    ]


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern))


def _find_complete_episodes(root: Path, total_episodes: int, limit: int) -> list[int]:
    complete: list[int] = []
    for episode_index in range(total_episodes):
        if all(path.is_file() for path in _episode_paths(root, episode_index)):
            complete.append(episode_index)
            if len(complete) >= limit:
                break
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/inspire/hdd/global_user/gongjingjing-25039/ytchen/data/libero_plus_lerobot",
    )
    parser.add_argument("--repo-id", default="Sylvest/libero_plus_lerobot")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--subset-episodes", type=int, default=2)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"[missing] {info_path}")
        return 2

    info = json.loads(info_path.read_text())
    total_episodes = int(info["total_episodes"])
    expected_videos = int(info.get("total_videos", total_episodes * 2))
    parquet_count = _count_files(root, "data/**/*.parquet")
    video_count = _count_files(root, "videos/**/*.mp4")
    print(f"[counts] parquet={parquet_count}/{total_episodes} videos={video_count}/{expected_videos}")

    missing_examples = []
    for episode_index in range(total_episodes):
        missing = [str(path.relative_to(root)) for path in _episode_paths(root, episode_index) if not path.is_file()]
        if missing:
            missing_examples.append((episode_index, missing))
            if len(missing_examples) >= 5:
                break
    if missing_examples:
        print("[incomplete] first missing episode files:")
        for episode_index, missing in missing_examples:
            print(f"  episode {episode_index}: {missing}")
    else:
        print("[complete] all expected parquet/front/wrist files are present")

    if args.skip_decode:
        return 0 if not missing_examples else 1

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    complete_episodes = _find_complete_episodes(root, total_episodes, args.subset_episodes)
    print(f"[subset] complete_episodes={complete_episodes}")
    if len(complete_episodes) < args.subset_episodes:
        print("[error] not enough complete episodes for subset decode smoke")
        return 2

    metadata = LeRobotDatasetMetadata(args.repo_id, root=root)
    delta_timestamps = {"action": [t / metadata.fps for t in range(args.action_horizon)]}
    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        episodes=complete_episodes,
        delta_timestamps=delta_timestamps,
    )
    item = dataset[0]
    print(f"[decode] len={len(dataset)} fps={metadata.fps}")
    for key in ("observation.images.front", "observation.images.wrist", "observation.state", "action", "task"):
        value = item[key]
        print(f"[decode] {key}: type={type(value).__name__} shape={getattr(value, 'shape', None)}")

    if missing_examples:
        print("[warn] subset decode works, but full training should wait for the remaining files")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
