from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import re
import time
import typing

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import wandb

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _base_task_suite_name(task_suite_name: str) -> str:
    for base_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if task_suite_name == base_name or task_suite_name.startswith(f"{base_name}_"):
            return base_name
    return task_suite_name


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    gpu_id: int = 0  # GPU device for offscreen rendering
    replan_steps: int = 5

    #################################################################################################################
    # Adaptive execution parameters
    #################################################################################################################
    adaptive: bool = False  # Enable variance-based adaptive execution
    n_samples: int = 5  # Number of action samples for variance estimation
    std_threshold: float = 0.1  # Std threshold (normalized action space) to trigger replan
    min_execute_steps: int = 1  # Minimum actions to execute before replanning
    max_execute_steps: int = 10  # Maximum actions before forced replan

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    n_history_steps: int = 1  # Number of state history steps to send (>1 for A2A/RAE models)
    task_ids: str | None = None  # Optional comma-separated task ids, useful for LIBERO-Plus subsets.
    max_tasks: int | None = None  # Optional cap on evaluated tasks, applied after task_ids.
    # Language source for the policy prompt. Standard LIBERO: keep False -> task.language
    # (the BDDL filename with underscores replaced by spaces), which matches the training
    # dataset text and the paper eval protocol. LIBERO-Pro (custom BDDL suites such as
    # *_lan perturbations): True -> use the BDDL ":language" field instead.
    use_bddl_language: bool = False
    save_video: bool = True  # Save rollout videos. Disable for high-throughput eval sharding.
    result_jsonl: str | None = None  # Optional per-episode result log for sharded eval aggregation.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize wandb (offline by default for no-internet compute nodes)
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project="openpi-libero-eval",
        name=os.environ.get("EVAL_NAME", None),
        config=dataclasses.asdict(args),
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    task_ids = _resolve_task_ids(num_tasks_in_suite, args.task_ids, args.max_tasks)
    logging.info(f"Evaluating {len(task_ids)} / {num_tasks_in_suite} tasks: {task_ids}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    base_task_suite_name = _base_task_suite_name(args.task_suite_name)
    if base_task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif base_task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif base_task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif base_task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif base_task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    result_jsonl_path = pathlib.Path(args.result_jsonl) if args.result_jsonl else None
    if result_jsonl_path is not None:
        result_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Create env once per task, reuse across all episodes (avoids repeated EGL context creation)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed, gpu_id=args.gpu_id, use_bddl_language=args.use_bddl_language
        )

        logging.info(f"\nTask: {task_description}")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Reset environment
            env.reset()
            action_plan = collections.deque()
            state_history = collections.deque(maxlen=args.n_history_steps)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            episode_start_time = time.monotonic()
            t = 0
            replay_images = []
            done = False
            episode_policy_calls = 0
            episode_policy_batch_sizes = []
            episode_policy_infer_ms = []
            episode_server_infer_ms = []
            episode_batch_wait_ms = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # Accumulate state history at every timestep (including wait period)
                    state_history.append(_get_state(obs))

                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Build state observation: 2D [N, state_dim] for history, 1D for single step
                        if args.n_history_steps > 1:
                            # Pad with earliest state if buffer not yet full
                            while len(state_history) < args.n_history_steps:
                                state_history.appendleft(state_history[0])
                            obs_state = np.stack(list(state_history))  # [N, state_dim]
                        else:
                            obs_state = state_history[-1]  # [state_dim]

                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": obs_state,
                            "prompt": str(task_description),
                        }

                        if args.adaptive:
                            # Multi-sample inference for variance-based adaptive execution
                            element["__n_samples__"] = args.n_samples
                            result = client.infer(element)
                            _record_policy_timing(
                                result,
                                episode_policy_batch_sizes,
                                episode_policy_infer_ms,
                                episode_server_infer_ms,
                                episode_batch_wait_ms,
                            )
                            episode_policy_calls += 1
                            action_chunk = result["actions"]  # [H, D] mean
                            action_std = result["actions_std"]  # [H, D] per-step std

                            # Determine how many steps to execute based on uncertainty
                            per_step_std = action_std.mean(axis=-1)  # [H] avg std across dims
                            n_execute = args.min_execute_steps
                            for t_idx in range(args.min_execute_steps, min(len(action_chunk), args.max_execute_steps)):
                                if per_step_std[t_idx] > args.std_threshold:
                                    break
                                n_execute = t_idx + 1
                            action_plan.extend(action_chunk[:n_execute])
                        else:
                            # Original fixed replan
                            result = client.infer(element)
                            _record_policy_timing(
                                result,
                                episode_policy_batch_sizes,
                                episode_policy_infer_ms,
                                episode_server_infer_ms,
                                episode_batch_wait_ms,
                            )
                            episode_policy_calls += 1
                            action_chunk = result["actions"]
                            assert len(action_chunk) >= args.replan_steps, (
                                f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            )
                            action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            if args.save_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if result_jsonl_path is not None:
                _append_jsonl(
                    result_jsonl_path,
                    {
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "elapsed_seconds": time.monotonic() - episode_start_time,
                        "success": bool(done),
                        "steps": int(t),
                        "task_description": task_description,
                        "policy_calls": episode_policy_calls,
                        "policy_batch_size_mean": _mean_or_none(episode_policy_batch_sizes),
                        "policy_batch_size_max": max(episode_policy_batch_sizes)
                        if episode_policy_batch_sizes
                        else None,
                        "policy_infer_ms_mean": _mean_or_none(episode_policy_infer_ms),
                        "server_infer_ms_mean": _mean_or_none(episode_server_infer_ms),
                        "batch_wait_ms_mean": _mean_or_none(episode_batch_wait_ms),
                    },
                )
            wandb.log(
                {
                    "episode": total_episodes,
                    "cumulative_success_rate": total_successes / total_episodes,
                    "cumulative_successes": total_successes,
                }
            )

        env.close()

        # Log per-task results
        task_success_rate = float(task_successes) / float(task_episodes)
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        wandb.log(
            {
                "task_id": task_id,
                f"task_success_rate/{task_description}": task_success_rate,
                "task_success_rate/mean": float(total_successes) / float(total_episodes),
            }
        )

    total_success_rate = float(total_successes) / float(total_episodes)
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    # Log final summary
    wandb.summary["total_success_rate"] = total_success_rate
    wandb.summary["total_episodes"] = total_episodes
    wandb.summary["total_successes"] = total_successes
    wandb.finish()


def _get_libero_env(task, resolution, seed, *, gpu_id: int = 0, use_bddl_language: bool = False):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_language = _read_bddl_language(task_bddl_file) if use_bddl_language else None
    task_description = bddl_language or task.language
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": gpu_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _read_bddl_language(task_bddl_file: pathlib.Path) -> str | None:
    """Read the task instruction from BDDL when perturbation suites rewrite it."""
    try:
        match = re.search(r"\(:language\s+([^\)]+)\)", task_bddl_file.read_text())
    except OSError:
        return None
    if match is None:
        return None
    return match.group(1).strip()


def _append_jsonl(path: pathlib.Path, row: typing.Mapping[str, typing.Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _record_policy_timing(
    result: typing.Mapping[str, typing.Any],
    batch_sizes: list[float],
    policy_infer_ms: list[float],
    server_infer_ms: list[float],
    batch_wait_ms: list[float],
) -> None:
    policy_timing = result.get("policy_timing") or {}
    server_timing = result.get("server_timing") or {}
    if "batch_size" in policy_timing:
        batch_sizes.append(float(policy_timing["batch_size"]))
    elif "batch_size" in server_timing:
        batch_sizes.append(float(server_timing["batch_size"]))
    if "infer_ms" in policy_timing:
        policy_infer_ms.append(float(policy_timing["infer_ms"]))
    if "infer_ms" in server_timing:
        server_infer_ms.append(float(server_timing["infer_ms"]))
    if "batch_wait_ms" in server_timing:
        batch_wait_ms.append(float(server_timing["batch_wait_ms"]))


def _mean_or_none(values: typing.Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _resolve_task_ids(num_tasks_in_suite: int, task_ids: str | None, max_tasks: int | None) -> list[int]:
    if task_ids:
        resolved = []
        for raw_part in task_ids.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"Invalid descending task id range: {part}")
                resolved.extend(range(start, end + 1))
            else:
                resolved.append(int(part))
    else:
        resolved = list(range(num_tasks_in_suite))

    if max_tasks is not None and max_tasks > 0:
        resolved = resolved[:max_tasks]

    invalid = [task_id for task_id in resolved if task_id < 0 or task_id >= num_tasks_in_suite]
    if invalid:
        raise ValueError(f"Task ids out of range for suite with {num_tasks_in_suite} tasks: {invalid}")
    if not resolved:
        raise ValueError("No task ids selected for evaluation.")
    return resolved


def _get_state(obs):
    """Extract state vector from LIBERO observation."""
    return np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
