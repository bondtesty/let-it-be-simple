"""Batch-one decode latency decomposition for internal LIBERO SimVLA policies.

Mirrors the protocol of analysis_results/external_vla/pi0_droid_h10_decode_latency_b1_*:
warmed batch-one timing of the condition prefix (image/language/state encoding) and the
action decoder at 1 and 10 Euler steps, on synthetic LIBERO observations (latency is
content-independent; token counts and shapes match the deployed LIBERO eval path).

Example:
    uv run python scripts/profile_libero_decoding_latency.py \
        --config-name pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4 \
        --checkpoint-dir /path/to/checkpoints/pi0_libero_simvla_tiny_siglip4_head4_w768_50k_h10_a4/h10_a4_50k/49999 \
        --output analysis_results/latency/tiny_h10_a4_decode_latency_b1.json

(checkpoint root is LBS_CKPT_ROOT elsewhere; this script takes it explicitly)
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _config


def _time_blocked(fn, *, warmup: int, calls: int) -> list[float]:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(calls):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, nargs="+", default=(1, 10))
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--profiled-calls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir, default_prompt="pick up the bowl")

    rng = np.random.default_rng(args.seed)
    obs = {
        "observation/state": rng.random(8).astype(np.float32),
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the bowl",
    }

    # Model-level observation (post-transform, batched), as inside Policy.infer.
    inputs = policy._input_transform(dict(obs))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)

    key = jax.random.key(args.seed)
    noise = jnp.asarray(
        rng.standard_normal((1, config.model.action_horizon, config.model.action_dim)), dtype=jnp.float32
    )

    sample_actions = policy._sample_actions  # module-jitted
    embed_prefix = jax.jit(policy._model.embed_prefix)

    result = {
        "schema_version": "openpi.libero_decoding_latency.v1",
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint_dir),
        "protocol": {
            "batch_size": 1,
            "device": jax.devices()[0].platform,
            "device_name": jax.devices()[0].device_kind,
            "warmup_calls": args.warmup_calls,
            "profiled_calls": args.profiled_calls,
            "timed_boundary": "post-transform model forward: condition-prefix encoding and action denoising; "
            "excludes numpy input transforms, tokenizer, network, simulator",
            "inputs": "synthetic LIBERO observation (2x 224x224x3 uint8 images, 8-dim state, fixed prompt)",
        },
        "per_num_steps": {},
    }

    prefix_ms = _time_blocked(lambda: embed_prefix(observation), warmup=args.warmup_calls, calls=args.profiled_calls)
    result["condition_prefix_ms"] = {"mean": float(np.mean(prefix_ms)), "p50": float(np.median(prefix_ms))}

    for steps in args.num_steps:
        total_ms = _time_blocked(
            lambda: sample_actions(key, observation, num_steps=steps, noise=noise),
            warmup=args.warmup_calls,
            calls=args.profiled_calls,
        )
        # Deployed-path wall time through Policy.infer (includes numpy transforms).
        infer_ms = []
        np_noise = np.asarray(noise[0])
        for _ in range(args.warmup_calls):
            policy.infer(obs, noise=np_noise, num_steps=steps)
        for _ in range(args.profiled_calls):
            t0 = time.perf_counter()
            policy.infer(obs, noise=np_noise, num_steps=steps)
            infer_ms.append((time.perf_counter() - t0) * 1e3)
        result["per_num_steps"][str(steps)] = {
            "policy_forward_ms": {"mean": float(np.mean(total_ms)), "p50": float(np.median(total_ms))},
            "policy_infer_wall_ms": {"mean": float(np.mean(infer_ms)), "p50": float(np.median(infer_ms))},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    prefix = result["condition_prefix_ms"]["mean"]
    for steps, row in result["per_num_steps"].items():
        total = row["policy_forward_ms"]["mean"]
        print(
            f"steps={steps:>2}  prefix={prefix:6.2f} ms  decoder={total - prefix:6.2f} ms  "
            f"total={total:6.2f} ms ({1000.0 / total:5.1f} Hz)  infer_wall={row['policy_infer_wall_ms']['mean']:6.2f} ms"
        )


if __name__ == "__main__":
    main()
