"""Quick mode collapse test for RAE models.

Feeds different prompts and images to the model and checks if outputs vary.
A mode-collapsed model produces nearly identical outputs regardless of input.

Usage:
    python scripts/test_mode_collapse.py <config_name> <checkpoint_path>

Example:
    python scripts/test_mode_collapse.py pi0_rae_libero_v4 checkpoints/pi0_rae_libero_v4/rae_v4/5000
"""

import sys
import os
import json
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cuda")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <config_name> <checkpoint_path>")
        sys.exit(1)

    config_name = sys.argv[1]
    checkpoint_path = sys.argv[2]

    print(f"Loading model: {config_name} from {checkpoint_path}", flush=True)

    from openpi.training import config as config_lib
    from openpi.policies import policy_config as pc

    config = config_lib.get_config(config_name)
    policy = pc.create_trained_policy(config, checkpoint_path)
    print("Model loaded.", flush=True)

    # Load norm stats for realistic state
    ns_path = os.path.join(checkpoint_path, "assets/physical-intelligence/libero")
    ns = json.load(open(os.path.join(ns_path, "norm_stats.json")))
    ns = ns.get("norm_stats", ns)
    state_mean = np.array(ns["state"]["mean"][:7])
    state_std = np.array(ns["state"]["std"][:7])

    state = (state_mean + 0.1 * state_std).astype(np.float32)
    gray_img = np.ones((256, 256, 3), dtype=np.uint8) * 128

    # --- Test 1: Prompt sensitivity ---
    prompts = [
        "pick up the black bowl on the wooden cabinet and place it on the plate",
        "open the top drawer of the cabinet",
        "close the bottom drawer of the cabinet",
    ]
    print("\n=== Test 1: Prompt Sensitivity ===", flush=True)
    prompt_results = []
    for prompt in prompts:
        example = {
            "observation/state": state,
            "observation/image": gray_img,
            "observation/wrist_image": gray_img,
            "prompt": prompt,
        }
        output = policy.infer(example)
        actions = output["actions"][0, :7]
        prompt_results.append(actions)
        print(f"  {prompt[:40]:40s} -> {np.array2string(actions, precision=3, suppress_small=True)}", flush=True)

    pr = np.array(prompt_results)
    prompt_var = np.mean(np.var(pr, axis=0))
    prompt_cosines = []
    for i in range(3):
        for j in range(i + 1, 3):
            cos = np.dot(pr[i], pr[j]) / (np.linalg.norm(pr[i]) * np.linalg.norm(pr[j]) + 1e-8)
            prompt_cosines.append(cos)
    avg_prompt_cos = np.mean(prompt_cosines)

    # --- Test 2: Image sensitivity ---
    print("\n=== Test 2: Image Sensitivity ===", flush=True)
    images = {
        "dark (30)": np.ones((256, 256, 3), dtype=np.uint8) * 30,
        "bright (220)": np.ones((256, 256, 3), dtype=np.uint8) * 220,
        "random": np.random.RandomState(42).randint(0, 255, (256, 256, 3), dtype=np.uint8),
    }
    prompt = prompts[0]
    img_results = []
    for name, img in images.items():
        example = {
            "observation/state": state,
            "observation/image": img,
            "observation/wrist_image": img,
            "prompt": prompt,
        }
        output = policy.infer(example)
        actions = output["actions"][0, :7]
        img_results.append(actions)
        print(f"  {name:15s} -> {np.array2string(actions, precision=3, suppress_small=True)}", flush=True)

    ir = np.array(img_results)
    img_var = np.mean(np.var(ir, axis=0))
    # Compare dark vs random (most different)
    dark_rand_cos = np.dot(ir[0], ir[2]) / (np.linalg.norm(ir[0]) * np.linalg.norm(ir[2]) + 1e-8)
    dark_rand_l2 = np.linalg.norm(ir[0] - ir[2])

    # --- Results ---
    print("\n=== Results ===", flush=True)
    print(f"Prompt avg cosine similarity: {avg_prompt_cos:.4f}", flush=True)
    print(f"Prompt action variance:       {prompt_var:.6f}", flush=True)
    print(f"Image dark-vs-random cosine:  {dark_rand_cos:.4f}", flush=True)
    print(f"Image dark-vs-random L2:      {dark_rand_l2:.4f}", flush=True)

    # Also compute max per-dim range (more robust than cosine for subtle diffs)
    max_dim_range = np.max(np.ptp(pr, axis=0))
    print(f"Prompt max dim range:         {max_dim_range:.4f}", flush=True)

    # Thresholds based on empirical observations:
    # Full mode collapse (v3@30k): prompt cosine > 0.99, max dim range < 0.10
    # Partial sensitivity (v4@30k): cosine ~0.98, max dim range ~0.24
    # Good sensitivity (v4@2k): cosine 0.56-0.86, max dim range > 0.50
    # Use max dim range as primary indicator (cosine misses subtle but important diffs)
    mode_collapsed = max_dim_range < 0.10

    if mode_collapsed:
        print("\n** VERDICT: MODE COLLAPSE DETECTED **", flush=True)
        print("   Model outputs are nearly identical for different prompts.", flush=True)
    else:
        print("\n** VERDICT: NO MODE COLLAPSE **", flush=True)
        print("   Model produces meaningfully different outputs for different inputs.", flush=True)

    return 0 if not mode_collapsed else 1


if __name__ == "__main__":
    sys.exit(main())
