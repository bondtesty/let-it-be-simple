#!/usr/bin/env python3
"""
Python CLI for LIBERO evaluation on Inspire GPU nodes.

Usage:
    .venv/bin/python scripts/eval_libero.py \
        --config-name <config_name> \
        --checkpoint-dir <checkpoint_dir> \
        --task-suite <task_suite> \
        --gpu-id 0 \
        --port 8000

Equivalent to scripts/eval_libero.sh but with named args, socket-based
server readiness detection, and PID-file cleanup.
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

OPENPI_ROOT = Path(__file__).parent.parent.resolve()
GL_LIBS = OPENPI_ROOT / ".gl_libs"
LIBERO_CONFIG = OPENPI_ROOT / ".libero_config"
LOG_DIR = OPENPI_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────

def check_gl_libs() -> None:
    if not GL_LIBS.exists():
        raise RuntimeError(
            f"ERROR: EGL libs not found at {GL_LIBS}. "
            "Run scripts/fix_egl.sh to install them."
        )


def check_libero_config() -> None:
    cfg_path = LIBERO_CONFIG / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(
            f"ERROR: LIBERO config not found at {cfg_path}. "
            "Ensure .libero_config/config.yaml exists."
        )


def check_libero_venv() -> None:
    venv_path = OPENPI_ROOT / "examples" / "libero" / ".venv"
    if not venv_path.exists():
        raise RuntimeError(
            f"ERROR: LIBERO venv not found at {venv_path}. "
            "Set up the libero eval environment first."
        )


def check_gpu_available(gpu_id: int) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        ids = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                ids.append(int(parts[0]))
        if gpu_id not in ids:
            raise RuntimeError(
                f"ERROR: GPU {gpu_id} not available. Available GPUs: {sorted(ids)}"
            )
    except FileNotFoundError:
        raise RuntimeError("ERROR: nvidia-smi not found. Are NVIDIA drivers installed?")


def preflight_checks(gpu_id: int) -> None:
    check_gl_libs()
    check_libero_config()
    check_libero_venv()
    check_gpu_available(gpu_id)


def patch_safe_close() -> None:
    """Apply safe_close patch to prevent EGL crashes after ~430 episodes."""
    main_py = OPENPI_ROOT / "examples" / "libero" / "main.py"
    patched_py = OPENPI_ROOT / "examples" / "libero" / "main_patched.py"
    if patched_py.exists():
        import shutil
        shutil.copy(str(patched_py), str(main_py))
        print("Applied safe_close patch (main_patched.py -> main.py).")
    else:
        print("WARNING: main_patched.py not found; safe_close patch not applied.")


# ──────────────────────────────────────────────────────────────────────────────
# Server readiness wait (socket-based)
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_server(port: int, server_pid: int, server_log: Path, timeout: int = 600) -> bool:
    print(f"Waiting for server on port {port}...")
    for i in range(timeout):
        # Check if server process is still alive
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            print(f"ERROR: Server died during startup. Check {server_log}")
            return False

        # Check if port is listening
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            print(f"Server ready after {i + 1}s")
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)

    print(f"ERROR: Server not ready after {timeout}s. Check {server_log}")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# EGL device remap
# ──────────────────────────────────────────────────────────────────────────────

def get_egl_device_id(cuda_gpu_id: int) -> int:
    """Resolve /dev/nvidia* minor number for a given CUDA GPU index."""
    try:
        dev_nodes = sorted(
            Path("/dev").glob("nvidia[0-9]*"),
            key=lambda p: int(p.name.replace("nvidia", "")),
        )
        dev_ids = [int(p.name.replace("nvidia", "")) for p in dev_nodes]
        if cuda_gpu_id < len(dev_ids):
            return dev_ids[cuda_gpu_id]
    except Exception:
        pass
    return cuda_gpu_id


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def build_eval_name(config_name: str, checkpoint_dir: str, task_suite: str, port: int, n_history: int, adaptive: bool, replan_steps: int) -> str:
    import os as _os
    ckpt_base = _os.path.basename(checkpoint_dir.rstrip("/"))
    eval_name = f"eval_{config_name}_{ckpt_base}_{task_suite}"
    if port != 8000:
        eval_name += f"_p{port}"
    if n_history != 1:
        eval_name += f"_hist{n_history}"
    if adaptive:
        eval_name += "_adaptive"
    if replan_steps > 0:
        eval_name += f"_replan{replan_steps}"
    return eval_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIBERO evaluation")
    parser.add_argument("--config-name", required=True, help="Config name (e.g. pi0_scd_libero)")
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to checkpoint directory (e.g. checkpoints/pi0_scd_libero/scd_mean_debug)"
    )
    parser.add_argument(
        "--task-suite", required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO task suite"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--flow-steps", type=int, default=10, help="Flow matching steps (default: 10)")
    parser.add_argument("--cfg-scale", type=float, default=0.0, help="CFG scale (default: 0)")
    parser.add_argument("--n-history", type=int, default=1, help="Number of history steps (default: 1)")
    parser.add_argument("--adaptive", type=int, default=1, help="Use adaptive sampling (default: 1)")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples for adaptive (default: 50)")
    parser.add_argument("--std-threshold", type=float, default=0.1, help="STD threshold for adaptive (default: 0.1)")
    parser.add_argument("--replan-steps", type=int, default=10, help="Replan steps (default: 10)")
    args = parser.parse_args()

    eval_name = build_eval_name(
        args.config_name, args.checkpoint_dir, args.task_suite,
        args.port, args.n_history, bool(args.adaptive), args.replan_steps,
    )
    server_log = LOG_DIR / f"{eval_name}_server.log"
    client_log = LOG_DIR / f"{eval_name}_client.log"

    print(f"=== LIBERO Eval: {eval_name} ===")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Task suite: {args.task_suite}")
    print(f"Server log: {server_log}")
    print(f"Client log: {client_log}")

    # Pre-flight checks
    preflight_checks(args.gpu_id)
    print("Pre-flight checks passed.")
    patch_safe_close()

    # Rotate existing logs instead of truncating to preserve prior run data
    if server_log.exists() and server_log.stat().st_size > 0:
        server_log.rename(server_log.with_suffix(".log.bak"))
    if client_log.exists() and client_log.stat().st_size > 0:
        client_log.rename(client_log.with_suffix(".log.bak"))

    # Kill any existing server on the same port
    _kill_existing_server(args.port)

    # Build serve_policy args (explicit venv python to avoid system python mismatch)
    venv_python = str(OPENPI_ROOT / ".venv" / "bin" / "python")
    serve_args = [
        venv_python, "scripts/serve_policy.py",
        "--env", "LIBERO",
        "--port", str(args.port),
    ]
    if args.flow_steps > 0:
        serve_args.extend(["--num-flow-steps", str(args.flow_steps)])
    if args.cfg_scale != 0.0:
        serve_args.extend(["--cfg-scale", str(args.cfg_scale)])
    serve_args.extend([
        "policy:checkpoint",
        "--policy.config", args.config_name,
        "--policy.dir", args.checkpoint_dir,
    ])

    # Environment for server
    server_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "OPENPI_DATA_HOME": str(OPENPI_ROOT / "data"),
        "TORCH_HOME": str(OPENPI_ROOT.parent / "data"),
        "HF_HOME": str(OPENPI_ROOT.parent / "data"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
    }

    # Resolve EGL device before server start (needed for log headers)
    egl_device = get_egl_device_id(args.gpu_id)
    if egl_device != args.gpu_id:
        client_cuda = egl_device
    else:
        client_cuda = args.gpu_id

    # Write parameter header to both logs before any process output
    _write_log_header(server_log, args, eval_name, egl_device, client_cuda, serve_args)
    _write_log_header(client_log, args, eval_name, egl_device, client_cuda, serve_args)

    # Start policy server
    print("Starting policy server...")
    server_proc = subprocess.Popen(
        serve_args,
        cwd=str(OPENPI_ROOT),
        env=server_env,
        stdout=open(server_log, "w"),
        stderr=subprocess.STDOUT,
    )
    server_pid = server_proc.pid
    print(f"Server PID: {server_pid}")

    # Wait for server to be ready
    if not wait_for_server(args.port, server_pid, server_log):
        server_proc.terminate()
        sys.exit(1)

    print(f"EGL device: {egl_device}, client CUDA: {client_cuda}")

    # Build client args
    client_args = [
        str(OPENPI_ROOT / "examples/libero/main.py"),
        "--args.host", "localhost",
        "--args.port", str(args.port),
        "--args.task-suite-name", args.task_suite,
        "--args.video-out-path", str(OPENPI_ROOT / f"data/libero/videos/{eval_name}"),
    ]
    if args.n_history > 1:
        client_args.extend(["--args.n-history-steps", str(args.n_history)])
    if args.adaptive:
        client_args.extend([
            "--args.adaptive",
            "--args.n-samples", str(args.n_samples),
            "--args.std-threshold", str(args.std_threshold),
        ])
    if args.replan_steps > 0:
        client_args.extend(["--args.replan-steps", str(args.replan_steps)])

    # Environment for client
    # If the host already has NVIDIA EGL libs, prefer the system path to avoid
    # .gl_libs driver mismatches (e.g. libXext.so.6 missing in container).
    system_egl = Path("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0")
    if system_egl.exists():
        ld_library_path = "/usr/lib/x86_64-linux-gnu/nvidia"
        egl_vendor_dirs = "/usr/share/glvnd/egl_vendor.d"
    else:
        ld_library_path = f"{GL_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        egl_vendor_dirs = f"{GL_LIBS}/glvnd/egl_vendor.d"

    client_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(client_cuda),
        "LD_LIBRARY_PATH": ld_library_path,
        "__EGL_VENDOR_LIBRARY_DIRS": egl_vendor_dirs,
        "EGL_PLATFORM": "surfaceless",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(egl_device),
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": str(LIBERO_CONFIG),
        "PYTHONPATH": f"{OPENPI_ROOT / 'third_party/libero'}:{os.environ.get('PYTHONPATH', '')}",
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
        "EVAL_NAME": eval_name,
    }

    try:
        print(f"Starting LIBERO client ({args.task_suite}, n_history_steps={args.n_history})...")
        client_proc = subprocess.Popen(
            [str(OPENPI_ROOT / "examples/libero/.venv/bin/python"), *client_args],
            env=client_env,
            stdout=open(client_log, "w"),
            stderr=subprocess.STDOUT,
        )

        # Start watchdog to detect silent server death
        watchdog_stop = threading.Event()
        watchdog_thread = threading.Thread(
            target=_watchdog,
            args=(server_pid, args.port, client_proc, client_log, watchdog_stop),
            daemon=True,
        )
        watchdog_thread.start()

        client_exit = client_proc.wait()
        watchdog_stop.set()
        watchdog_thread.join(timeout=5)
    except Exception as e:
        print(f"ERROR: Client failed to start: {e}")
        client_exit = 1

    print(f"Client finished with exit code: {client_exit}")
    print("=== Results ===")
    _print_results(client_log)

    # Write completion marker on clean exit
    if client_exit == 0:
        _write_done_marker(client_log, eval_name)

    # Cleanup server
    print("Cleaning up server...")
    _cleanup_server(server_pid, args.port)

    print(f"Done. Full logs: {client_log}")
    sys.exit(client_exit)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kill_existing_server(port: int) -> None:
    """Kill any existing serve_policy.py process on the given port."""
    # Method 1: lsof (catches stale servers regardless of cmdline args)
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    os.kill(int(line), 9)
                    print(f"Killed existing process on port {port} PID {line}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass

    # Method 2: pgrep (works when lsof is unavailable)
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"serve_policy.py.*--port.*{port}"],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    os.kill(int(line), 9)
                    print(f"Killed existing server PID {line}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass
    time.sleep(1)


def _watchdog(
    server_pid: int,
    port: int,
    client_proc: subprocess.Popen,
    client_log: Path,
    stop_event: threading.Event,
    check_interval: int = 10,
    port_fail_threshold: int = 3,
    log_stall_threshold: int = 6,
) -> None:
    """Monitor server port and client log growth during evaluation.

    If the server stops responding on its port or the client log stalls,
    kill both processes to avoid wasting GPU time on a dead run.
    """
    port_fails = 0
    log_stalls = 0
    last_log_size = client_log.stat().st_size if client_log.exists() else 0

    while not stop_event.is_set() and client_proc.poll() is None:
        time.sleep(check_interval)
        if stop_event.is_set() or client_proc.poll() is not None:
            break

        # Check 1: is the server process still alive?
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            print(f"\nWATCHDOG: Server PID {server_pid} died. Killing client...")
            client_proc.kill()
            break

        # Check 2: is the server port still responding?
        port_ok = False
        try:
            sock = socket.create_connection(("localhost", port), timeout=2)
            sock.close()
            port_ok = True
        except (ConnectionRefusedError, OSError):
            pass

        if not port_ok:
            port_fails += 1
            print(f"\nWATCHDOG: Server not responding on port {port} ({port_fails}/{port_fail_threshold})")
            if port_fails >= port_fail_threshold:
                print(f"WATCHDOG: Server port dead for {port_fail_threshold * check_interval}s — killing client and server.")
                client_proc.kill()
                try:
                    os.kill(server_pid, 9)
                except ProcessLookupError:
                    pass
                break
        else:
            port_fails = 0

        # Check 3: is the client log still growing?
        current_log_size = client_log.stat().st_size if client_log.exists() else 0
        if current_log_size == last_log_size:
            log_stalls += 1
            if log_stalls >= log_stall_threshold:
                print(f"\nWATCHDOG: Client log stalled for {log_stall_threshold * check_interval}s — killing client and server.")
                client_proc.kill()
                try:
                    os.kill(server_pid, 9)
                except ProcessLookupError:
                    pass
                break
        else:
            log_stalls = 0
            last_log_size = current_log_size


def _cleanup_server(pid: int, port: int) -> None:
    """Kill server by PID, then do a final sweep by port."""
    try:
        os.kill(pid, 9)
    except (ProcessLookupError, PermissionError):
        pass
    _kill_existing_server(port)


def _write_log_header(log_path: Path, args: argparse.Namespace, eval_name: str, egl_device: int, client_cuda: int, server_args: list[str]) -> None:
    """Prepend an eval parameter header to a log file for reproducibility."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = [
        f"# === EVAL PARAMS @ {timestamp} ===",
        f"# config_name:    {args.config_name}",
        f"# checkpoint_dir: {args.checkpoint_dir}",
        f"# task_suite:     {args.task_suite}",
        f"# gpu_id:         {args.gpu_id}  (EGL device={egl_device}, CUDA device={client_cuda})",
        f"# port:           {args.port}",
        f"# flow_steps:     {args.flow_steps}",
        f"# cfg_scale:      {args.cfg_scale}",
        f"# n_history:      {args.n_history}",
        f"# adaptive:       {args.adaptive}",
        f"# n_samples:      {args.n_samples}",
        f"# std_threshold:  {args.std_threshold}",
        f"# replan_steps:   {args.replan_steps}",
        f"# eval_name:      {eval_name}",
        f"# server_cmd:     {' '.join(str(a) for a in server_args)}",
        "# === END PARAMS ===",
        "",
    ]

    existing = log_path.read_text() if log_path.exists() else ""
    log_path.write_text("\n".join(header) + existing)


def _print_results(log_path: Path) -> None:
    """Extract and print success rate / total episodes from client log."""
    if not log_path.exists():
        return
    content = log_path.read_text()
    for line in content.splitlines():
        line_lower = line.lower()
        if "success" in line_lower or "total episode" in line_lower:
            print(line.strip())
    # Also show the last 5 lines for context
    lines = content.splitlines()
    if lines:
        print("--- Last 5 lines ---")
        for line in lines[-5:]:
            print(line)


def _write_done_marker(log_path: Path, eval_name: str) -> None:
    """Write a .done marker file next to the client log on successful completion."""
    if not log_path.exists():
        return
    content = log_path.read_text()

    # Parse episodes
    episodes = 0
    for line in content.splitlines():
        m = re.search(r"episodes completed so far[:\s]*(\d+)", line, re.IGNORECASE)
        if m:
            episodes = int(m.group(1))

    # Parse success rate and derive successes
    rate = 0.0
    successes = 0
    for line in content.splitlines():
        m = re.search(r"success rate[:\s]*([\d.]+)%", line, re.IGNORECASE)
        if m:
            rate = float(m.group(1))
            if episodes > 0:
                successes = int(round(rate * episodes / 100.0))
            break
        # Fallback: look for explicit success/total pattern
        m2 = re.search(r"(\d+)[\s/]+(\d+)\s*(?:episodes?|trials?)?\s*success", line, re.IGNORECASE)
        if m2:
            successes = int(m2.group(1))
            episodes = int(m2.group(2))
            if episodes > 0:
                rate = successes * 100.0 / episodes
            break

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    done_path = log_path.with_suffix(log_path.suffix + ".done")
    done_path.write_text(f"{timestamp} {eval_name} {episodes} {successes} {rate:.1f}\n")
    print(f"Wrote done marker: {done_path}")


if __name__ == "__main__":
    main()
