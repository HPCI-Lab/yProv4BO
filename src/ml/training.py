
import os
import json
import time
import subprocess
from typing import Tuple

from src.bayesian.context import RunContext
from src.consts import VERBOSE

def run_training(params: dict, iteration: int, candidate: int, ctx: RunContext) -> Tuple[float, float]:
    run_tag   = f"bo_iter{iteration:03d}_cand{candidate:02d}"
    trial_out = str(ctx.trial_dir / run_tag)
    os.makedirs(trial_out, exist_ok=True)
    metrics_path = os.path.join(trial_out, "bayesopt_metrics.json")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    # ------------------------------------------------------------------
    # torchrun invocation: --standalone is only valid for single-node.
    # Multi-node requires a rendezvous endpoint via c10d.
    # ------------------------------------------------------------------
    torchrun_flags = [
        f"--nnodes={ctx.nnodes}",
        f"--nproc_per_node={ctx.nproc_per_node}",
    ]
    if ctx.nnodes == 1:
        torchrun_flags.append("--standalone")
    else: 
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = str(29500 + (iteration % 100))
        torchrun_flags += [
            f"--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
        ]

    cmd = [
        # "torchrun",
        # *torchrun_flags,
        "python", 

        ctx.train_script,

        "--wireup_method",        "local", #"torchrun",
        "--save_frequency",       str(0),
        "--data_dir_prefix",      ctx.data_dir,
        "--output_dir",           trial_out,
        "--run_tag",              run_tag,
        "--max_epochs",           str(ctx.max_epochs),
        "--local_batch_size",     str(params["local_batch_size"]),
        "--optimizer",            params["optimizer"],
        "--start_lr",             f"{params['start_lr']:.6e}",
        "--weight_decay",         f"{params['weight_decay']:.6e}",
        "--lr_warmup_steps",      str(params["lr_warmup_steps"]),
        "--lr_schedule",          f"type={params['lr_schedule']},t_max={ctx.max_epochs},decay_rate=0.1",
        "--batchnorm_group_size", str(params["batchnorm_group_size"]),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"
    env["TP_SOCKET_CHECK_HOSTNAME"] = "0"
    env["GLOO_SOCKET_IFNAME"] = "lo0"

    # For single-node --standalone, torchrun manages MASTER_ADDR/PORT itself.
    # For multi-node we rely on MASTER_ADDR already being in the environment
    # (set by the batch scheduler) and let torchrun's rdzv handle the rest.
    if ctx.nnodes == 1:
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(29500 + (iteration % 100))

    env["MIOPEN_USER_DB_PATH"]    = "/tmp/my-miopen-cache"
    env["MIOPEN_CUSTOM_CACHE_DIR"] = "/tmp/my-miopen-cache"

    # Strip any inherited distributed-training env vars so torchrun can set
    # them cleanly for the child process.
    for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "TORCHELASTIC_RESTART_COUNT"]:
        env.pop(var, None)

    if VERBOSE: 
        print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, check=False)

    if proc.returncode != 0 and VERBOSE:
        print(f"  [WARNING] torchrun exited with code {proc.returncode} ")

    # Timing log stored inside the run directory (not in bare CWD)
    with open(ctx.timing_log, "a") as f:
        f.write(f"iter={iteration:03d} cand={candidate:02d}\n")

    return _read_metrics(metrics_path)


def _read_metrics(metrics_path: str) -> Tuple[float, float]:
    """Read bayesopt_metrics.json from the given per-trial path."""
    retries = 5
    for attempt in range(retries):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            iou    = float(m["iou_validation"])
            energy = float(m["energy_kWh"])
            # Sanity check: (0, 0) means the trial silently crashed
            if iou == 0.0 and energy == 0.0:
                raise ValueError("metrics are (0, 0) — trial likely crashed")
            return iou, energy
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
            if attempt < retries - 1 and VERBOSE:
                print(f"  [WARNING] metrics not ready ({e}), retrying in 5s...")
                time.sleep(5)
            else:
                if VERBOSE: 
                    print(f"  [ERROR] could not read {metrics_path} — returning pessimistic fallback")
                # Pessimistic fallback: BoTorch will avoid this region
                return 0.0, 1e6

