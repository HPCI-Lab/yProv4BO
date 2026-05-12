import argparse
import torch
from pathlib import Path
from datetime import datetime

from botorch.utils.sampling import draw_sobol_samples

from src.consts import BASE_RESULTS_DIR, _DEFAULT_TRAIN_SCRIPT, N_ITERATIONS, N_INITIAL, NNODES, NPROC_PER_NODE, MAX_EPOCHS, VERBOSE
from src.params import params_to_dict, to_objectives, PARAM_NAMES, N_OBJECTIVES, BOUNDS_NORM, _build_bn_group_map
from src.bayesian.context import RunContext
from src.bayesian.training import run_bayes_opt_loop, run_pipeline, save_gp
from src.bayesian.reporting import final_report, save_state

def run_mobo(data_dir, n_iter, n_initial, max_epochs, run_id, train_script, nnodes, nproc_per_node, results_dir: Path, bn_group_map: dict):

    ctx     = RunContext(run_id, results_dir, data_dir, train_script, nnodes, nproc_per_node, max_epochs, bn_group_map)
    train_x = torch.empty((0, len(PARAM_NAMES)), dtype=torch.double)
    train_y = torch.empty((0, N_OBJECTIVES),     dtype=torch.double)

    # --- Phase 1: Sobol initialisation ---------------------------------------
    sobol_x = draw_sobol_samples(bounds=BOUNDS_NORM, n=n_initial, q=1).squeeze(1)

    for i in range(n_initial):
        x = sobol_x[i]
        p = params_to_dict(x, bn_group_map)

        iou, energy = run_pipeline(p, iteration=0, candidate=i, ctx=ctx)

        train_x = torch.cat([train_x, x.unsqueeze(0)])
        train_y = torch.cat([train_y, to_objectives(iou, energy).unsqueeze(0)])

    save_state(train_x, train_y, 0, ctx)

    model = run_bayes_opt_loop(train_x, train_y, 0, n_iter, ctx)
    
    if model is not None:
        save_gp(model, train_x, train_y, n_iter, ctx)

    return final_report(train_x, train_y, ctx, bn_group_map)

def resume_from_checkpoint(ckpt_path, data_dir, n_iter, train_script, nnodes, nproc_per_node, max_epochs, results_dir: Path, bn_group_map: dict):
    state   = torch.load(ckpt_path)
    train_x = state["train_x"]
    train_y = state["train_y"]

    done = int(Path(ckpt_path).stem.split("iter")[1])
    orig = Path(ckpt_path).parent.parent.name
    ctx  = RunContext(
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_resumed_{orig}_iter{done:03d}",
        results_dir, data_dir, train_script, nnodes, nproc_per_node, max_epochs, bn_group_map
    )
    if VERBOSE: 
        print(f"Resumed from iter {done} — {len(train_x)} pts evaluated.")

    model = run_bayes_opt_loop(train_x, train_y, done, n_iter, ctx)

    if model:
        save_gp(model, train_x, train_y, n_iter, ctx)
    return final_report(train_x, train_y, ctx, bn_group_map)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOBO — runs inside a BSub/Slurm job or locally, launches train.py via torchrun subprocess per iteration")
    # ---- paths / identity --------------------------------------------------
    parser.add_argument("--data_dir", required=True, help="Path to the dataset root (deepcam_dataset_1/ or MNIST root)")
    parser.add_argument("--train_script", default=_DEFAULT_TRAIN_SCRIPT, help=f"Absolute path to train.py (default: {_DEFAULT_TRAIN_SCRIPT})")
    parser.add_argument("--results_dir", default=str(BASE_RESULTS_DIR), help="Root directory for all run outputs")
    parser.add_argument("--run_id", default=None, help="Optional run identifier")
    parser.add_argument("--resume", default=None, help="Path to state_iterXXX.pt checkpoint to resume from")
    # ---- BO budget ---------------------------------------------------------
    parser.add_argument("--n_iter",    type=int, default=N_ITERATIONS, help=f"BO iterations (default: {N_ITERATIONS})")
    parser.add_argument("--n_initial", type=int, default=N_INITIAL, help=f"Sobol initialisation points (default: {N_INITIAL})")
    # ---- distributed / compute budget --------------------------------------
    parser.add_argument("--nnodes", type=int, default=NNODES, help=f"Number of nodes per training run (default: {NNODES})")
    parser.add_argument("--nproc_per_node", type=int, default=NPROC_PER_NODE, help=f"GPUs/processes per node (default: {NPROC_PER_NODE})")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS, help=f"Training epochs per trial (default: {MAX_EPOCHS})")
    args = parser.parse_args()

    # Rebuild BN_GROUP_MAP for the actual world size chosen at runtime
    world_size   = args.nnodes * args.nproc_per_node
    bn_group_map = _build_bn_group_map(world_size)

    results_dir  = Path(args.results_dir)

    if args.resume:
        resume_from_checkpoint(
            args.resume, args.data_dir, args.n_iter,
            args.train_script, args.nnodes, args.nproc_per_node,
            args.max_epochs, results_dir, bn_group_map,
        )
    else:
        run_mobo(
            args.data_dir, args.n_iter, args.n_initial, args.max_epochs,
            args.run_id, args.train_script, args.nnodes, args.nproc_per_node,
            results_dir, bn_group_map
        )