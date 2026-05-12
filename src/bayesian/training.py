
import json
import torch
from typing import Tuple

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

from src.consts import BATCH_SIZE_BO, VERBOSE
from src.bayesian.context import RunContext
from src.ml.training import run_training
from src.bayesian.reporting import print_pareto, save_state, save_gp
from src.params import N_OBJECTIVES, BOUNDS_NORM, REF_POINT, params_to_dict, to_objectives

try:
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as qNEHVI,
    )
    if VERBOSE: 
        print("Using qLogNoisyExpectedHypervolumeImprovement")
except ImportError:
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement as qNEHVI,
    )
    if VERBOSE: 
        print("Using qNoisyExpectedHypervolumeImprovement (fallback)")


def fit_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=N_OBJECTIVES))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model

def next_candidate(model, train_x):
    acq = qNEHVI(
        model=model, ref_point=REF_POINT, X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        prune_baseline=True,
    )
    cand, _ = optimize_acqf(
        acq_function=acq, bounds=BOUNDS_NORM, q=BATCH_SIZE_BO,
        num_restarts=10, raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return cand


def run_pipeline(params: dict, iteration: int, candidate: int, ctx: RunContext) -> Tuple[float, float]:
    if VERBOSE: 
        print(f"\n{'='*60}")
        print(f"  [{ctx.run_id}]  Iter {iteration}  |  Cand {candidate}")
        print(f"  lr={params['start_lr']:.2e}  wd={params['weight_decay']:.2e}  "
            f"bs={params['local_batch_size']}  opt={params['optimizer']}")
        print(f"  sched={params['lr_schedule']}  wu={params['lr_warmup_steps']}  "
            f"bn={params['batchnorm_group_size']}")
        print(f"{'='*60}")

    iou, energy = run_training(params, iteration, candidate, ctx)

    if VERBOSE: 
        print(f"\n  → IoU={iou:.4f}  Energy={energy:.4f} kWh")

    log_path = ctx.log_dir / f"iter{iteration:03d}_cand{candidate:02d}.json"
    with open(log_path, "w") as f:
        json.dump({
            "iteration":  iteration,
            "candidate":  candidate,
            "parameters": params,
            "objectives": {"iou_validation": iou, "energy_kWh": energy},
        }, f, indent=2)

    return iou, energy


def run_bayes_opt_loop(train_x, train_y, done, n_iter, ctx): 
    model = None
    for iteration in range(done + 1, n_iter + 1):
        if VERBOSE: 
            print(f"\n[BO {iteration}/{n_iter}]")

        model  = fit_model(train_x, train_y)

        save_gp(train_x, train_y, iteration, ctx)

        x_next = next_candidate(model, train_x).squeeze(0)

        p = params_to_dict(x_next, ctx.bn_group_map)

        iou, energy = run_pipeline(p, iteration=iteration, candidate=0, ctx=ctx)

        train_x = torch.cat([train_x, x_next.unsqueeze(0)])
        train_y = torch.cat([train_y, to_objectives(iou, energy).unsqueeze(0)])
        save_state(train_x, train_y, iteration, ctx)

        if VERBOSE: 
            print_pareto(train_x, train_y, iteration, ctx.bn_group_map)

    return model