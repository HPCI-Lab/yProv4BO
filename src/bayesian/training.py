
import json
import torch
from typing import Tuple
import sys
sys.path.append("/Users/gabrielepadovani/Desktop/Università/yProv4ML")
import yprov4ml

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


def fit_model(train_x: torch.Tensor, train_y: torch.Tensor, n_objectives: int) -> SingleTaskGP:
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=n_objectives))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def next_candidate(
    model: SingleTaskGP,
    train_x: torch.Tensor,
    bounds_norm: torch.Tensor,
    ref_point: torch.Tensor,
) -> torch.Tensor:
    acq = qNEHVI(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        prune_baseline=True,
    )
    cand, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_norm,
        q=BATCH_SIZE_BO,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return cand


def run_pipeline(
    params: dict,
    iteration: int,
    candidate: int,
    ctx: RunContext,
) -> Tuple[float, ...]:
    """
    Run one training trial and return objective values as a plain tuple
    (in the same order as ctx.space.objectives).
    """
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"  [{ctx.run_id}]  Iter {iteration}  |  Cand {candidate}")
        for k, v in params.items():
            print(f"  {k}={v}")
        print(f"{'='*60}")

    # run_training returns (iou, energy) — kept as-is; adapt here if the
    # training script ever returns a different number of objectives.
    result = run_training(params, iteration, candidate, ctx)

    if VERBOSE:
        obj_str = "  ".join(
            f"{name}={val:.4f}"
            for name, val in zip(ctx.space.objective_names, result)
        )
        print(f"\n  → {obj_str}")

    log_path = ctx.log_dir / f"iter{iteration:03d}_cand{candidate:02d}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "iteration":  iteration,
                "candidate":  candidate,
                "parameters": params,
                "objectives": dict(zip(ctx.space.objective_names, result)),
            },
            f,
            indent=2,
        )

    return result


def run_bayes_opt_loop(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    done: int,
    n_iter: int,
    ctx: RunContext,
) -> SingleTaskGP | None:
    space = ctx.space
    model = None

    for iteration in range(done + 1, n_iter + 1):
        if VERBOSE:
            print(f"\n[BO {iteration}/{n_iter}]")

        model = fit_model(train_x, train_y, space.n_objectives)

        save_gp(train_x, train_y, iteration, ctx)

        x_next = next_candidate(model, train_x, space.bounds_norm, space.ref_point).squeeze(0)

        p = space.params_to_dict(x_next)

        result = run_pipeline(p, iteration=iteration, candidate=0, ctx=ctx)

        train_x = torch.cat([train_x, x_next.unsqueeze(0)])
        train_y = torch.cat([train_y, space.to_objectives(*result).unsqueeze(0)])
        save_state(train_x, train_y, iteration, ctx)

        if VERBOSE:
            print_pareto(train_x, train_y, iteration, ctx)

        yprov4ml.log_system_metrics("iteration", step=iteration)
        

    return model
