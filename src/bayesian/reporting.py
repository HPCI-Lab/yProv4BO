import pandas as pd
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated

from src.consts import VERBOSE


def print_pareto(train_x, train_y, iteration, ctx):
    space = ctx.space
    mask  = is_non_dominated(train_y)
    py, px = train_y[mask], train_x[mask]
    if VERBOSE:
        print(f"\n  Pareto front iter {iteration}  ({mask.sum().item()} pts):")
        print("  " + "  ".join(f"{n:>12}" for n in space.objective_names + space.param_names))
        print(f"  {'-' * (14 * (space.n_objectives + space.n_params))}")
        for i in range(len(py)):
            p = space.params_to_dict(px[i])
            obj_vals = [
                v.item() if obj["direction"] == "maximize" else -v.item()
                for v, obj in zip(py[i], space.objectives)
            ]
            row = obj_vals + [p[n] for n in space.param_names]
            print("  " + "  ".join(f"{v:>12}" if isinstance(v, str) else f"{v:>12.4g}" for v in row))


def save_state(train_x, train_y, iteration, ctx):
    torch.save({"train_x": train_x, "train_y": train_y},
               ctx.state_dir / f"state_iter{iteration:03d}.pt")


def save_gp(train_x, train_y, iteration, ctx):
    space = ctx.space
    torch.save({
        "train_x":     train_x,
        "train_y":     train_y,
        "param_names": space.param_names,
        "obj_names":   space.objective_names,
        "bounds_raw":  space.bounds_raw,
        "iteration":   iteration,
        "run_id":      ctx.run_id,
    }, ctx.gp_dir / f"state_iter{iteration:03d}.pt")


def final_report(train_x, train_y, ctx):
    space = ctx.space
    mask  = is_non_dominated(train_y)
    rows  = []
    for py, px in zip(train_y[mask], train_x[mask]):
        p = space.params_to_dict(px)
        obj_vals = {
            obj["name"]: v.item() if obj["direction"] == "maximize" else -v.item()
            for v, obj in zip(py, space.objectives)
        }
        rows.append({**obj_vals, **p})

    primary_obj = space.objective_names[0]
    df   = pd.DataFrame(rows).sort_values(primary_obj, ascending=False)
    path = ctx.run_dir / "pareto_front_final.csv"
    df.to_csv(path, index=False)
    if VERBOSE:
        print("\n" + "=" * 60)
        print(f"  COMPLETE  [{ctx.run_id}]")
        print(f"  Pareto  : {path}")
        print("=" * 60)
        print(df.to_string(index=False))
    return df