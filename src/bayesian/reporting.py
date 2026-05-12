
import pandas as pd
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated

from src.params import params_to_dict, PARAM_NAMES, OBJECTIVE_NAMES, BOUNDS_RAW
from src.consts import VERBOSE

def print_pareto(train_x, train_y, iteration, bn_group_map: dict):
    mask   = is_non_dominated(train_y)
    py, px = train_y[mask], train_x[mask]
    if VERBOSE: 
        print(f"\n  Pareto front iter {iteration}  ({mask.sum().item()} pts):")
        print(f"  {'IoU':>8}  {'Energy':>10}  {'lr':>10}  "
            f"{'opt':>6}  {'sched':>18}  {'wu':>5}  {'bn':>4}")
        print(f"  {'-'*70}")
    for i in range(len(py)):
        p = params_to_dict(px[i], bn_group_map)
        if VERBOSE: 
            print(f"  {py[i,0].item():8.4f}  {-py[i,1].item():10.4f}  "
                f"{p['start_lr']:10.2e}  {p['optimizer']:>6}  "
                f"{p['lr_schedule']:>18}  {p['lr_warmup_steps']:5d}  "
                f"{p['batchnorm_group_size']:4d}")


def save_state(train_x, train_y, iteration, ctx):
    torch.save({"train_x": train_x, "train_y": train_y},
               ctx.state_dir / f"state_iter{iteration:03d}.pt")

def save_gp(train_x, train_y, iteration, ctx):
    torch.save({
        "train_x":     train_x,
        "train_y":     train_y,
        "param_names": PARAM_NAMES,
        "obj_names":   OBJECTIVE_NAMES,
        "bounds_raw":  BOUNDS_RAW,
        "iteration":   iteration,
        "run_id":      ctx.run_id,
    }, ctx.gp_dir / f"state_iter{iteration:03d}.pt")


def final_report(train_x, train_y, ctx, bn_group_map: dict):
    mask = is_non_dominated(train_y)
    rows = []
    for py, px in zip(train_y[mask], train_x[mask]):
        p = params_to_dict(px, bn_group_map)
        rows.append({
            "iou_validation": py[0].item(),
            "energy_kWh":    -py[1].item(),
            **p,
        })
    df   = pd.DataFrame(rows).sort_values("iou_validation", ascending=False)
    path = ctx.run_dir / "pareto_front_final.csv"
    df.to_csv(path, index=False)
    if VERBOSE: 
        print("\n" + "="*60)
        print(f"  COMPLETE  [{ctx.run_id}]")
        print(f"  Pareto  : {path}")
        print("="*60)
        print(df.to_string(index=False))
    return df
