
import torch
import numpy as np

from botorch.utils.transforms import unnormalize

from src.consts import WORLD_SIZE

PARAM_NAMES = [
    "log10_start_lr",       # float, LOG10 space  [-4 .. -2]  → 1e-4 .. 1e-2
    "log10_weight_decay",   # float, LOG10 space  [-6 .. -2]  → 1e-6 .. 1e-2
    "local_batch_size",     # categorical: 1,2,4  (continuous encoding)
    "lr_schedule_idx",      # categorical: 0=cosine_annealing  1=multistep
    "lr_warmup_steps",      # int                 [0 .. 800]
    "batchnorm_group_size", # categorical (continuous): maps to BN_GROUP_MAP
    "optimizer_idx",        # categorical: 0=Adam  1=AdamW
]

# LR and weight_decay bounds are in log10 space; params_to_dict does 10^x.
BOUNDS_RAW = torch.tensor([
    [-4.0,  -6.0,  1,  0,    0,  0,  0],   # lower bounds
    [-2.0,  -2.0,  4,  1,  800,  4,  1],   # upper bounds
], dtype=torch.double)

BOUNDS_NORM = torch.zeros_like(BOUNDS_RAW)
BOUNDS_NORM[1] = 1.0

N_OBJECTIVES    = 2
OBJECTIVE_NAMES = ["iou_validation", "energy_kWh"]

# Reference point in maximisation space [+iou, -energy]
REF_POINT = torch.tensor([-0.1, -0.5], dtype=torch.double)

LR_SCHEDULE_MAP = {0: "cosine_annealing", 1: "multistep"}

# BN_GROUP_MAP is rebuilt in main() once WORLD_SIZE is known (it may be set
# via CLI).  The module-level version uses the default WORLD_SIZE constant.
def _build_bn_group_map(world_size: int) -> dict:
    m: dict[int, int] = {}
    v, k = 1, 0
    while v <= world_size:
        m[k] = v
        k += 1
        v *= 2
    if world_size not in m.values():
        m[k] = world_size
    return m

BN_GROUP_MAP = _build_bn_group_map(WORLD_SIZE)

BS_VALUES     = [1, 2, 4]
OPTIMIZER_MAP = {0: "Adam", 1: "AdamW"}


def params_to_dict(x: torch.Tensor, bn_group_map: dict | None = None) -> dict:
    """Normalised [0,1]^7 tensor → real hyperparameter dict."""
    if bn_group_map is None:
        bn_group_map = BN_GROUP_MAP

    x_raw = unnormalize(x, BOUNDS_RAW)

    def snap(val, choices):
        return min(choices, key=lambda c: abs(c - val))

    start_lr     = float(10 ** float(x_raw[0]))
    weight_decay = float(np.clip(10 ** float(x_raw[1]), 0.0, 1e-2))

    return {
        "start_lr":             start_lr,
        "weight_decay":         weight_decay,
        "local_batch_size":     snap(float(x_raw[2]), BS_VALUES),
        "lr_schedule":          LR_SCHEDULE_MAP[max(0, min(1, int(round(float(x_raw[3])))))],
        "lr_warmup_steps":      int(np.clip(round(float(x_raw[4])), 0, 800)),
        "batchnorm_group_size": bn_group_map[max(0, min(len(bn_group_map)-1, int(round(float(x_raw[5])))))],
        "optimizer":            OPTIMIZER_MAP[max(0, min(1, int(round(float(x_raw[6])))))],
    }


def to_objectives(iou: float, energy: float) -> torch.Tensor:
    return torch.tensor([iou, -energy], dtype=torch.double)
