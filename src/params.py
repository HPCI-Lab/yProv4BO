from __future__ import annotations

import math
import yaml
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from botorch.utils.transforms import unnormalize

from src.consts import _DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Parameter descriptor
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    """Describes one hyperparameter: its type, raw bounds, and how to decode it."""

    name:    str
    type:    str              # log_float | float | int | categorical
    bounds:  list             # [lo, hi] for continuous types; unused for categorical
    fixed:   bool 
    choices: list | None = None
    clip:    list | None = None   # [min, max] — optional hard clamp after decoding

    # Internal raw (un-normalised) bounds used by BoTorch
    _raw_lo: float = field(init=False, repr=False)
    _raw_hi: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.type == "log_float":
            self._raw_lo = math.log10(self.bounds[0])
            self._raw_hi = math.log10(self.bounds[1])
        elif self.type in ("float", "int"):
            self._raw_lo = float(self.bounds[0])
            self._raw_hi = float(self.bounds[1])
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"Parameter '{self.name}' is categorical but has no choices. "
                    "Pass them via dynamic_choices when calling load_param_space()."
                )
            self._raw_lo = 0.0
            self._raw_hi = float(len(self.choices) - 1)
        else:
            raise ValueError(f"Unknown parameter type: {self.type!r}")

    def decode(self, raw_val: float) -> Any:
        """Convert a raw (un-normalised) scalar to the actual hyperparameter value."""
        if self.type == "log_float":
            v = float(10 ** raw_val)
        elif self.type == "float":
            v = float(raw_val)
        elif self.type == "int":
            v = int(np.clip(round(raw_val), self.bounds[0], self.bounds[1]))
            return v                          # no clip list for int
        elif self.type == "categorical":
            idx = max(0, min(len(self.choices) - 1, int(round(raw_val))))
            return self.choices[idx]

        # Apply optional clamp for float-like types
        if self.clip:
            v = float(np.clip(v, self.clip[0], self.clip[1]))
        return v


# ---------------------------------------------------------------------------
# Parameter space — the object the rest of the codebase interacts with
# ---------------------------------------------------------------------------

class ParamSpace:
    """
    Encapsulates the full search space: parameter definitions, BoTorch bounds
    tensors, objective directions, and encoding/decoding utilities.
    """

    def __init__(self, params: list[ParamDef], objectives: list[dict], ref_point: list[float]):
        self.params     = params
        self.objectives = objectives

        self.n_params        = len(params)
        self.n_objectives    = len(objectives)
        self.param_names     = [p.name for p in params]
        self.objective_names = [o["name"] for o in objectives]

        # BoTorch expects bounds as (2, d) tensors in [0, 1] (normalised) and raw
        lows  = [p._raw_lo for p in params]
        highs = [p._raw_hi for p in params]
        self.bounds_raw  = torch.tensor([lows, highs], dtype=torch.double)
        self.bounds_norm = torch.zeros_like(self.bounds_raw)
        self.bounds_norm[1] = 1.0

        # Reference point in maximisation space
        self.ref_point = torch.tensor(ref_point, dtype=torch.double)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def params_to_dict(self, x: torch.Tensor) -> dict:
        """
        Convert a normalised [0, 1]^n_params tensor to a {name: value} dict
        of real hyperparameters ready to pass to the training script.
        """
        x_raw = unnormalize(x, self.bounds_raw)
        return {p.name: p.decode(float(x_raw[i])) for i, p in enumerate(self.params)}

    def to_objectives(self, *values: float) -> torch.Tensor:
        """
        Convert raw objective measurements (in their natural direction) to a
        maximisation-space tensor expected by BoTorch.

        Values must be supplied in the same order as objectives in the YAML.
        Example:
            space.to_objectives(iou, energy_kwh)
        """
        if len(values) != self.n_objectives:
            raise ValueError(
                f"Expected {self.n_objectives} objective values, got {len(values)}"
            )
        out = []
        for v, obj in zip(values, self.objectives):
            out.append(v if obj["direction"] == "maximize" else -v)
        return torch.tensor(out, dtype=torch.double)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["ParamSpace:"]
        for p in self.params:
            if p.type == "categorical":
                lines.append(f"  {p.name:30s}  categorical  choices={p.choices}")
            elif p.type == "int":
                lines.append(f"  {p.name:30s}  int          bounds={p.bounds}")
            else:
                lines.append(
                    f"  {p.name:30s}  {p.type:10s}  "
                    f"bounds=[{10**p._raw_lo:.2e}, {10**p._raw_hi:.2e}]"
                    + (f"  clip={p.clip}" if p.clip else "")
                )
        lines.append(f"Objectives: {self.objective_names}")
        lines.append(f"Ref point:  {self.ref_point.tolist()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_param_space(
    config_path: str | Path | None = None,
) -> ParamSpace:
    """
    Build a ParamSpace from a YAML config file.

    Args:
        config_path:     Path to the YAML config. Defaults to hpo_config.yaml
                         next to the project root.
        dynamic_choices: Dict mapping parameter name → list of choices, used to
                         populate parameters whose choices are null in the YAML.
                         Example:
                             dynamic_choices={"batchnorm_group_size": [1, 2, 4, 8]}
    """

    # Necessary params: (SLURM) (or LOCAL)
    # - walltime (hard cap)
    # - nnodes ()
    # - nproc_per_node ()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    nnodes = [p["bounds"][-1] for p in cfg["parameters"] if p["name"] == "nnodes"][0]
    nproc_per_node = [p["bounds"][-1] for p in cfg["parameters"] if p["name"] == "nproc_per_node"][0]

    world_size = nnodes * nproc_per_node
    bn_choices = build_bn_group_choices(world_size)
    dynamic_choices = {"batchnorm_group_size": bn_choices}

    param_defs = []
    for p in cfg["parameters"]:
        choices = p.get("choices")
        if choices is None and dynamic_choices and p["name"] in dynamic_choices:
            choices = dynamic_choices[p["name"]]
        param_defs.append(ParamDef(
            name    = p["name"],
            type    = p["type"],
            fixed   = p["fixed"], 
            bounds  = p.get("bounds", [0, 1]),
            choices = choices,
            clip    = p.get("clip"),
        ))

    return ParamSpace(
        params     = param_defs,
        objectives = cfg["objectives"],
        ref_point  = cfg["ref_point"],
    )


# ---------------------------------------------------------------------------
# Runtime utility: build the batchnorm group size choices from world_size
# ---------------------------------------------------------------------------

def build_bn_group_choices(world_size: int) -> list[int]:
    """
    Return sorted list of valid batchnorm group sizes for the given world_size:
    all powers of 2 up to world_size, plus world_size itself if not already included.
    """
    choices: list[int] = []
    v = 1
    while v <= world_size:
        choices.append(v)
        v *= 2
    if world_size not in choices:
        choices.append(world_size)
    return sorted(choices)


# ---------------------------------------------------------------------------
# Legacy shims — keep the rest of the codebase working without changes.
# These are derived from the loaded ParamSpace at import time.
# ---------------------------------------------------------------------------

def _build_bn_group_map(world_size: int) -> dict:
    """
    Legacy helper retained for checkpoint-resume compatibility.
    Returns an int→int dict mapping index → group size, matching the old format.
    """
    choices = build_bn_group_choices(world_size)
    return {i: v for i, v in enumerate(choices)}


# Module-level ParamSpace instance — loaded once, reused everywhere.
# Callers that need runtime dynamic_choices (e.g. batchnorm_group_size keyed
# on world_size) should call load_param_space() directly and pass the result
# through RunContext instead of relying on these module-level names.
_space: ParamSpace | None = None


def get_space(
    config_path: str | Path | None = None,
    dynamic_choices: dict[str, list] | None = None,
) -> ParamSpace:
    """
    Return the module-level ParamSpace, loading it on first call.
    Pass config_path / dynamic_choices on the first call to customise it;
    subsequent calls return the cached instance.
    """
    global _space
    if _space is None:
        _space = load_param_space(config_path, dynamic_choices)
    return _space
