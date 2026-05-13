import json
from pathlib import Path
from datetime import datetime

from src.consts import BASE_RESULTS_DIR, N_INITIAL, N_ITERATIONS, MAX_EPOCHS, VERBOSE
from src.params import ParamSpace


class RunContext:
    def __init__(self,
                 run_id=None,
                 results_dir: Path = BASE_RESULTS_DIR,
                 data_dir: str = "./",
                 train_script: str = "",
                 nnodes: int = 1,
                 nproc_per_node: int = 1,
                 max_epochs: int = 1,
                 space: ParamSpace = None,
                 ):
        self.run_id         = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir        = results_dir / self.run_id
        self.data_dir       = data_dir
        self.train_script   = train_script
        self.nnodes         = nnodes
        self.nproc_per_node = nproc_per_node
        self.max_epochs     = max_epochs
        self.space          = space

        for sub in ["detailed_logs", "state", "gp_checkpoints", "trials"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

        self.log_dir    = self.run_dir / "detailed_logs"
        self.state_dir  = self.run_dir / "state"
        self.gp_dir     = self.run_dir / "gp_checkpoints"
        self.trial_dir  = self.run_dir / "trials"
        self.timing_log = self.run_dir / "timing.txt"

        meta = {
            "run_id":          self.run_id,
            "benchmark":       "DeepCAM-MLPerf-HPC",
            "started_at":      datetime.now().isoformat(),
            "n_initial":       N_INITIAL,
            "n_iterations":    N_ITERATIONS,
            "max_epochs":      MAX_EPOCHS,
            "param_names":     space.param_names     if space else [],
            "objective_names": space.objective_names if space else [],
            "bounds_raw":      space.bounds_raw.tolist() if space else [],
        }
        with open(self.run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if VERBOSE:
            print(f"\n  Run ID    : {self.run_id}")
            print(f"  Output    : {self.run_dir}")