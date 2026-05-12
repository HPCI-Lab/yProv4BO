import json
from pathlib import Path
from datetime import datetime

from src.consts import BASE_RESULTS_DIR, N_INITIAL, N_ITERATIONS, MAX_EPOCHS, VERBOSE
from src.params import PARAM_NAMES, OBJECTIVE_NAMES, BOUNDS_RAW

class RunContext:
    def __init__(self, 
                 run_id=None, 
                 results_dir: Path = BASE_RESULTS_DIR, 
                 data_dir : str = "./", 
                 train_script : str = "", 
                 nnodes : int = 1, 
                 nproc_per_node : int = 1, 
                 max_epochs : int = 1, 
                 bn_group_map : dict = {}
                ):
        self.run_id  = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = results_dir / self.run_id
        self.data_dir = data_dir
        self.train_script = train_script
        self.nnodes = nnodes
        self.nproc_per_node = nproc_per_node
        self.max_epochs = max_epochs
        self.bn_group_map = bn_group_map

        for sub in ["detailed_logs", "state", "gp_checkpoints", "trials"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

        self.log_dir   = self.run_dir / "detailed_logs"
        self.state_dir = self.run_dir / "state"
        self.gp_dir    = self.run_dir / "gp_checkpoints"
        self.trial_dir = self.run_dir / "trials"
        self.timing_log = self.run_dir / "timing.txt"   # replaces bare tmp.txt

        meta = {
            "run_id":          self.run_id,
            "benchmark":       "DeepCAM-MLPerf-HPC",
            "started_at":      datetime.now().isoformat(),
            "n_initial":       N_INITIAL,
            "n_iterations":    N_ITERATIONS,
            "max_epochs":      MAX_EPOCHS,
            "param_names":     PARAM_NAMES,
            "objective_names": OBJECTIVE_NAMES,
            "bounds_raw":      BOUNDS_RAW.tolist(),
        }
        with open(self.run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if VERBOSE: 
            print(f"\n  Run ID    : {self.run_id}")
            print(f"  Output    : {self.run_dir}")

