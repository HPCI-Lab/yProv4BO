
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_TRAIN_SCRIPT = Path("./examples/train.py")
_DEFAULT_CONFIG = Path("./examples/params.yaml")

BASE_RESULTS_DIR = Path("./deepcam_results")

N_INITIAL     = 3
N_ITERATIONS  = 50
BATCH_SIZE_BO = 1
MAX_EPOCHS    = 1

NNODES         = 1
NPROC_PER_NODE = 1   # GPUs (or CPU workers) per training run
WORLD_SIZE     = NNODES * NPROC_PER_NODE
VERBOSE = False
