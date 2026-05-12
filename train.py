import os
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from tqdm import tqdm

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("[train.py] codecarbon not found — energy will be reported as 0.0")


# =============================================================================
# Tiny CNN  (fast, ~99 % accuracy in a few epochs on MNIST)
# =============================================================================

class MnistCNN(nn.Module):
    """Two conv blocks + two FC layers."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 14×14
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# Argument parsing  (superset of original flags — extras silently ignored)
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(description="MNIST CNN — MOBO HPO target")

    # ---- identifiers / I/O -------------------------------------------------
    p.add_argument("--output_dir",       default="./output")
    p.add_argument("--run_tag",          default="mnist_run")
    p.add_argument("--data_dir_prefix",  default="./data",
                   help="Root dir; MNIST will be downloaded here if absent")

    # ---- distributed -------------------------------------------------------
    p.add_argument("--wireup_method",    default="torchrun")

    # ---- training ----------------------------------------------------------
    p.add_argument("--max_epochs",       type=int,   default=3)
    p.add_argument("--local_batch_size", type=int,   default=64)
    p.add_argument("--start_lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--optimizer",        default="Adam",
                   choices=["Adam", "AdamW", "SGD"])
    p.add_argument("--lr_schedule",      default=None,
                   help="Serialised schedule string, e.g. type=cosine_annealing,t_max=3")
    p.add_argument("--lr_warmup_steps",  type=int,   default=0)
    p.add_argument("--lr_warmup_factor", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_frequency", type=int, default=1)
    p.add_argument("--target_iou",       type=float, default=0.82,
                   help="Val accuracy threshold that counts as 'target reached'")

    # ---- checkpoint --------------------------------------------------------
    p.add_argument("--save_frequency",   type=int,   default=0)
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--model_prefix",     default="mnist_model")

    # ---- ignored DeepCAM flags (accepted for CLI compatibility) ------------
    p.add_argument("--channels",         nargs="+",  default=[0, 1, 2, 3],
                   help="(ignored) DeepCAM input channels")
    p.add_argument("--batchnorm_group_size", type=int, default=1,
                   help="(ignored) DeepCAM BN group size")
    p.add_argument("--seed",             type=int,   default=333)
    p.add_argument("--logging_frequency",type=int,   default=100)

    return p.parse_args()


# =============================================================================
# LR schedule helpers  (mirrors original train.py behaviour)
# =============================================================================

def _parse_schedule_str(s: str) -> dict:
    """'type=cosine_annealing,t_max=3,decay_rate=0.1' → dict"""
    result = {}
    for tok in s.split(","):
        k, _, v = tok.partition("=")
        result[k.strip()] = v.strip()
    return result


def build_scheduler(optimizer, lr_schedule_str, max_epochs, warmup_steps, last_step=0):
    if not lr_schedule_str:
        return None

    cfg  = _parse_schedule_str(lr_schedule_str)
    stype = cfg.get("type", "cosine_annealing")

    schedulers = []

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        schedulers.append(warmup)

    if stype == "cosine_annealing":
        t_max = int(cfg.get("t_max", max_epochs))
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        )
    elif stype == "multistep":
        milestones = [int(max_epochs * 0.5), int(max_epochs * 0.75)]
        decay_rate = float(cfg.get("decay_rate", 0.1))
        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                  gamma=decay_rate)
        )
    else:
        print(f"[train.py] Unknown schedule type '{stype}', skipping scheduler.")
        return None

    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=[warmup_steps]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    pargs = parse_arguments()

    # ---- distributed init --------------------------------------------------
    dist_available = dist.is_available()
    if dist_available and pargs.wireup_method == "torchrun":
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank       = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1

    # ---- device ------------------------------------------------------------
    torch.manual_seed(pargs.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(pargs.seed)
    else:
        device = torch.device("cpu")

    is_master = (rank == 0)

    if is_master:
        os.makedirs(pargs.output_dir, exist_ok=True)
        print(f"[train.py] device={device}  world={world_size}  "
              f"lr={pargs.start_lr:.2e}  wd={pargs.weight_decay:.2e}  "
              f"bs={pargs.local_batch_size}  opt={pargs.optimizer}  "
              f"epochs={pargs.max_epochs}")

    # ---- data --------------------------------------------------------------
    data_root = os.path.join(pargs.data_dir_prefix, "mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Only rank 0 downloads; others wait

    if is_master:
        torchvision.datasets.MNIST(data_root, train=True,  download=False)
        torchvision.datasets.MNIST(data_root, train=False, download=False)
    if dist.is_initialized():
        dist.barrier()

    train_ds = torchvision.datasets.MNIST(data_root, train=True,  transform=transform)
    val_ds   = torchvision.datasets.MNIST(data_root, train=False, transform=transform)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=pargs.seed) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_ds = Subset(train_ds, range(200))
    val_ds = Subset(val_ds, range(200))

    train_loader = DataLoader(train_ds, batch_size=pargs.local_batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,
                              sampler=val_sampler,
                              shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---- model -------------------------------------------------------------
    model = MnistCNN().to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.index is not None else None
        )

    # ---- optimizer ---------------------------------------------------------
    opt_cls = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": lambda p, **kw: optim.SGD(p, momentum=0.9, **kw)}[pargs.optimizer]
    optimizer = opt_cls(model.parameters(), lr=pargs.start_lr, weight_decay=pargs.weight_decay)

    scheduler = build_scheduler(optimizer, pargs.lr_schedule, pargs.max_epochs, pargs.lr_warmup_steps)

    criterion = nn.CrossEntropyLoss().to(device)

    # ---- optional checkpoint -----------------------------------------------
    start_epoch = 0
    if pargs.checkpoint:
        ckpt = torch.load(pargs.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if is_master:
            print(f"[train.py] Resumed from epoch {start_epoch}")

    # ---- training loop -----------------------------------------------------
    if HAS_CODECARBON:
        tracker = EmissionsTracker(output_dir=pargs.output_dir, log_level="error")
        tracker.start()

    best_acc      = 0.0
    stop_training = False

    for epoch in range(start_epoch, pargs.max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        for step, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if is_master and pargs.logging_frequency > 0 and \
            #         step % pargs.logging_frequency == 0:
                # print(f"  epoch {epoch+1}  step {step:4d}  "
                #       f"loss={loss.item():.4f}  "
                #       f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step()

        # ---- validation ----------------------------------------------------
        model.eval()
        correct = torch.tensor(0, dtype=torch.long, device=device)
        total   = torch.tensor(0, dtype=torch.long, device=device)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds   = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum()
                total   += labels.size(0)

        if world_size > 1:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total,   op=dist.ReduceOp.SUM)

        acc = float(correct) / float(total)
        best_acc = max(best_acc, acc)

        if is_master:
            print(f"  [epoch {epoch+1}/{pargs.max_epochs}]  "
                  f"val_acc={acc:.4f}  best={best_acc:.4f}")

        if acc >= pargs.target_iou:
            stop_training = True
            if is_master:
                print(f"  Target IoU/acc {pargs.target_iou} reached — stopping.")
            break

        # ---- optional checkpoint save --------------------------------------
        if pargs.save_frequency > 0 and is_master and \
                (epoch + 1) % pargs.save_frequency == 0:
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(pargs.output_dir,
                            f"{pargs.model_prefix}_epoch{epoch+1}.pt"))

    # ---- energy ------------------------------------------------------------
    energy_kWh = 0.0
    if HAS_CODECARBON:
        emissions  = tracker.stop()  # noqa: F841
        energy_kWh = float(tracker._total_energy.kWh)

    epochs_run = (start_epoch +
                  (pargs.max_epochs - start_epoch
                   if not stop_training
                   else (epoch + 1 - start_epoch)))

    print(f"---  METRICS: {best_acc, energy_kWh}  ---")

    # ---- write metrics for HPO controller ----------------------------------
    if is_master:
        results = {
            "iou_validation": best_acc,    # accuracy in [0, 1]
            "energy_kWh":     energy_kWh,
            "epochs_run":     epochs_run,
            "target_reached": bool(stop_training),
        }
        out_path = os.path.join(pargs.output_dir, "bayesopt_metrics.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[train.py] Wrote metrics → {out_path}")
        print(f"[train.py] acc={best_acc:.4f}  energy={energy_kWh:.6f} kWh")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()