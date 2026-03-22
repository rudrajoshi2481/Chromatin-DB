"""
train.py — End-to-end training loop for MQ-VAE Hi-C structural fingerprinting.

Usage:
    cd /app/tmp/DATABASE_CONCEPT
    python src/train.py                        # full training, all 5 cell lines
    python src/train.py --epochs 20            # quick run
    python src/train.py --cell_lines K562 IMR-90   # subset
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINTS_DIR, TRASH_DIR, PLOTS_DIR, PROCESSED_DIR, CELL_LINE_REGISTRY,
    BATCH_SIZE, NUM_EPOCHS, LR, BETAS, WEIGHT_DECAY,
    GRAD_CLIP, WARMUP_STEPS, MIN_STEPS_PER_EPOCH, NUM_WORKERS, SEED,
    LOG_EVERY_N_STEPS, SAVE_EVERY_N_EPOCHS, PROJECT_ROOT,
    MCOOL_DIR, CCRE_REGISTRY,
)

RUNS_DIR = PROJECT_ROOT / "runs"

from dataset import build_dataloaders, build_dataloaders_h5, build_dataloaders_dat, load_cell_line_label_map
from model import MQVAE
from loss import total_loss
from masker import get_tau_f
from replicates import validate_registry, active_samples, print_registry_status
from dashboard import TrainingDashboard, collect_fingerprints, collect_codebook_usage, compute_silhouette


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine annealing."""
    sched1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched1, sched2], milestones=[warmup_steps]
    )


def train_one_epoch(
    model:     MQVAE,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    device:    torch.device,
    log_every: int = LOG_EVERY_N_STEPS,
) -> dict:
    model.train()
    running = {k: 0.0 for k in ["total", "recon", "vq", "classifier", "classifier_acc"]}
    n_steps = 0
    all_indices: set = set()  # collect unique codebook indices across all batches

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for batch in pbar:
        contact  = batch["contact"].to(device)   # [B, 1, 256, 256]
        assay_id = batch["assay_id"].to(device)  # [B]
        cell_idx = batch["cell_idx"].to(device)  # [B] unique int per cell line

        outputs = model(contact, assay_id)
        targets = {"contact": contact}

        # Accumulate unique code indices (works correctly with DataParallel)
        if "indices" in outputs and outputs["indices"] is not None:
            all_indices.update(outputs["indices"].reshape(-1).unique().cpu().tolist())

        loss, metrics = total_loss(outputs, targets, epoch, cell_idx)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += metrics.get(k, 0.0)
        n_steps += 1

        if n_steps % log_every == 0:
            pbar.set_postfix(
                loss=f"{metrics['total']:.4f}",
                recon=f"{metrics['recon']:.4f}",
                cls=f"{metrics['classifier_acc']:.2f}",
            )

    result = {k: v / max(n_steps, 1) for k, v in running.items()}
    result["active_codes"] = len(all_indices)  # definitive count, unaffected by DataParallel
    return result


@torch.no_grad()
def validate(
    model:  MQVAE,
    loader: DataLoader,
    epoch:  int,
    device: torch.device,
) -> dict:
    model.eval()
    running = {k: 0.0 for k in ["total", "recon", "vq", "classifier", "classifier_acc"]}
    n_steps = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch:02d} [val]  ", leave=False):
        contact  = batch["contact"].to(device)
        assay_id = batch["assay_id"].to(device)
        cell_idx = batch["cell_idx"].to(device)

        outputs = model(contact, assay_id)
        targets = {"contact": contact}
        _, metrics = total_loss(outputs, targets, epoch, cell_idx)

        for k in running:
            running[k] += metrics.get(k, 0.0)
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in running.items()}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_dir: Path,
                    tag: str = "", arch: dict = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"mqvae_epoch{epoch:03d}{('_'+tag) if tag else ''}.pt"
    torch.save({
        "epoch":      epoch,
        "arch":       arch or {},
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "metrics":    metrics,
    }, ckpt_dir / name)
    print(f"  ✓ Checkpoint saved: {name}")


def train(
    cell_lines:    list   = None,
    epochs:        int    = NUM_EPOCHS,
    batch_size:    int    = BATCH_SIZE,
    lr:            float  = LR,
    device_str:    str    = "auto",
    run_name:      str    = "full",
    run_dir:       Path   = None,    # all outputs go here (overrides ckpt_dir/plots_dir)
    ckpt_dir:      Path   = None,
    plots_dir:     Path   = None,
    plot_every:    int    = 1,
    processed_dir: Path   = PROCESSED_DIR,
    h5_path:       Path   = None,    # if set, use combined HDF5 instead of .npz dir
    dat_path:      Path   = None,    # if set, use .dat memmap (fastest) — overrides h5_path
    gpu_ids:       list   = None,    # e.g. [0,1,2,3,4,5,6,7]; None = all available
    # Ablation flags
    use_classifier_head:  bool = True,
    use_masking:          bool = True,
    use_film:             bool = True,
    n_codes:              int  = None,   # None = use config.N_CODES
) -> dict:
    """
    Main training function. Returns dict of final validation metrics.
    All outputs (checkpoints, plots, logs) go into runs/<run_name>/ by default.
    """
    from config import N_CODES as _DEFAULT_N_CODES
    if n_codes is None:
        n_codes = _DEFAULT_N_CODES
    # ── Run directory: everything goes here ──────────────────────────────────
    if run_dir is None:
        run_dir = RUNS_DIR / run_name
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_dir is None:
        ckpt_dir = run_dir / "checkpoints"
    if plots_dir is None:
        plots_dir = run_dir / "plots"
    plots_dir = Path(plots_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] Run directory: {run_dir}", flush=True)

    # ── Validate registry + print status ──────────────────────────────────────
    validated = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
    print_registry_status(validated)

    # ── Device + multi-GPU ────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Multi-GPU: wrap in DataParallel if multiple GPUs requested/available
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_ids is None and n_gpus > 1:
        gpu_ids = list(range(n_gpus))
    use_multi_gpu = (gpu_ids is not None and len(gpu_ids) > 1
                     and torch.cuda.is_available())
    print(f"[train] Device: {device}  |  GPUs: {gpu_ids if use_multi_gpu else 'single'}  |  Run: {run_name}")

    # ── Data ─────────────────────────────────────────────────────────────────────────────
    # Filter to only active (non-skipped) samples
    active = active_samples(validated)
    if cell_lines:
        active = {k: v for k, v in active.items() if k in cell_lines}

    # If cell_lines were given but none matched registry, use them as-is
    # (allows pointing at a custom processed_dir with arbitrary sample names)
    active_cell_lines = list(active.keys()) if active else cell_lines

    # Scale batch size with GPU count so each GPU sees batch_size samples.
    effective_batch = batch_size * len(gpu_ids) if use_multi_gpu else batch_size

    # ── Determine n_cell_types from label map (always use the JSON when present) ──
    label_map = load_cell_line_label_map(processed_dir)
    n_cell_types_from_map = len(label_map) if label_map else None

    if dat_path is not None:
        print(f"[train] Using .dat memmap: {dat_path}")
        meta = np.load(str(dat_path) + "_meta.npz", allow_pickle=True)
        n_train_tiles = int(meta["train_idx"].shape[0]) or int(meta["n_tiles"])
        n_cell_types  = n_cell_types_from_map or (int(meta["cell_idxs"].max()) + 1)
        effective_batch = min(effective_batch, max(batch_size, n_train_tiles // MIN_STEPS_PER_EPOCH))
        train_loader, val_loader = build_dataloaders_dat(
            dat_path    = dat_path,
            batch_size  = effective_batch,
            num_workers = NUM_WORKERS,
        )
    elif h5_path is not None:
        print(f"[train] Using combined HDF5: {h5_path}")
        import h5py as _h5py
        with _h5py.File(str(h5_path), "r") as _hf:
            n_train_tiles = int(_hf["train_idx"].shape[0]) if "train_idx" in _hf \
                            else int(_hf["matrices"].shape[0])
            n_cell_types  = n_cell_types_from_map or (int(_hf["cell_idxs"][:].max()) + 1)
        effective_batch = min(effective_batch, max(batch_size, n_train_tiles // MIN_STEPS_PER_EPOCH))
        train_loader, val_loader = build_dataloaders_h5(
            h5_path     = h5_path,
            batch_size  = effective_batch,
            num_workers = NUM_WORKERS,
        )
    else:
        # Pass cell_lines=None so HiCTileDataset auto-discovers all sample subdirs
        # (dirs are full stems like GM12878_4DNFI..., not clean registry names)
        # If user explicitly requested a subset, honour it; otherwise discover all.
        explicit_cell_lines = cell_lines if cell_lines else None
        train_loader, val_loader = build_dataloaders(
            cell_lines    = explicit_cell_lines,
            batch_size    = effective_batch,
            num_workers   = NUM_WORKERS,
            processed_dir = processed_dir,
        )
        n_cell_types  = n_cell_types_from_map or \
                        (len(active_cell_lines) if active_cell_lines else 16)
        n_train_tiles   = len(train_loader.dataset)
        effective_batch = min(effective_batch, max(batch_size, n_train_tiles // MIN_STEPS_PER_EPOCH))
        if effective_batch != (batch_size * len(gpu_ids) if use_multi_gpu else batch_size):
            train_loader, val_loader = build_dataloaders(
                cell_lines    = explicit_cell_lines,
                batch_size    = effective_batch,
                num_workers   = NUM_WORKERS,
                processed_dir = processed_dir,
            )
    print(f"[train] Cell-line classifier: {n_cell_types} classes  "
          f"(label map: {list(label_map.items())[:5]}{'...' if len(label_map)>5 else ''})",
          flush=True)
    print(f"[train] Effective batch size: {effective_batch} "
          f"({len(train_loader)} steps/epoch)")
    total_steps  = epochs * max(len(train_loader), 1)

    # ── Model ──────────────────────────────────────────────────────────────────────────────────
    base_model = MQVAE(
        n_codes             = n_codes,
        use_classifier_head = use_classifier_head,
        n_cell_types        = n_cell_types,
        use_masking         = use_masking,
        use_film            = use_film,
    ).to(device)

    if use_multi_gpu:
        model = nn.DataParallel(base_model, device_ids=gpu_ids)
        print(f"[train] DataParallel across GPUs: {gpu_ids}")
    else:
        model = base_model

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    arch = {
        "n_codes":             n_codes,
        "use_classifier_head": use_classifier_head,
        "n_cell_types":        n_cell_types,
        "use_masking":         use_masking,
        "use_film":            use_film,
        "n_gpus":              len(gpu_ids) if use_multi_gpu else 1,
        "cell_line_label_map": label_map,
    }
    print(f"[train] Model parameters: {n_params:,}")
    print(f"[train] Codebook: {n_codes} codes  |  Epochs: {epochs}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps // 2),
        total_steps=total_steps,
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard = TrainingDashboard(run_name=run_name, out_dir=plots_dir)

    # ── Training Loop ─────────────────────────────────────────────────────────
    history       = []
    best_val_loss = float("inf")
    best_epoch    = 0
    val_metrics   = {}

    for epoch in range(epochs):
        tau_f = get_tau_f(epoch)
        base_model.set_masker_temperatures(tau_f)
        base_model.reset_epoch_usage()           # reset per-epoch codebook usage counter

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, device
        )
        val_metrics = validate(model, val_loader, epoch, device)
        elapsed     = time.time() - t0

        active_codes = train_metrics.get("active_codes", 0)

        row = {
            "epoch":        epoch,
            "tau_f":        tau_f,
            "active_codes": active_codes,
            "elapsed_s":    round(elapsed, 1),
            **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()},
            **{f"val_{k}":   round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Ep {epoch:02d} | "
            f"train={train_metrics['total']:.4f} "
            f"val={val_metrics['total']:.4f} | "
            f"recon={val_metrics['recon']:.4f} "
            f"cls={val_metrics['classifier_acc']:.2f} "
            f"sil={val_metrics.get('silhouette', 0.0):.3f} | "
            f"codes={active_codes}/{n_codes} τ={tau_f:.2f} [{elapsed:.0f}s]"
        )

        # ── Per-epoch dashboard update ─────────────────────────────────────
        if (epoch + 1) % plot_every == 0:
            fp_emb, fp_labels = collect_fingerprints(base_model, val_loader, device)
            cb_usage          = collect_codebook_usage(base_model)
            sil = compute_silhouette(fp_emb, fp_labels) if len(fp_emb) > 0 else 0.0
            val_metrics["silhouette"] = sil
            dashboard.update(
                epoch          = epoch,
                train_metrics  = train_metrics,
                val_metrics    = val_metrics,
                fp_embeddings  = fp_emb if len(fp_emb) > 0 else None,
                fp_labels      = fp_labels if fp_labels else None,
                codebook_usage = cb_usage if len(cb_usage) > 0 else None,
                tau_f          = tau_f,
            )

        # ── Checkpointing ─────────────────────────────────────────────────
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch    = epoch
            save_checkpoint(
                base_model, optimizer, scheduler, epoch, val_metrics, ckpt_dir,
                tag="best", arch=arch,
            )

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                base_model, optimizer, scheduler, epoch, val_metrics, ckpt_dir, arch=arch
            )

    # Final save
    save_checkpoint(
        base_model, optimizer, scheduler, epochs - 1, val_metrics, ckpt_dir,
        tag="final", arch=arch,
    )

    # Final dashboard update
    fp_emb, fp_labels = collect_fingerprints(base_model, val_loader, device)
    cb_usage          = collect_codebook_usage(base_model)
    sil = compute_silhouette(fp_emb, fp_labels) if len(fp_emb) > 0 else 0.0
    val_metrics["silhouette"] = sil
    dashboard.update(
        epoch          = epochs - 1,
        train_metrics  = train_metrics,
        val_metrics    = val_metrics,
        fp_embeddings  = fp_emb if len(fp_emb) > 0 else None,
        fp_labels      = fp_labels if fp_labels else None,
        codebook_usage = cb_usage if len(cb_usage) > 0 else None,
        tau_f          = get_tau_f(epochs - 1),
    )

    # ── JSON log ──────────────────────────────────────────────────────────────
    log_path = run_dir / f"train_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "run_name":       run_name,
            "best_epoch":     best_epoch,
            "best_val_loss":  best_val_loss,
            "n_params":       n_params,
            "n_codes":        n_codes,
            "epochs":         epochs,
            "history":        history,
        }, f, indent=2)

    print(f"\n[train] Log:       {log_path}")
    print(f"[train] Dashboard: {dashboard.png_path}")
    print(f"[train] Best epoch: {best_epoch}  val_loss: {best_val_loss:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "history":       history,
        "dashboard":     str(dashboard.png_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MQ-VAE Hi-C model")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--run_name",    type=str,   default="full")
    p.add_argument("--cell_lines",  nargs="+",  default=None,
                   help="Subset of cell lines to train on. Defaults to all 5.")
    p.add_argument("--no_classifier",  action="store_true")
    p.add_argument("--no_masking",     action="store_true")
    p.add_argument("--no_film",        action="store_true")
    from config import N_CODES as _DEFAULT_N_CODES
    p.add_argument("--n_codes",       type=int,   default=_DEFAULT_N_CODES)
    p.add_argument("--plot_every",    type=int,   default=1,
                   help="Update dashboard every N epochs (default=1)")
    p.add_argument("--processed_dir", type=Path,  default=PROCESSED_DIR,
                   help="Path to preprocessed .npz tile directory")
    p.add_argument("--h5",            type=Path,  default=None,
                   help="Path to combined tiles.h5 (from combine.py); overrides --processed_dir")
    p.add_argument("--dat",           type=Path,  default=None,
                   help="Path to tiles.dat memmap (from combine.py); fastest option, overrides --h5")
    p.add_argument("--run_dir",        type=Path,  default=None,
                   help="Run root directory; all outputs saved here (default: runs/<run_name>)")
    p.add_argument("--ckpt_dir",      type=Path,  default=None,
                   help="Checkpoint output directory (default: <run_dir>/checkpoints)")
    p.add_argument("--plots_dir",     type=Path,  default=None,
                   help="Dashboard/plots output directory (default: <run_dir>/plots)")
    p.add_argument("--gpu_ids",       nargs="+",  type=int, default=None,
                   help="GPU IDs for DataParallel, e.g. --gpu_ids 0 1 2 3 4 5 6 7")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        cell_lines          = args.cell_lines,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        plot_every          = args.plot_every,
        device_str          = args.device,
        run_name            = args.run_name,
        run_dir             = args.run_dir,
        processed_dir       = args.processed_dir,
        h5_path             = args.h5,
        dat_path            = args.dat,
        ckpt_dir            = args.ckpt_dir,
        plots_dir           = args.plots_dir,
        gpu_ids             = args.gpu_ids,
        use_classifier_head = not args.no_classifier,
        use_masking         = not args.no_masking,
        use_film            = not args.no_film,
        n_codes             = args.n_codes,
    )
