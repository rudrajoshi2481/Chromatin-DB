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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINTS_DIR, TRASH_DIR, PLOTS_DIR, CELL_LINE_REGISTRY,
    BATCH_SIZE, NUM_EPOCHS, LR, BETAS, WEIGHT_DECAY,
    GRAD_CLIP, WARMUP_STEPS, NUM_WORKERS, SEED,
    LOG_EVERY_N_STEPS, SAVE_EVERY_N_EPOCHS,
    MCOOL_DIR, CCRE_REGISTRY,
)
from dataset import build_dataloaders
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
    running = {k: 0.0 for k in ["total","recon","vq","compartment","classifier","classifier_acc","compartment_r"]}
    n_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for batch in pbar:
        contact     = batch["contact"].to(device)        # [B, 1, 256, 256]
        boundary    = batch["boundary"].to(device)       # [B, 256]
        compartment = batch["compartment"].to(device)    # [B, 256]
        assay_id    = batch["assay_id"].to(device)       # [B]
        cell_idx    = batch["cell_idx"].to(device)       # [B] unique int per cell line

        outputs = model(contact, assay_id)
        targets = {"contact": contact, "boundary": boundary, "compartment": compartment}

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

    return {k: v / max(n_steps, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model:  MQVAE,
    loader: DataLoader,
    epoch:  int,
    device: torch.device,
) -> dict:
    model.eval()
    running = {k: 0.0 for k in ["total","recon","vq","compartment","classifier","classifier_acc","compartment_r"]}
    n_steps = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch:02d} [val]  ", leave=False):
        contact     = batch["contact"].to(device)
        boundary    = batch["boundary"].to(device)
        compartment = batch["compartment"].to(device)
        assay_id    = batch["assay_id"].to(device)
        cell_idx    = batch["cell_idx"].to(device)

        outputs = model(contact, assay_id)
        targets = {"contact": contact, "boundary": boundary, "compartment": compartment}
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
    cell_lines=None,
    epochs:     int   = NUM_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LR,
    device_str: str   = "auto",
    run_name:   str   = "full",
    ckpt_dir:   Path  = None,
    plot_every: int   = 1,      # update dashboard every N epochs
    # Ablation flags
    use_boundary_head:    bool = True,
    use_compartment_head: bool = True,
    use_classifier_head:  bool = True,
    use_masking:          bool = True,
    use_film:             bool = True,
    n_codes:              int  = 512,
) -> dict:
    """
    Main training function. Returns dict of final validation metrics.
    Writes dashboard PNG + JSON log every epoch.
    """
    if ckpt_dir is None:
        ckpt_dir = CHECKPOINTS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Validate registry + print status ──────────────────────────────────────
    validated = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
    print_registry_status(validated)

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] Device: {device}  |  Run: {run_name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Filter to only active (non-skipped) samples
    active = active_samples(validated)
    if cell_lines:
        active = {k: v for k, v in active.items() if k in cell_lines}

    active_cell_lines = list(active.keys()) if active else cell_lines
    train_loader, val_loader = build_dataloaders(
        cell_lines=active_cell_lines,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )
    total_steps  = epochs * max(len(train_loader), 1)
    n_cell_types = len(active_cell_lines) if active_cell_lines else 16

    # ── Model ───────────────────────────────────────────────────────────────────────
    model = MQVAE(
        n_codes              = n_codes,
        use_boundary_head    = use_boundary_head,
        use_compartment_head = use_compartment_head,
        use_classifier_head  = use_classifier_head,
        n_cell_types         = n_cell_types,
        use_masking          = use_masking,
        use_film             = use_film,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = {
        "n_codes":              n_codes,
        "use_boundary_head":    use_boundary_head,
        "use_compartment_head": use_compartment_head,
        "use_classifier_head":  use_classifier_head,
        "n_cell_types":         n_cell_types,
        "use_masking":          use_masking,
        "use_film":             use_film,
    }
    print(f"[train] Model parameters: {n_params:,}")
    print(f"[train] Codebook: {n_codes} codes  |  Epochs: {epochs}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=min(WARMUP_STEPS, total_steps // 2),
        total_steps=total_steps,
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard = TrainingDashboard(run_name=run_name, out_dir=PLOTS_DIR)

    # ── Training Loop ─────────────────────────────────────────────────────────
    history       = []
    best_val_loss = float("inf")
    best_epoch    = 0
    val_metrics   = {}

    for epoch in range(epochs):
        tau_f = get_tau_f(epoch)
        model.set_masker_temperatures(tau_f)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, device
        )
        val_metrics = validate(model, val_loader, epoch, device)
        elapsed     = time.time() - t0

        active_codes = model.active_codes()
        train_metrics["active_codes"] = active_codes

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
            fp_emb, fp_labels = collect_fingerprints(model, val_loader, device)
            cb_usage          = collect_codebook_usage(model)
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
                model, optimizer, scheduler, epoch, val_metrics, ckpt_dir,
                tag="best", arch=arch,
            )

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, ckpt_dir, arch=arch
            )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, epochs - 1, val_metrics, ckpt_dir,
        tag="final", arch=arch,
    )

    # Final dashboard update
    fp_emb, fp_labels = collect_fingerprints(model, val_loader, device)
    cb_usage          = collect_codebook_usage(model)
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
    log_path = TRASH_DIR / f"train_log_{run_name}.json"
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
    p.add_argument("--no_boundary",    action="store_true")
    p.add_argument("--no_compartment", action="store_true")
    p.add_argument("--no_classifier",  action="store_true")
    p.add_argument("--no_masking",     action="store_true")
    p.add_argument("--no_film",        action="store_true")
    p.add_argument("--n_codes",     type=int,   default=512)
    p.add_argument("--plot_every",  type=int,   default=1,
                   help="Update dashboard every N epochs (default=1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        cell_lines           = args.cell_lines,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        plot_every           = args.plot_every,
        device_str           = args.device,
        run_name             = args.run_name,
        use_boundary_head    = not args.no_boundary,
        use_compartment_head = not args.no_compartment,
        use_classifier_head  = not args.no_classifier,
        use_masking          = not args.no_masking,
        use_film             = not args.no_film,
        n_codes              = args.n_codes,
    )
