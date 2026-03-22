"""
loss.py — Streamlined loss for MQ-VAE structural fingerprinting.

L_total = LOSS_W_RECON      * L_recon
        + LOSS_W_VQ         * L_vq
        + LOSS_W_CLASSIFIER * L_classifier

No boundary, compartment, or region losses.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from config import (
    LOSS_W_RECON, LOSS_W_VQ, LOSS_W_CLASSIFIER,
)


# ── Pearson Correlation Helpers ───────────────────────────────────────────────

def pearson_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise Pearson correlation for 1D signals.
    x, y: [B, L]  →  scalar (mean over batch)
    """
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    num = (x_c * y_c).sum(dim=-1)
    den = torch.sqrt((x_c ** 2).sum(dim=-1) * (y_c ** 2).sum(dim=-1) + 1e-8)
    return (num / den).mean()


def pearson_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise Pearson for 2D maps (flatten spatial dims).
    x, y: [B, 1, H, W]  →  scalar
    """
    return pearson_1d(x.flatten(1), y.flatten(1))


# ── Total Loss ───────────────────────────────────────────────────────────────────

def total_loss(
    outputs:  Dict[str, torch.Tensor],
    targets:  Dict[str, torch.Tensor],
    epoch:    int,
    cell_idx: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute streamlined training loss: reconstruction + VQ + cell classifier.

    outputs keys: contact_recon, z_e, z_q, cell_logits (optional)
    targets keys: contact

    Returns (loss_tensor, metrics_dict)
    """
    contact_recon = outputs["contact_recon"]       # [B, 1, 256, 256]
    contact_true  = targets["contact"]             # [B, 1, 256, 256]
    device        = contact_recon.device

    # ── Head 1: Contact reconstruction ───────────────────────────────────────
    L_recon = (
        F.mse_loss(contact_recon, contact_true)
        + 0.5 * (1.0 - pearson_2d(contact_recon, contact_true))
    )

    # ── VQ commitment loss ────────────────────────────────────────────────────
    # Use pre-computed commit_loss from VQ module if available
    if "commit_loss" in outputs:
        L_vq = outputs["commit_loss"]
    else:
        z_e = outputs["z_e"]
        z_q = outputs["z_q"]
        L_vq = F.mse_loss(z_e, z_q.detach())

    # ── Head 2: Cell-type patch classifier ────────────────────────────────────────
    L_classifier  = torch.tensor(0.0, device=device)
    classifier_acc = 0.0
    has_classifier = (
        "cell_logits" in outputs
        and cell_idx is not None
        and LOSS_W_CLASSIFIER > 0
    )
    if has_classifier:
        logits = outputs["cell_logits"]          # [B, n_cell_types]
        L_classifier = F.cross_entropy(logits, cell_idx)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            classifier_acc = float((preds == cell_idx).float().mean().item())

    # ── Weighted total ─────────────────────────────────────────────────────────────────────
    L_total = (
        LOSS_W_RECON      * L_recon
        + LOSS_W_VQ       * L_vq
        + LOSS_W_CLASSIFIER * L_classifier
    )

    metrics = {
        "total":          float(L_total.item()),
        "recon":          float(L_recon.item()),
        "vq":             float(L_vq.item()),
        "classifier":     float(L_classifier.item()),
        "classifier_acc": classifier_acc,
    }

    return L_total, metrics
