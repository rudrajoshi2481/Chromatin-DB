"""
loss.py — Multi-head loss with auxiliary warmup schedule.

L_total = 1.0  * L_recon
        + 0.25 * L_vq
        + aux_w * 0.5  * L_boundary
        + aux_w * 0.75 * L_compartment

aux_w ramps 0→1 over epochs 5–15 (warmup_epochs=5, ramp_epochs=10).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from config import (
    POS_WEIGHT_BOUNDARY,
    LOSS_W_RECON, LOSS_W_VQ, LOSS_W_BOUNDARY, LOSS_W_COMPARTMENT,
    AUX_WARMUP_EPOCHS, AUX_RAMP_EPOCHS,
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


# ── Auxiliary Weight Schedule ─────────────────────────────────────────────────

def aux_weight(epoch: int, warmup: int = AUX_WARMUP_EPOCHS, ramp: int = AUX_RAMP_EPOCHS) -> float:
    """
    Epochs 0–(warmup-1):        0.0  (reconstruction + VQ only)
    Epochs warmup–(warmup+ramp): linear 0.0 → 1.0
    Epochs warmup+ramp+:        1.0
    """
    if epoch < warmup:
        return 0.0
    elif epoch < warmup + ramp:
        return (epoch - warmup) / ramp
    else:
        return 1.0


# ── Total Loss ────────────────────────────────────────────────────────────────

def total_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    epoch:   int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-head training loss.

    outputs keys: contact_recon, boundary_logits (optional), compartment (optional),
                  z_e, z_q, commit_loss
    targets keys: contact, boundary, compartment

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

    # ── Head 2: TAD Boundary detection ───────────────────────────────────────
    L_boundary    = torch.tensor(0.0, device=device)
    boundary_f1   = 0.0
    has_boundary  = "boundary_logits" in outputs and "boundary" in targets
    if has_boundary:
        bd_logits = outputs["boundary_logits"]     # [B, 256] raw logits
        bd_true   = targets["boundary"]            # [B, 256]
        pos_w     = torch.tensor(POS_WEIGHT_BOUNDARY, device=device)
        L_boundary = F.binary_cross_entropy_with_logits(
            bd_logits, bd_true, pos_weight=pos_w
        )
        # F1 estimate for logging
        with torch.no_grad():
            pred_bin  = (bd_logits.sigmoid() > 0.5).float()
            tp = (pred_bin * bd_true).sum()
            fp = (pred_bin * (1 - bd_true)).sum()
            fn = ((1 - pred_bin) * bd_true).sum()
            boundary_f1 = float(
                (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            )

    # ── Head 3: A/B Compartment score ────────────────────────────────────────
    L_compartment   = torch.tensor(0.0, device=device)
    compartment_r   = 0.0
    has_compartment = "compartment" in outputs and "compartment" in targets
    if has_compartment:
        comp_pred = outputs["compartment"]         # [B, 256]
        comp_true = targets["compartment"]         # [B, 256]
        L_compartment = (
            F.mse_loss(comp_pred, comp_true)
            + 0.5 * (1.0 - pearson_1d(comp_pred, comp_true))
        )
        with torch.no_grad():
            compartment_r = float(pearson_1d(comp_pred, comp_true).item())

    # ── Weighted total ────────────────────────────────────────────────────────
    aux_w = aux_weight(epoch)
    L_total = (
        LOSS_W_RECON        * L_recon
        + LOSS_W_VQ         * L_vq
        + aux_w * LOSS_W_BOUNDARY    * L_boundary
        + aux_w * LOSS_W_COMPARTMENT * L_compartment
    )

    metrics = {
        "total":        float(L_total.item()),
        "recon":        float(L_recon.item()),
        "vq":           float(L_vq.item()),
        "boundary":     float(L_boundary.item()),
        "compartment":  float(L_compartment.item()),
        "boundary_f1":  boundary_f1,
        "compartment_r": compartment_r,
        "aux_weight":   aux_w,
    }

    return L_total, metrics
