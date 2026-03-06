"""
masker.py — Decoupled Straight-Through Gumbel-Softmax Masker.

Separates forward temperature (τ_f, sharpness of selection) from
backward temperature (τ_b = τ_f × 4, gradient dispersion).

Training:  stochastic top-K via Gumbel perturbation
Inference: deterministic hard top-K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KEEP_RATIO, TAU_F_WARMUP, TAU_F_FINAL, TAU_F_WARMUP_EPOCHS, TAU_F_ANNEAL_EPOCHS


def get_tau_f(epoch: int) -> float:
    """
    Temperature schedule for forward pass:
      Epochs 0–(W-1):    τ_f = TAU_F_WARMUP   (1.0)
      Epochs W–(W+A-1):  linear anneal 1.0 → 0.3
      Epochs W+A+:       τ_f = TAU_F_FINAL     (0.3)
    τ_b = τ_f × 4 always (set inside set_temperatures).
    """
    W = TAU_F_WARMUP_EPOCHS
    A = TAU_F_ANNEAL_EPOCHS
    if epoch < W:
        return float(TAU_F_WARMUP)
    elif epoch < W + A:
        frac = (epoch - W) / A
        return float(TAU_F_WARMUP - frac * (TAU_F_WARMUP - TAU_F_FINAL))
    else:
        return float(TAU_F_FINAL)


class VanillaMasker(nn.Module):
    """
    Learns to select the K most informative spatial tokens.

    Input:  tokens [B, N, D]  (N=1024 spatial positions, D=256)
    Output: visible_tokens [B, K, D], topk_idx [B, K]
            K = int(N * keep_ratio) = 512
    """

    def __init__(self, embed_dim: int = 256, keep_ratio: float = KEEP_RATIO):
        super().__init__()
        self.keep_ratio = keep_ratio

        # Score each token: D → 64 → 1
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.register_buffer("tau_f", torch.tensor(float(TAU_F_WARMUP)))
        self.register_buffer("tau_b", torch.tensor(float(TAU_F_WARMUP) * 4.0))

    def set_temperatures(self, tau_f: float) -> None:
        """Called by trainer each epoch. τ_b = τ_f × 4 always."""
        self.tau_f.fill_(tau_f)
        self.tau_b.fill_(tau_f * 4.0)

    def forward(
        self, tokens: torch.Tensor
    ):
        """
        tokens: [B, N, D]
        returns:
          visible_tokens: [B, K, D]
          topk_idx:       [B, K]   (indices into N positions)
        """
        B, N, D = tokens.shape
        K = int(N * self.keep_ratio)                           # 512
        scores = self.scorer(tokens).squeeze(-1)               # [B, N]

        if self.training:
            # Sample Gumbel noise once; reuse for both forward and backward
            gumbel = -torch.log(
                -torch.log(
                    torch.rand_like(scores).clamp(min=1e-10, max=1.0 - 1e-10)
                )
            )

            # ── Forward: sharp selection ─────────────────────────────
            perturbed_f = (scores + gumbel) / self.tau_f       # [B, N]
            topk_idx    = perturbed_f.topk(K, dim=-1).indices  # [B, K]
            hard_mask   = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)

            # ── Backward: smooth gradient path ───────────────────────
            perturbed_b = (scores + gumbel) / self.tau_b
            soft_mask   = torch.softmax(perturbed_b, dim=-1) * N  # scale to ~1 per token
            # Straight-through: forward uses hard_mask, backward flows through soft_mask
            mask = hard_mask + (soft_mask - soft_mask.detach())    # [B, N]

        else:
            # Inference: deterministic hard top-K (no Gumbel, no temperature)
            topk_idx = scores.topk(K, dim=-1).indices              # [B, K]
            mask     = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)

        # Gather the K visible tokens in order of their original positions
        sorted_idx   = topk_idx.sort(dim=-1).values               # [B, K] sorted
        visible_idx  = sorted_idx.unsqueeze(-1).expand(-1, -1, D) # [B, K, D]
        visible_tokens = tokens.gather(1, visible_idx)             # [B, K, D]

        return visible_tokens, sorted_idx
