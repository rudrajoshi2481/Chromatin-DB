"""
codebook.py — EMA Vector Quantizer with dead code revival and fingerprint projection.

VQ lookup:      z_e [B, N, D] → z_q [B, N, D]  (nearest codebook entry)
EMA update:     γ=0.99, Laplace smoothing
Dead revival:   replace unused codes with random encoder outputs + noise
Fingerprint:    mean(z_q, dim=1) → Linear(256→32) → [B, 32]
Histogram:      bincount(indices) → L1-norm → [B, n_codes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import N_CODES, CODE_DIM, EMA_GAMMA, DEAD_THRESHOLD, REVIVAL_INTERVAL, FP_DIM, USE_ATTENTION_POOLING


class EMAVectorQuantizer(nn.Module):
    """
    EMA-updated Vector Quantizer (no codebook gradient needed).

    forward returns:
      z_q_st:      [B, N, D]  straight-through quantized tokens
      commit_loss: scalar     MSE(z_e, sg(z_q))
      indices:     [B, N]     codebook assignment indices
    """

    def __init__(
        self,
        n_codes:          int   = N_CODES,
        code_dim:         int   = CODE_DIM,
        gamma:            float = EMA_GAMMA,
        dead_threshold:   int   = DEAD_THRESHOLD,
        revival_interval: int   = REVIVAL_INTERVAL,
        fp_dim:           int   = FP_DIM,
    ):
        super().__init__()
        self.n_codes          = n_codes
        self.code_dim         = code_dim
        self.gamma            = gamma
        self.dead_threshold   = dead_threshold
        self.revival_interval = revival_interval

        # Codebook and EMA accumulators (not model parameters — updated manually)
        self.register_buffer("codebook",    torch.randn(n_codes, code_dim))
        self.register_buffer("ema_count",   torch.ones(n_codes) * 1e-5)
        self.register_buffer("ema_sum",     torch.randn(n_codes, code_dim))
        self.register_buffer("usage_count", torch.zeros(n_codes))
        self.register_buffer("step",        torch.tensor(0, dtype=torch.long))

        # Normalise initial codebook
        with torch.no_grad():
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)

        # Fingerprint projection: 256 → fp_dim (32)
        self.fp_proj = nn.Linear(code_dim, fp_dim)

        # Attention pooling (alternative to mean pooling)
        self.use_attention_pool = USE_ATTENTION_POOLING
        if self.use_attention_pool:
            self.attn_pool = nn.Sequential(
                nn.Linear(code_dim, code_dim // 2),
                nn.Tanh(),
                nn.Linear(code_dim // 2, 1)
            )
            # Separate projection for attention-pooled features
            self.fp_proj_attn = nn.Linear(code_dim, fp_dim)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e: [B, N, D]   — encoder outputs (visible tokens after masker)
        """
        B, N, D = z_e.shape
        flat = z_e.reshape(-1, D)                           # [B*N, D]

        # Nearest-neighbour lookup via squared L2 distance
        dists   = torch.cdist(flat.float(), self.codebook.float())  # [B*N, n_codes]
        indices = dists.argmin(dim=-1)                      # [B*N]
        z_q     = self.codebook[indices].reshape(B, N, D)  # [B, N, D]

        if self.training:
            self._ema_update(flat.detach(), indices.detach())
            self.step += 1
            if self.step % self.revival_interval == 0:
                self._revive_dead_codes(flat.detach())

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()                # [B, N, D]

        # Commitment loss: push encoder outputs toward (stopped) codebook entries
        commit_loss = F.mse_loss(z_e, z_q.detach())

        return z_q_st, commit_loss, indices.reshape(B, N)

    # ── EMA Update ─────────────────────────────────────────────────────────────

    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook entries via exponential moving average."""
        onehot = F.one_hot(indices, self.n_codes).float()   # [B*N, n_codes]
        counts = onehot.sum(0)                               # [n_codes]
        sums   = onehot.T @ flat                             # [n_codes, D]

        self.ema_count = self.gamma * self.ema_count + (1.0 - self.gamma) * counts
        self.ema_sum   = self.gamma * self.ema_sum   + (1.0 - self.gamma) * sums

        # Laplace smoothing to avoid division by zero
        n         = self.ema_count.sum()
        smoothed  = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n
        self.codebook.data = self.ema_sum / smoothed.unsqueeze(-1)

        self.usage_count += counts

    # ── Dead Code Revival ──────────────────────────────────────────────────────

    def _revive_dead_codes(self, flat: torch.Tensor) -> None:
        """Replace rarely-used codes with perturbed encoder outputs."""
        dead   = self.usage_count < self.dead_threshold
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            n_flat = flat.size(0)
            if n_flat < n_dead:
                # Repeat flat if not enough samples
                repeat = (n_dead // n_flat) + 1
                flat   = flat.repeat(repeat, 1)
            perm = torch.randperm(flat.size(0), device=flat.device)[:n_dead]
            self.codebook.data[dead] = (
                flat[perm] + torch.randn_like(flat[perm]) * 0.01
            )
            self.usage_count[dead] = float(self.dead_threshold)
        self.usage_count.zero_()                             # reset for next interval

    # ── Fingerprint Extraction ─────────────────────────────────────────────────

    def encode_fingerprint(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Extract per-window 32-dim fingerprint for database storage.
        z_e: [B, N, D]  →  [B, fp_dim]

        Uses attention pooling if enabled, otherwise mean pooling.
        Attention pooling learns which spatial positions matter most.
        """
        if self.use_attention_pool:
            # Attention pooling: learn importance weights per token
            # z_e: [B, N, D]
            attn_weights = self.attn_pool(z_e)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)  # normalize across tokens
            pooled = (attn_weights * z_e).sum(dim=1)  # [B, D] weighted sum
            return self.fp_proj_attn(pooled)  # [B, fp_dim]
        else:
            # Mean pooling (original)
            mean_z = z_e.mean(dim=1)  # [B, D]
            return self.fp_proj(mean_z)  # [B, fp_dim]

    def encode_histogram(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Extract L1-normalised code-usage histogram for sample-level fingerprint.
        indices: [B, N]  →  [B, n_codes]
        """
        B = indices.size(0)
        hist = torch.zeros(B, self.n_codes, device=indices.device)
        for b in range(B):
            hist[b] = torch.bincount(
                indices[b].long(), minlength=self.n_codes
            ).float()
        return F.normalize(hist, p=1, dim=-1)               # [B, n_codes]

    # ── Codebook Perplexity ────────────────────────────────────────────────────

    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Effective codebook utilization metric.
        indices: [B, N]  →  scalar
        """
        flat = indices.reshape(-1).long()
        probs = torch.bincount(flat, minlength=self.n_codes).float()
        probs = probs / (probs.sum() + 1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        return torch.exp(entropy)
