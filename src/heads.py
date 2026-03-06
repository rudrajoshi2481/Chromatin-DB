"""
heads.py — Three decoder heads branching from [B, 32, 256, 256].

Head 1: ContactReconHead   → [B, 1, 256, 256]  (MSE + Pearson)
Head 2: BoundaryHead       → [B, 256] raw logits  (BCEWithLogits)
Head 3: CompartmentHead    → [B, 256] in [-1, 1]  (MSE + Pearson)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Head 1: Contact Map Reconstruction ────────────────────────────────────────

class ContactReconHead(nn.Module):
    """
    Predicts the full 256×256 OE contact map.
    Output: raw values (no activation) — target is log2(OE+eps) clipped to [-5,5].
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),   # [B, 1, 256, 256]
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, 32, 256, 256]  →  [B, 1, 256, 256]"""
        return self.head(feat)


# ── Vectorised Diagonal Pooling ────────────────────────────────────────────────

class DiagonalPoolingFast(nn.Module):
    """
    Averages a band of ±band_width diagonals from a 2D feature map
    into a 1D feature vector of length H.

    Uses gather to extract a fixed [B, C, H, 2w+1] band — no variable-length
    diagonal issue.
    """

    def __init__(self, band_width: int = 5):
        super().__init__()
        self.w = band_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]  (H == W)
        returns: [B, C, H]
        """
        B, C, H, W = x.shape
        w = self.w
        # Pad W dimension only (left and right) so offset diagonals stay in bounds
        padded = F.pad(x, (w, w, 0, 0))              # [B, C, H, W+2w]

        # For each row i, gather columns [i, i+1, ..., i+2w] from padded
        # col_offset[i, k] = i + k  for k in [0, 2w]
        row_idx = torch.arange(H, device=x.device)                 # [H]
        col_offsets = torch.arange(2 * w + 1, device=x.device)     # [2w+1]
        gather_cols = (row_idx.unsqueeze(1) + col_offsets.unsqueeze(0))  # [H, 2w+1]

        # Expand for broadcast: [1, 1, H, 2w+1] → [B, C, H, 2w+1]
        gather_idx = gather_cols.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        band = padded.gather(3, gather_idx)           # [B, C, H, 2w+1]
        return band.mean(dim=-1)                      # [B, C, H]


# ── Head 2: TAD Boundary Detection ────────────────────────────────────────────

class BoundaryHead(nn.Module):
    """
    Predicts binary TAD boundary labels along the diagonal.
    Output: raw logits [B, 256]  — use BCEWithLogitsLoss externally.
    NO Sigmoid here.
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.diagonal_pool = DiagonalPoolingFast(band_width=5)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            # NO Sigmoid — BCEWithLogitsLoss handles this
        )

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        feat_2d: [B, 32, 256, 256]  →  [B, 256] raw logits
        """
        diag_feat = self.diagonal_pool(feat_2d)    # [B, 32, 256]
        return self.head(diag_feat).squeeze(1)     # [B, 256]


# ── Head 3: A/B Compartment Score ─────────────────────────────────────────────

class CompartmentHead(nn.Module):
    """
    Predicts the A/B compartment E1 eigenvector per bin.
    Output: [B, 256] in [-1, 1]  (Tanh activation matches phased E1 range).
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.row_pool = nn.AdaptiveAvgPool2d((256, 1))
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=11, padding=5),
            nn.GELU(),
            nn.Conv1d(16, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
            nn.Tanh(),   # output in [-1, 1] matching normalised E1 eigenvector
        )

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        feat_2d: [B, 32, 256, 256]  →  [B, 256]
        """
        pooled = self.row_pool(feat_2d).squeeze(-1)   # [B, 32, 256]
        return self.head(pooled).squeeze(1)            # [B, 256]
