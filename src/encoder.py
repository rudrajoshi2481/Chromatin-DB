"""
encoder.py — CNN Encoder with FiLM assay conditioning and ResBlocks.

Architecture:
  [B, 1, 256, 256]
    Stage1: Conv(1→32, k=7, s=2) + GN + GELU  → [B, 32, 128, 128]
    FiLM conditioning
    Stage2: ResBlock(32→64, s=2)               → [B, 64, 64, 64]
    Stage3: ResBlock(64→128, s=2)              → [B, 128, 32, 32]
    Stage4: ResBlock(128→256, s=1)             → [B, 256, 32, 32]
  Reshape: [B, 256, 32, 32] → [B, 1024, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ASSAY_TYPES, ASSAY_EMBED_DIM, ENCODER_CHANNELS


class ResBlock(nn.Module):
    """
    Residual block with optional stride for downsampling.
    Uses GroupNorm (8 groups) and GELU activation.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.GELU()

        # Projection shortcut when dimensions change
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for assay-type conditioning.
    features = features * (1 + gamma) + beta
    Initialized near identity (gamma/beta weights = 0).
    """

    def __init__(self, assay_embed_dim: int = 8, feature_channels: int = 32):
        super().__init__()
        self.gamma_proj = nn.Linear(assay_embed_dim, feature_channels)
        self.beta_proj  = nn.Linear(assay_embed_dim, feature_channels)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, features: torch.Tensor, assay_embed: torch.Tensor) -> torch.Tensor:
        # features:    [B, C, H, W]
        # assay_embed: [B, assay_embed_dim]
        gamma = self.gamma_proj(assay_embed).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = self.beta_proj(assay_embed).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        return features * (1.0 + gamma) + beta


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with num_groups = min(8, ch) — safe for small channel counts."""
    return nn.GroupNorm(min(8, ch), ch)


class Encoder(nn.Module):
    """
    Four-stage CNN encoder with FiLM assay conditioning after Stage 1.

    Output: [B, SPATIAL_TOKENS, out_dim] — spatial token sequence.
    out_dim defaults to channels[-1]; set explicitly for small-model overrides.
    """

    ASSAY_TYPES = ASSAY_TYPES

    def __init__(
        self,
        assay_embed_dim: int  = ASSAY_EMBED_DIM,
        channels:        list = None,
        out_dim:         int  = None,
    ):
        super().__init__()
        if channels is None:
            channels = ENCODER_CHANNELS   # [32, 64, 128, 256]
        if out_dim is None:
            out_dim = channels[-1]

        self.assay_embed = nn.Embedding(len(ASSAY_TYPES), assay_embed_dim)
        self.film        = FiLMLayer(assay_embed_dim, channels[0])

        # Stage 1: [B, 1, 256, 256] → [B, C0, 128, 128]
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
        )

        # Stage 2: → [B, C1, 64, 64]
        self.stage2 = ResBlock(channels[0], channels[1], stride=2)

        # Stage 3: → [B, C2, 32, 32]
        self.stage3 = ResBlock(channels[1], channels[2], stride=2)

        # Stage 4: → [B, C3, 32, 32]
        self.stage4 = ResBlock(channels[2], channels[3], stride=1)

        # Optional projection to out_dim when channels[-1] != out_dim
        self.proj = (
            nn.Conv2d(channels[3], out_dim, 1, bias=False)
            if channels[3] != out_dim else nn.Identity()
        )
        self._out_dim = out_dim

    def forward(self, x: torch.Tensor, assay_id: torch.Tensor) -> torch.Tensor:
        """
        x:        [B, 1, 256, 256]
        assay_id: [B]
        returns:  [B, H*W, out_dim]   (H=W=32, so 1024 tokens)
        """
        ae = self.assay_embed(assay_id)    # [B, embed_dim]
        z  = self.stage1(x)
        z  = self.film(z, ae)
        z  = self.stage2(z)
        z  = self.stage3(z)
        z  = self.stage4(z)
        z  = self.proj(z)                  # [B, out_dim, 32, 32]

        B, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1)          # [B, H, W, C]
        z = z.reshape(B, H * W, C)         # [B, 1024, out_dim]
        return z
