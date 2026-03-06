"""
decoder.py — Shared CNN Decoder Backbone.

Takes [B, 256, 32, 32] from Transformer Demasker and upsamples to [B, 32, 256, 256].
Three ResBlock+Upsample stages. Output feeds all three decoder heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DECODER_CHANNELS, CODE_DIM
from encoder import ResBlock


class UpsampleBlock(nn.Module):
    """ResBlock followed by bilinear 2× upsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.resblock  = ResBlock(in_ch, out_ch, stride=1)
        self.upsample  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.resblock(x))


class CNNDecoder(nn.Module):
    """
    Shared decoder backbone:
      [B, 256, 32, 32] → [B, 128, 64, 64]
                       → [B, 64, 128, 128]
                       → [B, 32, 256, 256]   ← branch point for all 3 heads
    """

    def __init__(
        self,
        in_channels:     int  = CODE_DIM,         # 256
        stage_channels: list = None,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = DECODER_CHANNELS      # [128, 64, 32]

        self.up1 = UpsampleBlock(in_channels,         stage_channels[0])
        self.up2 = UpsampleBlock(stage_channels[0],   stage_channels[1])
        self.up3 = UpsampleBlock(stage_channels[1],   stage_channels[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256, 32, 32]  →  [B, 32, 256, 256]
        """
        x = self.up1(x)   # [B, 128, 64, 64]
        x = self.up2(x)   # [B, 64, 128, 128]
        x = self.up3(x)   # [B, 32, 256, 256]
        return x
