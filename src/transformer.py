"""
transformer.py — Transformer Demasker for MQ-VAE.

Takes quantized visible tokens + learnable mask tokens + positional embeddings,
runs 4 Transformer blocks, and outputs a full [B, 1024, 256] sequence that
is reshaped to [B, 256, 32, 32] for the CNN decoder.
"""

import torch
import torch.nn as nn

from config import (
    N_TRANSFORMER_LAYERS, N_HEADS, FFN_DIM, CODE_DIM, SPATIAL_TOKENS,
)
from attention import MultiHeadSelfAttention, FFNBlock


class TransformerBlock(nn.Module):
    """
    Single Transformer block: pre-norm MHSA + pre-norm FFN, both with residuals.
    """

    def __init__(self, d_model: int = CODE_DIM, n_heads: int = N_HEADS,
                 ffn_dim: int = FFN_DIM, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn  = FFNBlock(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]  →  [B, L, d_model]"""
        x = x + self.attn(x)   # residual added here
        x = x + self.ffn(x)
        return x


class TransformerDemasker(nn.Module):
    """
    Reconstructs the full 1024-token sequence from the 512 visible quantized tokens.

    Inputs:
      z_q_visible: [B, K, D]     visible quantized tokens (K=512)
      vis_indices: [B, K]        positions of visible tokens in [0, N)

    Process:
      1. Build full sequence [B, N, D]:
           visible positions ← z_q_visible
           masked positions  ← learnable mask token (broadcast)
      2. Add sinusoidal positional embeddings [N, D]
      3. Pass through N_TRANSFORMER_LAYERS TransformerBlocks
      4. Reshape [B, N, D] → [B, D, sqrt(N), sqrt(N)] = [B, 256, 32, 32]

    Output: [B, 256, 32, 32]
    """

    def __init__(
        self,
        n_tokens: int   = SPATIAL_TOKENS,
        d_model:  int   = CODE_DIM,
        n_layers: int   = N_TRANSFORMER_LAYERS,
        n_heads:  int   = N_HEADS,
        ffn_dim:  int   = FFN_DIM,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens                           # 1024
        self.d_model  = d_model                            # 256
        side          = int(n_tokens ** 0.5)               # 32
        assert side * side == n_tokens, \
            f"n_tokens={n_tokens} must be a perfect square"
        self.side = side

        # Learnable mask token — broadcast over all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_q_visible: torch.Tensor,
        vis_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_q_visible: [B, K, D]
        vis_indices: [B, K]    — positions in [0, N)
        returns:     [B, D, H, W]  where H=W=32
        """
        B, K, D = z_q_visible.shape
        N       = self.n_tokens                            # 1024

        # Build full sequence filled with mask tokens
        seq = self.mask_token.expand(B, N, D).clone()      # [B, N, D]

        # Scatter visible tokens into their positions
        idx_expand = vis_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        seq.scatter_(1, idx_expand, z_q_visible)

        # Add positional embeddings
        seq = seq + self.pos_embed                         # [B, N, D]

        # Transformer blocks
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)                               # [B, N, D]

        # Reshape to spatial feature map: [B, N, D] → [B, D, H, W]
        seq = seq.transpose(1, 2)                          # [B, D, N]
        seq = seq.reshape(B, D, self.side, self.side)      # [B, 256, 32, 32]
        return seq
