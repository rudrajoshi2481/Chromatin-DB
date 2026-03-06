"""
attention.py — Multi-Head Self-Attention block used by the Transformer demasker.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import N_HEADS, CODE_DIM


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention with pre-LayerNorm.
    d_model must be divisible by n_heads.
    d_k = d_model // n_heads  (e.g., 256 // 8 = 32)
    """

    def __init__(self, d_model: int = CODE_DIM, n_heads: int = N_HEADS, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.norm   = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        self._scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        returns: [B, L, d_model]  (residual NOT added here — done in TransformerBlock)
        """
        B, L, _ = x.shape
        h = self.norm(x)

        def split_heads(t):
            # [B, L, d_model] → [B, n_heads, L, d_k]
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.q_proj(h))   # [B, H, L, d_k]
        K = split_heads(self.k_proj(h))
        V = split_heads(self.v_proj(h))

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # [B, H, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                                 # [B, H, L, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(out)                                        # [B, L, d_model]


class FFNBlock(nn.Module):
    """
    Position-wise Feed-Forward Network with pre-LayerNorm.
    d_model → ffn_dim → d_model  (GELU activation)
    """

    def __init__(self, d_model: int = CODE_DIM, ffn_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, ffn_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]  (residual NOT added here)"""
        h = self.act(self.fc1(self.norm(x)))
        return self.drop(self.fc2(h))
