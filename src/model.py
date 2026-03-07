"""
model.py — Full MQ-VAE assembly.

Pipeline:
  Encoder → VanillaMasker → EMAVectorQuantizer → TransformerDemasker
         → CNNDecoder → [ContactReconHead, BoundaryHead, CompartmentHead]

forward() returns a dict with all outputs needed by loss.py.
encode_fingerprint() returns [B, 32] fp for database ingestion.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

import config as _cfg
from encoder import Encoder
from masker import VanillaMasker
from codebook import EMAVectorQuantizer
from transformer import TransformerDemasker
from decoder import CNNDecoder
from heads import ContactReconHead, BoundaryHead, CompartmentHead, CellClassifierHead, RegionClassifierHead


class MQVAE(nn.Module):
    """
    Masked Quantized Variational Autoencoder for Hi-C structural fingerprinting.

    Ablation flags:
      use_boundary_head   — include TAD boundary prediction head
      use_compartment_head — include A/B compartment prediction head
      use_masking         — if False, all tokens are passed to VQ (no masking)
      use_film            — if False, assay embedding is zero (ablate FiLM)
    """

    @property
    def ASSAY_TYPES(self):
        return _cfg.ASSAY_TYPES

    def __init__(
        self,
        n_codes:              int   = None,
        code_dim:             int   = None,
        fp_dim:               int   = None,
        keep_ratio:           float = None,
        use_boundary_head:    bool  = True,
        use_compartment_head: bool  = True,
        use_classifier_head:  bool  = True,
        use_region_head:      bool  = True,
        n_cell_types:         int   = None,
        n_chroms:             int   = None,
        n_bins:               int   = None,
        use_masking:          bool  = True,
        use_film:             bool  = True,
        # Explicit architecture overrides (for small-model tests)
        encoder_channels:     list  = None,
        n_transformer_layers: int   = None,
        n_heads:              int   = None,
        ffn_dim:              int   = None,
        decoder_channels:     list  = None,
    ):
        super().__init__()
        self.use_masking          = use_masking
        self.use_film             = use_film
        self.use_boundary_head    = use_boundary_head
        self.use_compartment_head = use_compartment_head
        self.use_classifier_head  = use_classifier_head
        self.use_region_head      = use_region_head

        # Resolve dims — explicit args take precedence over config
        n_codes      = n_codes      if n_codes      is not None else _cfg.N_CODES
        code_dim     = code_dim     if code_dim     is not None else _cfg.CODE_DIM
        fp_dim       = fp_dim       if fp_dim       is not None else _cfg.FP_DIM
        keep_ratio   = keep_ratio   if keep_ratio   is not None else _cfg.KEEP_RATIO
        n_cell_types = n_cell_types if n_cell_types is not None else _cfg.N_CELL_TYPES
        n_chroms     = n_chroms     if n_chroms     is not None else _cfg.N_CHROMS
        n_bins       = n_bins       if n_bins       is not None else _cfg.N_GENOMIC_BINS
        enc_ch     = encoder_channels     or _cfg.ENCODER_CHANNELS
        n_layers   = n_transformer_layers or _cfg.N_TRANSFORMER_LAYERS
        n_h        = n_heads              or _cfg.N_HEADS
        ffn        = ffn_dim              or _cfg.FFN_DIM
        dec_ch     = decoder_channels     or _cfg.DECODER_CHANNELS

        # Store for state_dict rehydration
        self._arch = dict(
            n_codes=n_codes, code_dim=code_dim, fp_dim=fp_dim,
            keep_ratio=keep_ratio, enc_ch=enc_ch, n_layers=n_layers,
            n_h=n_h, ffn=ffn, dec_ch=dec_ch,
            use_boundary_head=use_boundary_head,
            use_compartment_head=use_compartment_head,
            use_classifier_head=use_classifier_head,
            use_region_head=use_region_head,
            n_cell_types=n_cell_types,
            n_chroms=n_chroms, n_bins=n_bins,
            use_masking=use_masking, use_film=use_film,
        )

        self.encoder    = Encoder(channels=enc_ch, out_dim=code_dim)
        self.masker     = VanillaMasker(embed_dim=code_dim, keep_ratio=keep_ratio)
        self.vq         = EMAVectorQuantizer(
            n_codes=n_codes, code_dim=code_dim,
            gamma=_cfg.EMA_GAMMA, dead_threshold=_cfg.DEAD_THRESHOLD,
            revival_interval=_cfg.REVIVAL_INTERVAL, fp_dim=fp_dim,
        )
        self.demasker   = TransformerDemasker(
            n_tokens=_cfg.SPATIAL_TOKENS, d_model=code_dim,
            n_layers=n_layers, n_heads=n_h, ffn_dim=ffn,
        )
        self.decoder    = CNNDecoder(in_channels=code_dim, stage_channels=dec_ch)

        self.contact_head     = ContactReconHead(in_channels=dec_ch[-1])
        self.boundary_head    = BoundaryHead(in_channels=dec_ch[-1])    if use_boundary_head    else None
        self.compartment_head = CompartmentHead(in_channels=dec_ch[-1]) if use_compartment_head else None
        self.classifier_head  = CellClassifierHead(
            in_dim=code_dim, n_classes=n_cell_types
        ) if use_classifier_head else None
        self.region_head      = RegionClassifierHead(
            in_dim=code_dim, n_chroms=n_chroms, n_bins=n_bins
        ) if use_region_head else None

    # ── Temperature control (called by trainer each epoch) ────────────────────

    def set_masker_temperatures(self, tau_f: float) -> None:
        self.masker.set_temperatures(tau_f)

    # ── Main forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        contact: torch.Tensor,
        assay_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        contact:  [B, 1, 256, 256]
        assay_id: [B]  (int64)

        Returns dict with keys:
          contact_recon      [B, 1, 256, 256]
          boundary_logits    [B, 256]          (if use_boundary_head)
          compartment        [B, 256]          (if use_compartment_head)
          z_e                [B, K, D]         encoder outputs at visible positions
          z_q                [B, K, D]         quantized (stopped gradient for loss)
          commit_loss        scalar
          indices            [B, K]
          fingerprint        [B, fp_dim]
        """
        # ── 1. Encode ─────────────────────────────────────────────────────────
        if not self.use_film:
            # Ablation: zero out assay_id effect by using a fixed zero assay
            dummy_id = torch.zeros_like(assay_id)
            z_full = self.encoder(contact, dummy_id)       # [B, 1024, 256]
        else:
            z_full = self.encoder(contact, assay_id)       # [B, 1024, 256]

        # ── 2. Mask ───────────────────────────────────────────────────────────
        if self.use_masking:
            z_vis, vis_idx = self.masker(z_full)           # [B, 512, 256], [B, 512]
        else:
            # Ablation: no masking — use all tokens
            B, N, D = z_full.shape
            vis_idx = torch.arange(N, device=z_full.device).unsqueeze(0).expand(B, -1)
            z_vis = z_full

        # ── 3. Vector Quantize ────────────────────────────────────────────────
        z_q_st, commit_loss, indices = self.vq(z_vis)     # [B, K, 256], scalar, [B, K]

        # ── 4. Fingerprint + cell-type classifier ────────────────────────────
        fingerprint = self.vq.encode_fingerprint(z_q_st)  # [B, fp_dim]
        z_e_mean    = z_vis.mean(dim=1)                    # [B, code_dim] full grad

        # ── 5. Demasker ───────────────────────────────────────────────────────
        feat_3d = self.demasker(z_q_st, vis_idx)          # [B, 256, 32, 32]

        # ── 6. Decode ─────────────────────────────────────────────────────────
        feat_2d = self.decoder(feat_3d)                   # [B, 32, 256, 256]

        # ── 7. Heads ──────────────────────────────────────────────────────────
        contact_recon = self.contact_head(feat_2d)        # [B, 1, 256, 256]

        out = {
            "contact_recon": contact_recon,
            "z_e":           z_vis,                       # encoder outputs
            "z_e_mean":      z_e_mean,                    # [B, D] for classifier
            "z_q":           z_q_st.detach(),             # stopped for loss
            "commit_loss":   commit_loss,
            "indices":       indices,
            "fingerprint":   fingerprint,
        }

        if self.use_boundary_head and self.boundary_head is not None:
            out["boundary_logits"] = self.boundary_head(feat_2d)   # [B, 256]

        if self.use_compartment_head and self.compartment_head is not None:
            out["compartment"] = self.compartment_head(feat_2d)    # [B, 256]

        if self.use_classifier_head and self.classifier_head is not None:
            out["cell_logits"] = self.classifier_head(z_e_mean)    # [B, n_cell_types]

        if self.use_region_head and self.region_head is not None:
            chrom_logits, bin_logits = self.region_head(z_e_mean)
            out["chrom_logits"] = chrom_logits                     # [B, n_chroms]
            out["bin_logits"]   = bin_logits                       # [B, n_bins]

        return out

    # ── Inference-only fingerprint extraction ─────────────────────────────────

    @torch.no_grad()
    def encode_fingerprint(
        self,
        contact: torch.Tensor,
        assay_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficient inference path: encode → mask → VQ → project → fp
        contact:  [B, 1, 256, 256]
        returns:  [B, fp_dim]
        """
        was_training = self.training
        self.eval()

        z_full = self.encoder(contact, assay_id)

        if self.use_masking:
            z_vis, vis_idx = self.masker(z_full)
        else:
            B, N, D = z_full.shape
            vis_idx = torch.arange(N, device=z_full.device).unsqueeze(0).expand(B, -1)
            z_vis   = z_full

        z_q_st, _, _ = self.vq(z_vis)
        fp = self.vq.encode_fingerprint(z_q_st)

        if was_training:
            self.train()
        return fp

    # ── Active code count (logging) ───────────────────────────────────────────

    def active_codes(self) -> int:
        return int((self.vq.usage_count > 0).sum().item())
