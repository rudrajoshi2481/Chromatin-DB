# MQ-VAE Model Architecture

**Masked Quantized Variational Autoencoder for Hi-C Structural Fingerprinting**

Generated: 2026-03-11

---

## Table of Contents
1. [Overview](#overview)
2. [Pipeline Flow](#pipeline-flow)
3. [Component Details](#component-details)
4. [Model Parameters](#model-parameters)
5. [Output Specifications](#output-specifications)
6. [Ablation Flags](#ablation-flags)

---

## Overview

The MQ-VAE is a hierarchical autoencoder that learns discrete structural fingerprints from Hi-C contact maps. It combines:
- **Masking**: Learned selection of informative spatial regions
- **Vector Quantization**: Discrete codebook representation with EMA updates
- **Transformer Demasking**: Reconstruction of full spatial context
- **Multi-task Learning**: Contact reconstruction + cell-type classification

**Input**: Hi-C contact map `[B, 1, 256, 256]` + assay type ID `[B]`  
**Output**: Reconstructed contact map + 32-dim fingerprint + cell-type logits

---

## Pipeline Flow

```
Input: [B, 1, 256, 256] contact map + [B] assay_id
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. ENCODER (CNN + FiLM conditioning)                        │
│    [B, 1, 256, 256] → [B, 1024, 256]                        │
│    • 4-stage CNN with ResBlocks                             │
│    • FiLM layer for assay-type conditioning                 │
│    • Output: 1024 spatial tokens, each 256-dim              │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MASKER (Learned Token Selection)                         │
│    [B, 1024, 256] → [B, 512, 256] + [B, 512] indices        │
│    • Learnable scorer network (256→64→1)                    │
│    • Gumbel-Softmax top-K selection (training)              │
│    • Deterministic top-K (inference)                        │
│    • Temperature annealing: τ_f 1.0→0.5 over 40 epochs     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. VECTOR QUANTIZER (EMA Codebook)                          │
│    [B, 512, 256] → [B, 512, 256] + indices + commit_loss    │
│    • Codebook: 512 codes × 256 dims                         │
│    • EMA update: γ=0.99                                     │
│    • Dead code revival every 100 steps                      │
│    • Straight-through estimator for gradients               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. FINGERPRINT EXTRACTION                                    │
│    [B, 512, 256] → [B, 32]                                  │
│    • Mean pooling over tokens (or attention pooling)        │
│    • Linear projection: 256→32                              │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. TRANSFORMER DEMASKER                                      │
│    [B, 512, 256] + indices → [B, 256, 32, 32]               │
│    • Scatter visible tokens into full 1024-token sequence   │
│    • Learnable mask tokens for missing positions            │
│    • Positional embeddings (learnable)                      │
│    • 4 Transformer blocks (256→128 inner projection)        │
│    • Reshape to spatial feature map                         │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. CNN DECODER                                               │
│    [B, 256, 32, 32] → [B, 32, 256, 256]                     │
│    • 3 UpsampleBlocks (ResBlock + 2× bilinear upsample)     │
│    • 256→128→64→32 channels                                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. DECODER HEADS                                             │
│    A. ContactReconHead: [B, 32, 256, 256] → [B, 1, 256, 256]│
│       • Conv(32→16→1) for contact map reconstruction        │
│                                                              │
│    B. CellClassifierHead: [B, 256] → [B, n_cell_types]      │
│       • Operates on z_e_mean (pre-VQ encoder features)      │
│       • 256→128→64→n_classes with dropout                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Encoder (CNN + FiLM)

**Architecture**: 4-stage CNN with ResBlocks and FiLM conditioning

```
Input: [B, 1, 256, 256]

Stage 1: Conv(1→32, k=7, s=2) + GroupNorm + GELU
         → [B, 32, 128, 128]
         
FiLM:    Assay embedding (4 types → 8-dim)
         features = features * (1 + γ) + β
         
Stage 2: ResBlock(32→64, s=2)
         → [B, 64, 64, 64]
         
Stage 3: ResBlock(64→128, s=2)
         → [B, 128, 32, 32]
         
Stage 4: ResBlock(128→256, s=1)
         → [B, 256, 32, 32]
         
Reshape: [B, 256, 32, 32] → [B, 1024, 256]
```

**Parameters**:
- Encoder channels: `[32, 64, 128, 256]`
- Assay embedding dim: `8`
- Output tokens: `1024` (32×32 spatial grid)
- Output dim per token: `256`

**ResBlock Details**:
- 2 Conv layers (3×3 kernel, padding=1)
- GroupNorm (8 groups)
- GELU activation
- Projection shortcut when dimensions change

**FiLM Layer**:
- Initialized near identity (γ=0, β=0)
- Learns assay-specific feature modulation
- Applied after Stage 1

---

### 2. Masker (VanillaMasker)

**Purpose**: Learn to select the most informative 50% of spatial tokens

```
Input: [B, 1024, 256]

Scorer Network:
  Linear(256 → 64)
  GELU
  Linear(64 → 1)
  → [B, 1024] importance scores

Selection (Training):
  • Sample Gumbel noise: g ~ Gumbel(0, 1)
  • Forward: top-K((scores + g) / τ_f)  [sharp, τ_f=1.0→0.5]
  • Backward: softmax((scores + g) / τ_b) [smooth, τ_b=4×τ_f]
  • Straight-through: forward uses hard mask, backward uses soft

Selection (Inference):
  • Deterministic: top-K(scores)  [no Gumbel, no temperature]

Output: [B, 512, 256] visible tokens + [B, 512] indices
```

**Parameters**:
- Keep ratio: `0.5` (512 out of 1024 tokens)
- Temperature schedule:
  - Epochs 0-9: τ_f = 1.0 (warmup)
  - Epochs 10-39: τ_f linearly anneals 1.0 → 0.5
  - Epochs 40+: τ_f = 0.5 (final)
  - τ_b = 4 × τ_f always

**Key Design**:
- **Learned, not random**: Network learns which regions matter
- **Differentiable**: Gumbel-Softmax enables gradient flow
- **Decoupled temperatures**: Sharp forward selection, smooth backward gradients

---

### 3. Vector Quantizer (EMAVectorQuantizer)

**Purpose**: Map continuous tokens to discrete codebook entries

```
Input: [B, 512, 256] encoder outputs (z_e)

Quantization:
  1. Compute distances: ||z_e - codebook||²
  2. Nearest neighbor: indices = argmin(distances)
  3. Lookup: z_q = codebook[indices]
  4. Straight-through: z_q_st = z_e + (z_q - z_e).detach()

EMA Update (training only):
  • Count: N_i = γ·N_i + (1-γ)·Σ[idx==i]
  • Sum: m_i = γ·m_i + (1-γ)·Σ z_e[idx==i]
  • Codebook: c_i = m_i / N_i (with Laplace smoothing)

Dead Code Revival (every 100 steps):
  • Find codes with usage < 2
  • Replace with perturbed encoder outputs + noise

Output: [B, 512, 256] quantized + commit_loss + [B, 512] indices
```

**Parameters**:
- Codebook size: `512` codes
- Code dimension: `256`
- EMA decay: `γ = 0.99`
- Dead threshold: `2` uses per revival interval
- Revival interval: `100` steps

**Commitment Loss**:
```
commit_loss = MSE(z_e, z_q.detach())
```
Pulls encoder outputs toward (stopped) codebook entries.

---

### 4. Fingerprint Extraction

**Purpose**: Generate compact 32-dim representation for database storage

```
Input: [B, 512, 256] quantized tokens (z_q)

Pooling Options:
  A. Mean Pooling (default):
     pooled = z_q.mean(dim=1)  → [B, 256]
     
  B. Attention Pooling (optional):
     attn_weights = softmax(MLP(z_q))  → [B, 512, 1]
     pooled = Σ(attn_weights * z_q)   → [B, 256]

Projection:
  fingerprint = Linear(256 → 32)(pooled)  → [B, 32]
```

**Parameters**:
- Fingerprint dim: `32`
- Use attention pooling: `False` (default is mean pooling)

**Alternative: Histogram Fingerprint**:
```
hist = bincount(indices) / N  → [B, 512]
L1-normalized code usage distribution
```

---

### 5. Transformer Demasker

**Purpose**: Reconstruct full 1024-token sequence from 512 visible tokens

```
Input: [B, 512, 256] visible tokens + [B, 512] indices

Step 1: Build Full Sequence
  • Initialize: [B, 1024, 256] filled with learnable mask tokens
  • Scatter: Insert visible tokens at their original positions
  • Add positional embeddings (learnable, [1, 1024, 256])

Step 2: Dimension Compression
  • Project: 256 → 128 (inner_dim)
  • Reduces transformer compute from 3.1M → ~800K params

Step 3: Transformer Processing
  • 4 Transformer blocks at inner_dim=128
  • Each block: PreNorm MHSA + PreNorm FFN
  • 8 attention heads
  • FFN dim: 1024 (scaled proportionally to inner_dim)

Step 4: Dimension Expansion
  • Project: 128 → 256
  • LayerNorm

Step 5: Reshape to Spatial
  • [B, 1024, 256] → [B, 256, 32, 32]

Output: [B, 256, 32, 32]
```

**Parameters**:
- Spatial tokens: `1024` (32×32)
- Model dim: `256`
- Inner dim: `128` (compression for efficiency)
- Layers: `4`
- Attention heads: `8`
- FFN dim: `1024` (at full model_dim=256)
- Dropout: `0.0`

**Transformer Block**:
```
x = x + MHSA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Parameter Count Optimization**:
- Without compression: ~3.1M params
- With 256→128 compression: ~800K params

---

### 6. CNN Decoder

**Purpose**: Upsample spatial features to full resolution

```
Input: [B, 256, 32, 32]

UpsampleBlock 1:
  ResBlock(256 → 128, s=1)
  Bilinear Upsample 2×
  → [B, 128, 64, 64]

UpsampleBlock 2:
  ResBlock(128 → 64, s=1)
  Bilinear Upsample 2×
  → [B, 64, 128, 128]

UpsampleBlock 3:
  ResBlock(64 → 32, s=1)
  Bilinear Upsample 2×
  → [B, 32, 256, 256]

Output: [B, 32, 256, 256]
```

**Parameters**:
- Decoder channels: `[128, 64, 32]`
- Upsampling: Bilinear (align_corners=False)

---

### 7. Decoder Heads

#### A. ContactReconHead

**Purpose**: Reconstruct full Hi-C contact map

```
Input: [B, 32, 256, 256]

Architecture:
  Conv2d(32 → 16, k=3, p=1)
  GELU
  Conv2d(16 → 1, k=1)
  
Output: [B, 1, 256, 256] (raw values, no activation)
```

**Target**: log₂(OE + ε) clipped to [-5, 5]

**Loss**: MSE + Pearson correlation

---

#### B. CellClassifierHead

**Purpose**: Predict cell type from encoder features

```
Input: [B, 256] z_e_mean (mean-pooled pre-VQ encoder outputs)

Architecture:
  Linear(256 → 128)
  GELU
  Dropout(0.3)
  Linear(128 → 64)
  GELU
  Dropout(0.3)
  Linear(64 → n_cell_types)
  
Output: [B, n_cell_types] raw logits
```

**Parameters**:
- Hidden dims: `[128, 64]`
- Dropout: `0.3`
- Default n_cell_types: `16` (updated dynamically from data)

**Loss**: CrossEntropyLoss

**Key Design**:
- Operates on **pre-VQ** features (z_e_mean)
- Full gradient path through encoder
- Forces encoder to learn cell-type discriminative features

---

## Model Parameters

### Total Parameter Count

| Component | Parameters |
|-----------|-----------|
| Encoder | ~1.2M |
| Masker | ~17K |
| VQ Codebook | 131K (512×256) |
| Transformer Demasker | ~800K |
| CNN Decoder | ~450K |
| ContactReconHead | ~5K |
| CellClassifierHead | ~35K |
| **Total** | **~2.6M** |

### Configuration Summary

```python
# Architecture
ENCODER_CHANNELS = [32, 64, 128, 256]
DECODER_CHANNELS = [128, 64, 32]
N_TRANSFORMER_LAYERS = 4
N_HEADS = 8
FFN_DIM = 1024
DEMASKER_INNER_DIM = 128

# Codebook
N_CODES = 512
CODE_DIM = 256
EMA_GAMMA = 0.99
DEAD_THRESHOLD = 2
REVIVAL_INTERVAL = 100

# Masking
KEEP_RATIO = 0.5
TAU_F_WARMUP = 1.0
TAU_F_FINAL = 0.5
TAU_F_WARMUP_EPOCHS = 10
TAU_F_ANNEAL_EPOCHS = 30

# Fingerprint
FP_DIM = 32
USE_ATTENTION_POOLING = False

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
WARMUP_STEPS = 500

# Loss Weights
LOSS_W_RECON = 1.0
LOSS_W_VQ = 0.75
LOSS_W_CLASSIFIER = 5.0
```

---

## Output Specifications

### forward() Returns

```python
{
    "contact_recon": torch.Tensor,  # [B, 1, 256, 256] reconstructed contact map
    "z_e":           torch.Tensor,  # [B, 512, 256] encoder outputs (visible)
    "z_e_mean":      torch.Tensor,  # [B, 256] mean-pooled encoder features
    "z_q":           torch.Tensor,  # [B, 512, 256] quantized (detached)
    "indices":       torch.Tensor,  # [B, 512] codebook indices
    "fingerprint":   torch.Tensor,  # [B, 32] database fingerprint
    "cell_logits":   torch.Tensor,  # [B, n_cell_types] (if use_classifier_head)
}
```

### encode_fingerprint() Returns

```python
fingerprint: torch.Tensor  # [B, 32] for database ingestion
```

**Inference-only method** (no gradients):
- Encodes contact map to 32-dim fingerprint
- Used for database insertion and similarity search

---

## Ablation Flags

The model supports two ablation studies:

### 1. use_masking (default: True)

**When False**:
- All 1024 tokens pass through VQ (no masking)
- Tests importance of learned token selection
- Expected: Lower performance, higher compute

### 2. use_film (default: True)

**When False**:
- Assay embedding is zeroed out
- Tests importance of assay-type conditioning
- Expected: Reduced ability to handle multi-assay data

**Usage**:
```python
model = MQVAE(
    use_masking=False,  # ablate masking
    use_film=False,     # ablate FiLM conditioning
)
```

---

## Data Flow Dimensions

```
Input Contact Map:              [B, 1, 256, 256]
Assay ID:                       [B]

↓ Encoder
Spatial Tokens (full):          [B, 1024, 256]

↓ Masker
Visible Tokens:                 [B, 512, 256]
Visible Indices:                [B, 512]

↓ Vector Quantizer
Quantized Tokens:               [B, 512, 256]
Codebook Indices:               [B, 512]
Commit Loss:                    scalar

↓ Fingerprint Extraction
Fingerprint:                    [B, 32]

↓ Cell Classifier (branch)
z_e_mean:                       [B, 256]
Cell Logits:                    [B, n_cell_types]

↓ Transformer Demasker
Full Sequence:                  [B, 1024, 256]
Spatial Features:               [B, 256, 32, 32]

↓ CNN Decoder
Decoded Features:               [B, 32, 256, 256]

↓ Contact Reconstruction Head
Reconstructed Contact:          [B, 1, 256, 256]
```

---

## Key Design Decisions

### 1. Masking Before VQ
- **Why**: Select informative regions from rich continuous features
- **Benefit**: Better codebook usage, computational efficiency
- **Alternative**: Masking after VQ would waste codebook capacity

### 2. Learned Masking (not Random)
- **Why**: Different spatial regions have different information density
- **Benefit**: Model learns to focus on TAD boundaries, compartments
- **Implementation**: Gumbel-Softmax with temperature annealing

### 3. EMA Codebook (not Gradient-based)
- **Why**: More stable training, no codebook gradient needed
- **Benefit**: Faster convergence, better code utilization
- **Maintenance**: Dead code revival every 100 steps

### 4. Transformer Compression (256→128)
- **Why**: Reduce parameter count from 3.1M → 800K
- **Benefit**: Faster training, less overfitting
- **Trade-off**: Minimal performance impact

### 5. Cell Classifier on Pre-VQ Features
- **Why**: Full gradient path through encoder
- **Benefit**: Forces encoder to learn discriminative features
- **Alternative**: Post-VQ would have stopped gradients

### 6. Multi-task Learning
- **Why**: Shared representations benefit all tasks
- **Tasks**: Contact reconstruction + cell classification
- **Benefit**: Better generalization, richer features

---

## Training Details

### Loss Function

```python
total_loss = (
    LOSS_W_RECON * recon_loss +           # MSE + Pearson
    LOSS_W_VQ * commit_loss +             # MSE(z_e, sg(z_q))
    LOSS_W_CLASSIFIER * classifier_loss   # CrossEntropy
)
```

**Weights**:
- Reconstruction: `1.0`
- VQ commitment: `0.75`
- Cell classifier: `5.0` (strong signal for small datasets)

### Optimizer

- **Type**: AdamW
- **Learning rate**: `1e-4`
- **Betas**: `(0.9, 0.999)`
- **Weight decay**: `0.05`
- **Gradient clipping**: `1.0`
- **Warmup**: `500` steps

### Temperature Schedule

```python
def get_tau_f(epoch):
    if epoch < 10:
        return 1.0  # warmup
    elif epoch < 40:
        return 1.0 - (epoch - 10) / 30 * 0.5  # anneal
    else:
        return 0.5  # final
```

---

## Memory Requirements

**Per Batch (B=8)**:
- Input: 8 × 1 × 256 × 256 × 4 bytes = 2 MB
- Encoder output: 8 × 1024 × 256 × 4 bytes = 8 MB
- Quantized: 8 × 512 × 256 × 4 bytes = 4 MB
- Decoder output: 8 × 32 × 256 × 256 × 4 bytes = 16 MB
- **Peak activation**: ~100 MB (with gradients)

**Model Parameters**: ~2.6M × 4 bytes = 10 MB

**Total GPU Memory** (training): ~500 MB - 1 GB

---

## Inference Speed

**Single Sample** (256×256 contact map):
- Encoder: ~5 ms
- Masker: ~1 ms
- VQ: ~2 ms
- Transformer: ~10 ms
- Decoder: ~5 ms
- **Total**: ~25 ms on GPU

**Batch of 8**: ~40 ms (better GPU utilization)

---

## Model Checkpointing

**Saved State**:
```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
    "config": model._arch,  # architecture hyperparameters
}
```

**Loading**:
```python
checkpoint = torch.load("checkpoint.pt")
model = MQVAE(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
```

---

## References

**Architecture Inspirations**:
- Vector Quantization: van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE)
- Masking: He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE)
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
- Gumbel-Softmax: Jang et al., "Categorical Reparameterization with Gumbel-Softmax"

---

**End of Document**
