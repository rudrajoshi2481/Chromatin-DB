# MQ-VAE Hi-C Training Guide

End-to-end walkthrough from raw `.mcool` files to a trained model and DuckDB fingerprint database.

---

## Prerequisites

Install all dependencies:

```bash
pip install -r /app/tmp/DATABASE_CONCEPT/requirements.txt
```

Verify data files are present:

```bash
ls /app/tmp/ink-react/chromatin-manager/data/downloads/mcool/
# Expected:
# HeLa-S3_4DNFIBM9QCFG.mcool
# IMR-90_4DNFIJTOIGOI.mcool
# K562_4DNFI18UHVRO.mcool
# KBM-7_4DNFI5IHU27G.mcool
# foreskin_fibroblast_4DNFIQJQY7PW.mcool
```

---

## Step 1 — Preprocess mcool Files

Tiles each chromosome into 256×256 OE contact windows at 100 kb resolution.
Also computes TAD boundary labels and A/B compartment eigenvectors.
Output: `.npz` files in `/app/tmp/DATABASE_CONCEPT/data/processed/<sample_id>/`

```bash
cd /app/tmp/DATABASE_CONCEPT
python src/preprocess.py
```

Expected output per chromosome:
```
=== Preprocessing K562 ===
  [K562] chr1: computing OE matrix...
  [K562] chr1: 47 tiles saved
  [K562] chr2: 44 tiles saved
  ...
  Total tiles: ~529
```

Total preprocessing time: ~30–60 min per cell line on CPU (parallelise with `--nproc` via cooltools if needed).

### What is stored per tile?

Each `.npz` file (one per chromosome per sample) contains:
| Array | Shape | Description |
|-------|-------|-------------|
| `matrices` | `[N, 256, 256]` | log2(OE+ε) contact values, clipped ±5 |
| `boundaries` | `[N, 256]` | binary TAD boundary labels (float32) |
| `compartments` | `[N, 256]` | normalised E1 eigenvector in [-1, 1] |
| `chroms` | `[N]` | chromosome name |
| `start_bps / end_bps` | `[N]` | genomic coordinates in bp |

---

## Step 2 — Train the Model

### Full training (50 epochs, all 5 cell lines)

```bash
cd /app/tmp/DATABASE_CONCEPT
python src/train.py \
    --epochs 50 \
    --batch_size 8 \
    --run_name full
```

### Quick smoke-test (5 epochs, 1 cell line)

```bash
python src/train.py \
    --epochs 5 \
    --cell_lines K562 \
    --run_name smoke_test
```

### Training CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 8 | Batch size (reduce to 4 if OOM) |
| `--lr` | 1e-4 | Learning rate |
| `--cell_lines` | all 5 | Space-separated subset of cell lines |
| `--run_name` | `full` | Name for checkpoint directory and logs |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--no_boundary` | — | Disable boundary prediction head |
| `--no_compartment` | — | Disable compartment prediction head |
| `--no_masking` | — | Disable Gumbel masking (all tokens visible) |
| `--no_film` | — | Disable FiLM conditioning (ablate assay conditioning) |
| `--n_codes` | 512 | Codebook size |

### What gets saved

```
/app/tmp/DATABASE_CONCEPT/checkpoints/<run_name>/
    mqvae_epoch004_best.pt      # best validation loss so far
    mqvae_epoch009.pt           # periodic checkpoint (every 5 epochs)
    mqvae_epoch049_final.pt     # final model

/app/tmp/DATABASE_CONCEPT/trash/
    train_log_<run_name>.json   # full per-epoch metrics history
```

### Expected training metrics

| Epoch | Total Loss | Recon | Boundary F1 | Compartment r | Active Codes |
|-------|-----------|-------|-------------|---------------|-------------|
| 0 | ~2.5 | ~2.0 | 0.00 | 0.00 | ~50/512 |
| 5 | ~1.8 | ~1.4 | 0.10 | 0.15 | ~150/512 |
| 15 | ~1.2 | ~0.9 | 0.30 | 0.45 | ~300/512 |
| 30 | ~0.9 | ~0.6 | 0.45 | 0.60 | ~400/512 |
| 50 | ~0.7 | ~0.5 | 0.55 | 0.70 | ~450/512 |

> **Note:** With only 5 cell lines, expect lower absolute scores than the 50-sample plan describes. The architecture and code are designed for the 50-sample scale — these 5 lines serve as a functional prototype.

### Temperature schedule (Decoupled ST-GS)

The masker uses two separate temperatures:

| Epoch | τ_f (forward) | τ_b (backward) | Effect |
|-------|-------------|----------------|--------|
| 0–9 | 1.0 | 4.0 | Soft exploration |
| 10–29 | 1.0 → 0.3 | 4.0 → 1.2 | Linear anneal |
| 30+ | 0.3 | 1.2 | Sharp, stable selection |

### Auxiliary loss warmup

| Epoch | aux_weight | Active losses |
|-------|-----------|---------------|
| 0–4 | 0.0 | Recon + VQ only |
| 5–14 | 0.0 → 1.0 | Gradual ramp |
| 15+ | 1.0 | All three heads |

---

## Step 3 — Build the DuckDB Database

After training, extract fingerprints from all tiles and insert them into DuckDB.
Also builds the FAISS locus-centroid index.

```bash
cd /app/tmp/DATABASE_CONCEPT

# Using the best checkpoint
python src/database.py \
    --ckpt checkpoints/full/mqvae_epoch049_best.pt \
    --db data/hic_fingerprints.duckdb \
    --faiss data/locus_centroids.faiss \
    --overwrite
```

### What gets created

```
/app/tmp/DATABASE_CONCEPT/data/
    hic_fingerprints.duckdb       # DuckDB database (~4-10 MB)
    locus_centroids.faiss         # FAISS flat index (~78 KB)
    locus_centroids.locus_keys.json  # locus key mapping for FAISS
```

### DuckDB schema

```sql
-- Sample metadata
SELECT * FROM samples LIMIT 3;
-- sample_id | cell_type | tissue | assay_type | n_tiles

-- Per-window 32-dim fingerprints
SELECT sample_id, chr, start_bp, end_bp FROM window_fingerprints LIMIT 5;

-- Per-sample code histograms (512-dim, L1-normalised)
SELECT sample_id FROM sample_histograms;

-- Locus centroids (mean fingerprint across all samples)
SELECT chr, start_bp, n_samples FROM locus_centroids LIMIT 5;
```

### Inspect the database directly

```python
import duckdb
conn = duckdb.connect("/app/tmp/DATABASE_CONCEPT/data/hic_fingerprints.duckdb")

# Total fingerprints
print(conn.execute("SELECT COUNT(*) FROM window_fingerprints").fetchone())

# Fingerprints per sample
conn.execute("SELECT sample_id, COUNT(*) as n FROM window_fingerprints GROUP BY sample_id").df()

# Locus coverage
conn.execute("SELECT chr, COUNT(*) FROM locus_centroids GROUP BY chr ORDER BY chr").df()
```

---

## Step 4 — Run Ablation Studies

```bash
cd /app/tmp/DATABASE_CONCEPT

# All ablation configs (8 configs × ablation_epochs)
python src/ablation.py --epochs 20

# Specific configs only (faster)
python src/ablation.py --epochs 10 --configs 1 4 6 7
```

Outputs in `/app/tmp/DATABASE_CONCEPT/trash/`:
```
trash/
    ablation_results.json           # raw metrics for all configs
    ablation_summary.txt            # formatted comparison table
    ablation_dim_sweep.json         # fingerprint dimensionality AUC
    checkpoints/ablation_*/         # per-config model checkpoints
    train_log_ablation_*.json       # per-config training histories
```

### Ablation config table

| # | Name | What it tests |
|---|------|---------------|
| 1 | `baseline_recon_only` | MAP@5 floor, recon MSE |
| 2 | `boundary_head_only` | Boundary head contribution |
| 3 | `compartment_head_only` | Compartment head contribution |
| 4 | `full_v4` | Combined multi-head benefit |
| 5 | (PCA sweep on #4) | Fingerprint dimensionality |
| 6 | `no_masking` | Masking contribution |
| 7 | `no_film` | FiLM vs no conditioning |
| 10 | `codebook_256` | Small codebook |
| 11 | `codebook_1024` | Large codebook |

---

## GPU Memory Requirements

| Batch size | Tile size | VRAM needed |
|-----------|-----------|-------------|
| 4 | 256×256 | ~6 GB |
| 8 | 256×256 | ~10 GB |
| 16 | 256×256 | ~18 GB |

Reduce `--batch_size` if you hit OOM errors. The model trains correctly at batch=2 (just slower).

---

## Troubleshooting

**`Dataset is empty. Run preprocess.py first`**
→ Run Step 1 before Step 2.

**`cooltools.expected_cis` fails with KeyError**
→ The fallback `_simple_oe` in `preprocess.py` will be used automatically.

**`No module named 'bioframe'`**
→ `pip install bioframe>=0.4.0`

**CUDA OOM during training**
→ Reduce `--batch_size` to 4 or 2.

**Codebook collapse (active_codes < 50)**
→ Increase `REVIVAL_INTERVAL` in `config.py`, or lower `DEAD_THRESHOLD` to 1.
→ Check that data normalisation is correct (OE should be clipped ±5).
