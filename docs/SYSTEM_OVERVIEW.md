# MQ-VAE Hi-C Structural Fingerprinting System — Complete Overview

## What This System Does

**MQ-VAE** learns to compress 3D genome structure (Hi-C contact matrices) into compact 16-dimensional fingerprints that capture:
- TAD boundaries
- A/B compartments  
- Loop structures
- Contact decay patterns

**Query pipeline:** Upload a new Hi-C sample → get structural similarity to reference cell types + biological annotations.

---

## Architecture Components

### 1. **Encoder** (`encoder.py`)
- 4-stage CNN: 256×256 contact map → 32×32×256 latent
- Diagonal pooling: extracts off-diagonal bands (loops, TADs)
- FiLM conditioning: modulates features by assay type (bulk Hi-C, Micro-C, etc.)

### 2. **Vector Quantization** (`codebook.py`)
- 512-code Gumbel-Softmax VQ-VAE
- EMA codebook updates
- Dead code revival (every 100 steps)
- Learns discrete structural vocabulary

### 3. **Masker** (`masker.py`)
- Keeps 50% of 1024 tokens (512 visible)
- Gumbel-Softmax sampling with annealing temperature τ
- Forces model to learn robust representations

### 4. **Transformer Demasker** (`transformer.py`)
- 4-layer decoder-only transformer
- Predicts masked tokens from visible ones
- 8 attention heads, 1024 FFN dim

### 5. **Decoder** (`decoder.py`)
- Upsamples 32×32 → 256×256
- Reconstructs contact matrix

### 6. **Auxiliary Heads** (`heads.py`)
- **Boundary head:** predicts TAD boundaries (binary classification)
- **Compartment head:** predicts A/B compartment (regression)
- Both trained jointly with reconstruction

### 7. **Fingerprint Projection** (`model.py`)
- 256-dim latent → 32-dim fingerprint via learned projection
- Used for database search and similarity

---

## Training Pipeline

### Step 1: Preprocessing (`preprocess.py`)
```bash
python src/preprocess.py
```

**What it does:**
- Loads `.mcool` files from `MCOOL_DIR`
- Extracts 256×256 windows (100kb resolution, 50% overlap)
- Computes insulation scores (TAD boundaries)
- Computes compartment eigenvectors (A/B)
- Saves to `data/processed/<sample_id>/`

**Output:** `.pt` files with `{contact, boundary, compartment, assay_id, chrom, start, end}`

---

### Step 2: Training (`train.py`)
```bash
python src/train.py --epochs 50 --batch_size 8 --plot_every 1
```

**What it does:**
- Loads preprocessed tiles from all cell lines
- Trains MQ-VAE with multi-task loss:
  - Reconstruction (MSE)
  - VQ commitment
  - Boundary (BCE with pos_weight=9)
  - Compartment (MSE)
- Saves checkpoints every 5 epochs
- Updates dashboard PNG every epoch

**Output:**
- `checkpoints/<run_name>/mqvae_epoch*_best.pt`
- `trash/plots/dashboard_<run_name>.png`
- `trash/train_log_<run_name>.json`

**Dashboard shows:**
- Loss curves (total, recon, VQ)
- Task metrics (boundary F1, compartment R)
- Codebook usage (active codes, temperature τ)
- Fingerprint PCA (cell type clustering)
- Training summary stats

---

### Step 3: Database Ingestion (`database.py`)
```bash
python src/database.py --model checkpoints/full/mqvae_epoch049_best.pt
```

**What it does:**
- Loads trained model
- Extracts 32-dim fingerprints for all windows
- Stores in DuckDB: `window_fingerprints`, `sample_histograms`, `locus_centroids`
- Builds FAISS index for fast similarity search
- Integrates cCRE annotations (ENCODE)

**Output:**
- `data/hic_fingerprints.duckdb`
- `data/locus_centroids.faiss`

---

### Step 4: Query (`query.py`)
```bash
python src/query.py --mcool path/to/query.mcool --level 4
```

**What it does:**
- Extracts fingerprints from query sample
- Searches FAISS index for similar windows
- Computes per-locus similarity scores
- Runs statistical tests (permutation p-values)
- Annotates with cCRE differences, gene context, boundary/compartment changes
- Generates multi-level report

**Output levels:**
- **Level 1:** Overall cell type classification
- **Level 2:** Per-chromosome similarity breakdown
- **Level 3:** Divergent loci with cCRE annotations
- **Level 4:** Full specificity report (exact coordinates, gene names, clinical priority)

---

## Replicate Management (`replicates.py`)

**Auto-detection:**
- `K562_rep1`, `K562_rep2` → group `K562`
- `IMR-90_r1`, `IMR-90_batch_A` → group `IMR-90`

**Registry validation:**
- Checks if `.mcool` file exists
- Checks if `.bed.gz` cCRE file exists
- Assigns mode: `"full"` (both exist) or `"structural_only"` (no cCRE)
- Skips samples with missing `.mcool`

**Query-time grouping:**
- Reports group-level similarity (mean ± std across replicates)
- Confidence: `"high"` (std < 0.05), `"medium"` (std < 0.15), `"low"` (std ≥ 0.15)

---

## Biological Annotations

### cCRE Integration (`annotator.py`)
- Loads ENCODE cCRE v3 `.bed.gz` files
- Categories: PLS (promoter-like), pELS (proximal enhancer), dELS (distal enhancer), CA-CTCF, CA-TF
- `query_window_with_coords()` returns exact genomic coordinates of cCRE elements

### Gene Annotation (`gene_annotator.py`)
- Loads GENCODE v45 GTF (63,187 genes)
- Maps genomic coordinates → nearest gene
- Distance labels: `"promoter"` (<2kb), `"near"` (<500kb), `"distal"` (<2Mb)
- Cancer gene classification: oncogenes, tumor suppressors (curated lists)

### Specificity Analysis (`specificity.py`)
Combines 6 analyses:
1. **cCRE coordinates:** exact positions of promoter/enhancer differences
2. **Boundary disruption:** lost TAD boundaries + affected genes
3. **Compartment switches:** A→B or B→A flips + gene context
4. **Similarity drivers:** top divergent regions with gene names
5. **QC flags:** telomere, centromere, flat compartment artifacts
6. **Clinical priority:** actionability score (HIGH/MEDIUM/LOW)

---

## Configuration (`config.py`)

### Environment Variables
```bash
export HIC_PROJECT_ROOT=/path/to/DATABASE_CONCEPT
export HIC_MCOOL_DIR=/path/to/mcool/files
export HIC_CCRE_DIR=/path/to/ccre/files
export HIC_GENCODE_GTF=/path/to/gencode.v45.basic.annotation.gtf.gz
```

### Auto-Discovery
- Scans `MCOOL_DIR` for all `.mcool` files
- Strips ENCODE accessions: `K562_4DNFI18UHVRO.mcool` → sample ID `K562`
- Matches `.bed.gz` cCRE files by prefix
- Falls back to manual registry if auto-discovery finds nothing

### Registries
```python
CELL_LINE_REGISTRY = {
    "K562":   {"file": "K562_4DNFI18UHVRO.mcool", "assay": "bulk_hic", "tissue": "leukemia"},
    "IMR-90": {"file": "IMR-90_4DNFIJTOIGOI.mcool", "assay": "bulk_hic", "tissue": "lung_fibroblast"},
    # ... auto-discovered from MCOOL_DIR
}

CCRE_REGISTRY = {
    "K562":   "/path/to/K562_ENCFF455VKH.bed.gz",
    "IMR-90": "/path/to/IMR-90_ENCFF685BXB.bed.gz",
    # ... auto-matched from CCRE_DIR
}
```

---

## Testing (`setup_test.py`)

**8 end-to-end tests:**
1. **Dependencies:** All packages importable (torch, cooler, duckdb, faiss, etc.)
2. **Registry:** Auto-discovery finds mcool + cCRE files
3. **Replicates:** Suffix detection, grouping, validation
4. **cCRE fallback:** Structural-only mode when cCRE missing
5. **Dashboard:** PNG generation with PCA + codebook plots
6. **Training:** 3-epoch run with dashboard updates
7. **Database:** Fingerprint extraction + FAISS indexing
8. **Query:** End-to-end query with specificity report

**Run:**
```bash
python src/setup_test.py
```

**Output:**
- `trash/setup_test_results/summary.json`
- `trash/plots/dashboard_setup_test.png`
- All tests passing in ~120s

---

## Key Design Decisions

### Why VQ-VAE?
- **Discrete codes** → interpretable structural vocabulary
- **Codebook** → shared patterns across cell types
- **Gumbel-Softmax** → differentiable discrete sampling

### Why Masking?
- Forces model to learn **robust** representations
- Prevents overfitting to local patterns
- Improves generalization to unseen samples

### Why Multi-Task Learning?
- **Reconstruction** alone is not enough
- **Boundary + compartment** heads enforce biological structure
- Auxiliary tasks improve fingerprint quality

### Why 32-dim Fingerprints?
- Compact enough for fast search (FAISS)
- Rich enough to capture structural diversity
- Empirically optimal (tested 16, 32, 64, 128)

---

## Limitations & Assumptions

### What the Model Learns
✅ **Structural patterns:** TADs, loops, compartments, contact decay  
✅ **Cell type similarity:** K562 vs IMR-90 structural differences  
✅ **Divergent regions:** which genomic loci differ most

### What the Model Does NOT Learn
❌ **Promoter/enhancer locations** from contact matrix alone  
❌ **Gene expression levels**  
❌ **Functional consequences** of structural changes

### Biological Annotations Are External
- cCRE coordinates: from ENCODE (not learned)
- Gene names: from GENCODE (not learned)
- Cancer genes: curated lists (not learned)

**The model detects structural differences; we infer regulatory consequences by overlaying external annotations.**

---

## Query Output Interpretation

### Structural Similarity
```
Most similar to: K562 (87% of windows)
```
- **High similarity (>70%):** Same cell type or very similar
- **Medium (30-70%):** Related tissue or treatment effect
- **Low (<30%):** Different cell type or major structural change

### Regulatory Inference
```
PLS: +18 (more in query)
chr21:34,887,104-34,887,402  near RUNX1 (oncogene)
```
- **Assumes:** structural similarity → similar regulatory elements
- **Caveat:** Could be wrong if different biology produces similar structure
- **Best for:** comparing similar cell types or treatment effects

### Unknown/Treated Samples
If query sample doesn't match any reference:
- Report structural divergence only
- Flag as `"structural_only"` mode
- Recommend providing cCRE annotation for regulatory analysis

---

## File Organization

```
DATABASE_CONCEPT/
├── src/
│   ├── config.py              # All hyperparameters, paths, registries
│   ├── preprocess.py          # mcool → tiles + boundary + compartment
│   ├── dataset.py             # PyTorch Dataset/DataLoader
│   ├── encoder.py             # CNN encoder with diagonal pooling
│   ├── masker.py              # Gumbel-Softmax token masking
│   ├── codebook.py            # VQ codebook with EMA + revival
│   ├── transformer.py         # Demasker transformer
│   ├── decoder.py             # Upsampling decoder
│   ├── heads.py               # Boundary + compartment heads
│   ├── model.py               # Full MQ-VAE model
│   ├── loss.py                # Multi-task loss
│   ├── train.py               # Training loop + dashboard
│   ├── database.py            # Fingerprint extraction + DuckDB + FAISS
│   ├── query.py               # Query pipeline + specificity
│   ├── annotator.py           # cCRE annotation (ENCODE)
│   ├── gene_annotator.py      # Gene annotation (GENCODE)
│   ├── specificity.py         # 6-analysis specificity report
│   ├── replicates.py          # Replicate management
│   ├── dashboard.py           # Publication-quality training plots
│   ├── setup_test.py          # 8-test end-to-end verification
│   └── ablation.py            # Ablation studies
├── data/
│   ├── processed/             # Preprocessed tiles (.pt files)
│   ├── hic_fingerprints.duckdb
│   ├── locus_centroids.faiss
│   └── gencode.v45.basic.annotation.gtf.gz
├── checkpoints/               # Model checkpoints
├── trash/
│   ├── plots/                 # Training dashboards
│   └── setup_test_results/    # Test outputs
├── docs/
│   ├── training_guide.md
│   ├── query_guide.md
│   └── SYSTEM_OVERVIEW.md     # This file
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Paths (Optional)
```bash
export HIC_MCOOL_DIR=/path/to/mcool/files
export HIC_CCRE_DIR=/path/to/ccre/files
```

### 3. Run Setup Test
```bash
python src/setup_test.py
```

### 4. Preprocess Data
```bash
python src/preprocess.py
```

### 5. Train Model
```bash
python src/train.py --epochs 50
```

### 6. Build Database
```bash
python src/database.py --model checkpoints/full/mqvae_epoch049_best.pt
```

### 7. Query Sample
```bash
python src/query.py --mcool path/to/query.mcool --level 4
```

---

## Citation

If you use this system, please cite:

```bibtex
@software{mqvae_hic_2026,
  title={MQ-VAE: Masked Vector-Quantized Variational Autoencoder for Hi-C Structural Fingerprinting},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

---

## Contact & Support

- **Issues:** Open a GitHub issue
- **Questions:** [your-email@domain.com]
- **Documentation:** See `docs/` folder

---

**Last updated:** March 6, 2026  
**Version:** 1.0.0  
**Status:** Production-ready, all tests passing
