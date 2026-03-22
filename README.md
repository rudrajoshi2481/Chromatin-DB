# Chromatin-DB: Hi-C Structural Fingerprint Database

A deep learning-based system for encoding Hi-C contact maps into compact 64-dimensional fingerprints for structural similarity search and comparison.

## Overview

Chromatin-DB uses a Masked Quantized Variational Autoencoder (MQ-VAE) to encode 3D chromatin structure from Hi-C data. The system provides:

- **Fingerprint extraction**: Convert Hi-C matrices to 64-dim vectors
- **Structural similarity search**: Query unknown samples against reference database
- **Cell type classification**: Identify cell lines from chromatin structure
- **cCRE comparison**: Compare regulatory landscapes between samples
- **Divergent loci detection**: Find regions with structural differences

## Model Architecture

- **Encoder**: Convolutional feature extraction from Hi-C matrices
- **VQ Layer**: Vector quantization with 512 codebook entries
- **Transformer Demasker**: Context-aware code reconstruction
- **Fingerprint Head**: Projects quantized codes to 64-dim fingerprints
- **Classifier Head**: Cell type prediction from latent representations

## Repository Structure

```
Chromatin-DB/
├── src/
│   ├── chromatin_query.py    # CLI tool for queries & comparisons
│   ├── model.py               # MQVAE architecture
│   ├── train.py               # Training script
│   ├── preprocess.py          # Hi-C preprocessing pipeline
│   ├── dataset.py             # PyTorch dataset & dataloaders
│   ├── eval_fingerprints.py   # Fingerprint quality evaluation
│   └── ...
├── runs/                      # Training runs & experiments
│   ├── 67cl_512codes/        # Best model checkpoint
│   └── experiments/           # Example query results (logs)
├── docs/
│   └── EXPERIMENTS_SUMMARY.md # Detailed experiment results
├── data/                      # Data directory (not in git)
└── trash/                     # Temporary outputs (gitignored)
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy scipy cooler cooltools duckdb faiss-gpu
pip install tabulate rich  # For CLI UI
```

### CLI Usage

The `chromatin_query.py` CLI provides interactive commands for querying and comparing Hi-C samples.

#### 1. List Available Data

```bash
python src/chromatin_query.py list
```

Shows available mcool files and cCRE annotations.

#### 2. Query a Sample

Query an unknown Hi-C file against the reference database:

```bash
python src/chromatin_query.py query \
    --input GM12878_4DNFI2A4OBS9.mcool \
    --max-tiles 400
```

**Output includes:**
- Top 5 structural matches with similarity scores
- Visual bars for similarity levels
- Divergent loci (regions with low similarity)
- cCRE regulatory comparison (if available)
- Processing summary

**Example result:**
```
╔════════════════════════════════════════════════════════════╗
║                    CHROMATIN QUERY RESULTS                   ║
╚════════════════════════════════════════════════════════════╝

🎯 TOP 5 STRUCTURAL MATCHES
┌─────────────┬────────────┬──────────────────┬───────────────┐
│ Cell Type   │ Similarity │ Visual           │ Confidence    │
├─────────────┼────────────┼──────────────────┼───────────────┤
│ GM12878     │ 0.9994     │ ████████████████ │ VERY HIGH ★★★ │
│ HCT116      │ 0.9971     │ ███████████████░ │ VERY HIGH ★★★ │
│ HeLa-S3     │ 0.9947     │ ██████████████░░ │ VERY HIGH ★★★ │
│ K562        │ 0.9939     │ ██████████████░░ │ VERY HIGH ★★★ │
│ HAP-1       │ 0.9936     │ █████████████░░░ │ HIGH ★★☆      │
└─────────────┴────────────┴──────────────────┴─────────────┘

📊 SUMMARY
┌─────────────────┬───────────────╮
│ Top Match       │ GM12878       │
│ Similarity      │ 0.9994        │
│ Tiles Analyzed  │ 195           │
│ Status          │ ✓ MATCH       │
└─────────────────┴───────────────╯
```

#### 3. Compare Two Samples

Direct head-to-head comparison of two Hi-C files:

```bash
python src/chromatin_query.py compare \
    --a GM12878_4DNFI2A4OBS9.mcool \
    --b K562_4DNFI18UHVRO.mcool \
    --max-tiles 300
```

**Output includes:**
- Mean and max similarity scores
- Per-chromosome similarity breakdown
- cCRE regulatory landscape differences
- Most divergent loci
- Overall verdict (Identical/Very Similar/Similar/Different)

**Example result:**
```
🧬 GM12878 vs K562 Comparison
┌───────────────────┬─────────┬──────────────────┐
│ Metric            │ Value   │ Visual           │
├───────────────────┼─────────┼──────────────────┤
│ Mean Similarity   │ 0.8264  │ █████████████░░░ │
│ Max Similarity    │ 0.9997  │ ████████████████ │
│ Loci > 0.85 match │ 100.0%  │ ████████████████ │
└───────────────────┴─────────┴──────────────────┘

Verdict: SIMILAR (different cell types with shared structural patterns)
```

#### 4. Run Accuracy Test

Validate model accuracy across known cell lines:

```bash
python src/chromatin_query.py test --n-pairs 10
```

Tests classification accuracy by extracting fingerprints from query samples and comparing against the reference database.

**Expected results:** 100% accuracy on tested cell lines (GM12878, K562, HCT116, etc.)

## Example Results

See `runs/experiments/` for detailed logs:

| Experiment | Description | Log File |
|------------|-------------|----------|
| GM12878 Query | Query GM12878 against database | `01_GM12878_query.log` |
| K562 Query | Query K562 against database | `02_K562_query.log` |
| GM12878 vs K562 | Direct comparison | `03_GM12878_vs_K562_compare.log` |
| Accuracy Test | 10-cell validation | `04_accuracy_test_10cells.log` |

## Model Performance

- **Classification Accuracy**: 100% on tested cell lines (10/10 correct)
- **Top Match Similarity**: >0.999 for correct matches
- **Cross-cell Similarity**: 0.82-0.95 (GM12878 vs K562)
- **Fingerprint Dimension**: 64
- **Processing Time**: ~5-9 minutes per sample

## Technical Details

### Data Pipeline

1. **Hi-C Loading**: `.mcool` files at 100 kb resolution
2. **OE Computation**: Observed/Expected normalization using cooltools
3. **Tiling**: 256×256 bin windows (25.6 Mb) with 128-bin step
4. **Fingerprint Extraction**: MQ-VAE encoding to 64-dim vectors
5. **Similarity**: Cosine similarity between fingerprints

### Key Parameters

- **Resolution**: 100 kb
- **Tile Size**: 256 bins (25.6 Mb)
- **Tile Step**: 128 bins (12.8 Mb overlap)
- **Codebook Size**: 512 entries
- **Fingerprint Dim**: 64
- **Model**: MQ-VAE Epoch 181

## Citation

If you use this tool, please cite:

```
Chromatin-DB: A Deep Learning Framework for Hi-C Structural Fingerprinting
[Your citation here]
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: March 22, 2026
