# MQ-VAE Hi-C Query Guide

How to query a new `.mcool` file against the reference DuckDB database and interpret the four-level structural similarity report.

---

## Prerequisites

- Completed Steps 1–3 of the Training Guide (preprocessed data, trained model, built database)
- A query `.mcool` file at 100 kb resolution

---

## Quick Start

```python
from pathlib import Path
import sys
sys.path.insert(0, "/app/tmp/DATABASE_CONCEPT/src")

from query import HiCStructuralDatabase

db = HiCStructuralDatabase(
    ckpt_path  = "/app/tmp/DATABASE_CONCEPT/checkpoints/full/mqvae_epoch049_best.pt",
    db_path    = Path("/app/tmp/DATABASE_CONCEPT/data/hic_fingerprints.duckdb"),
    faiss_path = Path("/app/tmp/DATABASE_CONCEPT/data/locus_centroids.faiss"),
)

results = db.query_mcool(
    mcool_path = "/app/tmp/ink-react/chromatin-manager/data/downloads/mcool/K562_4DNFI18UHVRO.mcool",
    assay_type = "bulk_hic",
    k          = 5,
)
db.print_report(results)
db.close()
```

---

## Command-Line Query

```bash
cd /app/tmp/DATABASE_CONCEPT

python src/query.py \
    --mcool  /path/to/query.mcool \
    --ckpt   checkpoints/full/mqvae_epoch049_best.pt \
    --db     data/hic_fingerprints.duckdb \
    --faiss  data/locus_centroids.faiss \
    --assay  bulk_hic \
    --k      5 \
    --out_json trash/query_results.json
```

### Supported assay types

| `--assay` value | Description |
|----------------|-------------|
| `bulk_hic` | Standard bulk Hi-C (restriction enzyme) |
| `micro_c` | Micro-C (MNase, nucleosome-resolution) |
| `sc_hic` | Single-cell Hi-C |
| `chia_pet` | ChIA-PET |

> **Note:** All 5 reference cell lines are `bulk_hic`. FiLM conditioning handles cross-assay normalisation for queries in other modalities.

---

## Understanding the Four-Level Output

### Level 1 — Sample Classification

```
── LEVEL 1: Sample Classification ─────────────────────────
  Most similar to: K562  (87% of windows match)
```

Computed as majority vote across all ~529 genome-wide windows.
Confidence: fraction of windows where `top_cell_type == winner`.

**Interpretation:**
- `>80%`: Strong match → the query sample is highly similar to this cell type
- `50–80%`: Moderate match → mixed structural profile
- `<50%`: Ambiguous → likely a novel cell type or cancer sample with extensive reorganisation

---

### Level 2 — Chromosome Summary

```
── LEVEL 2: Chromosome Summary ─────────────────────────────
  chr1    0.91  ████████████████████
  chr2    0.88  ███████████████████
  chr3    0.34  ██████░░░░░░░░░░░░░   ← DIVERGENT
  chr5    0.12  ██░░░░░░░░░░░░░░░░░   ← DIVERGENT
  chrX    0.85  ██████████████████
```

Per-chromosome mean cosine similarity to the top reference sample.

**Thresholds:**
| Score | Classification |
|-------|---------------|
| > 0.70 | STRUCTURALLY_SIMILAR |
| 0.30–0.70 | PARTIALLY_SIMILAR |
| < 0.30 | STRUCTURALLY_DIFFERENT / DIVERGENT |

---

### Level 3 — Window-Level Map

```
── LEVEL 3: Window-Level Map (divergent windows only) ──────
  chr3:0-25600000           0.142  ██░░░░░░░░░░░░░  nearest: IMR-90
  chr5:0-25600000           0.089  █░░░░░░░░░░░░░░  nearest: IMR-90
  chr8:89600000-115200000   0.228  ███░░░░░░░░░░░░  nearest: K562
```

Only shows windows below `SIM_THRESHOLD_MED` (0.30).
Full results (all windows) are available in `results["window_results"]`.

---

### Level 4 — Divergent Loci Report

```
── LEVEL 4: Divergent Loci Report ──────────────────────────

  ╔══════════════════════════════════════════════════════╗
  ║  Locus: chr8:89600000-115200000                      ║
  ║  Similarity to nearest: 0.228 (K562)                 ║
  ║  Status: STRUCTURALLY_DIFFERENT                      ║
  ║  Boundary Concordance: 0.312 (low)                   ║
  ║  Compartment Concordance: 0.184 (very low)           ║
  ╚══════════════════════════════════════════════════════╝
```

**Boundary Concordance:** Pearson correlation of query TAD boundary labels vs. nearest reference sample at the same locus. Low → TAD organisation differs.

**Compartment Concordance:** Pearson correlation of A/B compartment E1 eigenvectors. Very low → compartment identity has switched (e.g., B→A activation).

---

## Programmatic Access to Results

The `results` dict has the following structure:

```python
{
    "query_path":       str,
    "assay_type":       str,
    "n_windows":        int,

    # Level 1
    "level1_cell_type": str,    # top cell type
    "level1_fraction":  float,  # fraction of windows that voted for top cell type

    # Level 2
    "level2_chrom": {
        "chr1": {"mean_score": 0.91, "n_windows": 12, "divergent": False},
        "chr3": {"mean_score": 0.34, "n_windows": 11, "divergent": True},
        ...
    },

    # Level 3+4 raw data
    "window_results": {
        "chr1:0-25600000": {
            "fingerprint":             np.ndarray,   # [32] float32
            "chr":                     "chr1",
            "start_bp":                0,
            "end_bp":                  25600000,
            "similarity_score":        0.91,
            "top_cell_type":           "K562",
            "status":                  "STRUCTURALLY_SIMILAR",
            "boundary_concordance":    0.72,
            "compartment_concordance": 0.68,
            "top_matches": [
                {"sample_id": "K562", "score": 0.91, "cell_type": "K562", "tissue": "leukemia"},
                ...
            ],
        },
        ...
    },

    # Level 4
    "level4_divergent": [
        {
            "locus":                   "chr8:89600000-115200000",
            "chr":                     "chr8",
            "similarity":              0.228,
            "nearest_sample":          "K562",
            "nearest_ct":              "K562",
            "status":                  "STRUCTURALLY_DIFFERENT",
            "boundary_concordance":    0.31,
            "compartment_concordance": 0.18,
        },
        ...
    ],
}
```

---

## Direct DuckDB Queries

You can query the database directly for custom analyses:

```python
import duckdb
import numpy as np

conn = duckdb.connect("/app/tmp/DATABASE_CONCEPT/data/hic_fingerprints.duckdb",
                      read_only=True)

# All fingerprints for a specific locus
rows = conn.execute("""
    SELECT wf.sample_id, s.cell_type, wf.fingerprint
    FROM window_fingerprints wf
    JOIN samples s ON wf.sample_id = s.sample_id
    WHERE wf.chr = 'chr8' AND wf.start_bp = 89600000
""").fetchall()

for sample_id, cell_type, fp_blob in rows:
    fp = np.frombuffer(fp_blob, dtype=np.float32)
    print(f"{sample_id} ({cell_type}): {fp[:4]}...")

# Sample-level histogram comparison
hist_rows = conn.execute("""
    SELECT sample_id, code_histogram FROM sample_histograms
""").fetchall()

for sample_id, hist_blob in hist_rows:
    hist = np.frombuffer(hist_blob, dtype=np.float32)  # [512] L1-normalised
    print(f"{sample_id}: top 3 codes = {np.argsort(hist)[::-1][:3]}")

# Count divergent loci per chromosome across all samples
conn.execute("""
    SELECT chr, COUNT(*) as n_loci
    FROM locus_centroids
    GROUP BY chr ORDER BY chr
""").df()

conn.close()
```

---

## Using FAISS for Nearest-Locus Search

The FAISS index contains L2-normalised locus centroids. Use it to find which genomic window in the reference database is most structurally similar to a query fingerprint:

```python
import faiss
import json
import numpy as np

index     = faiss.read_index("/app/tmp/DATABASE_CONCEPT/data/locus_centroids.faiss")
with open("/app/tmp/DATABASE_CONCEPT/data/locus_centroids.locus_keys.json") as f:
    locus_keys = [tuple(k) for k in json.load(f)]  # [(chr, start_bp, end_bp), ...]

# Query with a 32-dim fingerprint
query_fp = np.random.randn(1, 32).astype(np.float32)
faiss.normalize_L2(query_fp)

D, I = index.search(query_fp, k=5)   # D: similarities, I: indices

for score, idx in zip(D[0], I[0]):
    if idx < 0:
        continue
    chr_, start_bp, end_bp = locus_keys[idx]
    print(f"{chr_}:{start_bp}-{end_bp}  score={score:.4f}")
```

---

## Interpreting Structural Similarity Scores

| Score Range | Biological Meaning |
|-------------|-------------------|
| 0.85–1.00 | Near-identical 3D structure; same cell type / replicate |
| 0.70–0.85 | Highly similar; related cell types (e.g., K562 vs KBM-7) |
| 0.50–0.70 | Partially similar; shared A/B compartment identity, different TADs |
| 0.30–0.50 | Divergent TAD organisation |
| 0.00–0.30 | Structurally unique locus; candidate novel rearrangement |
| < 0.00 | Anti-correlated (unusual; check data quality) |

---

## Troubleshooting

**`Database is read_only`**
→ Ensure `database.py` finished successfully before querying.

**All similarity scores ≈ 0**
→ The model may not have converged. Check that `active_codes > 100/512` in training logs.

**Missing loci (no matches for a chromosome)**
→ That chromosome was not present in any reference sample, or had <50% valid tiles during preprocessing.

**Slow query (>1 min per mcool)**
→ Set `num_workers=0` in `build_inference_loader` to avoid multiprocessing overhead on a single machine.
