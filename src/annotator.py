"""
annotator.py — Biological annotation of Hi-C windows using ENCODE cCREs.

Per-sample cCRE files (ENCODE v3 format, bed.gz) are loaded and indexed
by chromosome for O(log n) window overlap queries.

cCRE categories in the data:
  PLS          — Promoter-Like Signatures
  pELS         — Proximal Enhancer-Like Signatures
  dELS         — Distal Enhancer-Like Signatures
  CA-CTCF      — Chromatin Accessible + CTCF binding
  CA-TF        — Chromatin Accessible + TF binding
  CA-H3K4me3   — Chromatin Accessible + H3K4me3
  CA-only      — Chromatin Accessible only
  Low-DNase    — Low DNase accessibility (background)
  High-H3K27ac — High H3K27ac signal
  High-H3K4me3 — High H3K4me3 signal

For each 256×100kb window we compute:
  - Count and density of each cCRE category
  - Total active regulatory elements (non Low-DNase)
  - Regulatory complexity score (Shannon entropy over categories)
  - Whether CTCF sites are enriched (boundary-associated)
  - Promoter/enhancer ratio

These are stored as a fixed-length annotation vector in DuckDB alongside
the fingerprint for retrieval and differential comparison.
"""

import gzip
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── cCRE category vocabulary ──────────────────────────────────────────────────
# Fixed order — determines annotation vector layout
CCRE_CATEGORIES = [
    "PLS",           # 0 — promoter
    "pELS",          # 1 — proximal enhancer
    "dELS",          # 2 — distal enhancer
    "CA-CTCF",       # 3 — CTCF
    "CA-TF",         # 4 — TF binding
    "CA-H3K4me3",    # 5 — H3K4me3
    "CA-only",       # 6 — open chromatin
    "High-H3K27ac",  # 7 — active histone
    "High-H3K4me3",  # 8 — active histone
    "Low-DNase",     # 9 — background/inactive
]
N_CCRE_CATS   = len(CCRE_CATEGORIES)
CAT_TO_IDX    = {c: i for i, c in enumerate(CCRE_CATEGORIES)}

# Indices that represent "active" regulatory elements (exclude Low-DNase)
ACTIVE_CATS   = set(range(9))  # all except index 9 (Low-DNase)

# Annotation vector dimension:
#   N_CCRE_CATS counts  +  n_active  +  complexity  +  ctcf_density  +
#   promoter_ratio  +  enhancer_ratio  +  total_density
ANNOT_DIM = N_CCRE_CATS + 6
ANNOT_NAMES = (
    CCRE_CATEGORIES
    + ["n_active", "regulatory_complexity", "ctcf_density",
       "promoter_ratio", "enhancer_ratio", "total_density"]
)


# ── cCRE index (per-sample) ───────────────────────────────────────────────────

class CcreIndex:
    """
    Lightweight interval index for one sample's cCRE BED file.
    
    Internals: per-chromosome sorted arrays of (start, end, cat_idx).
    Overlap queries use np.searchsorted → O(log n + k).
    """

    def __init__(self, bed_gz_path: str):
        self.path = bed_gz_path
        # chrom → sorted np arrays: starts[N], ends[N], cat_idxs[N]
        self._starts:   Dict[str, np.ndarray] = {}
        self._ends:     Dict[str, np.ndarray] = {}
        self._cat_idxs: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        raw: Dict[str, List] = defaultdict(list)
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                chrom  = parts[0]
                start  = int(parts[1])
                end    = int(parts[2])
                cat    = parts[9].strip()
                ci     = CAT_TO_IDX.get(cat, CAT_TO_IDX.get("Low-DNase"))
                raw[chrom].append((start, end, ci))

        for chrom, records in raw.items():
            records.sort(key=lambda x: x[0])
            arr = np.array(records, dtype=np.int64)
            self._starts[chrom]   = arr[:, 0]
            self._ends[chrom]     = arr[:, 1]
            self._cat_idxs[chrom] = arr[:, 2]

    def query_window(self, chrom: str, win_start: int, win_end: int) -> np.ndarray:
        """
        Return count vector [N_CCRE_CATS] of cCREs overlapping [win_start, win_end).
        O(log n + k) where k = number of overlapping elements.
        """
        counts = np.zeros(N_CCRE_CATS, dtype=np.int32)
        if chrom not in self._starts:
            return counts

        starts   = self._starts[chrom]
        ends     = self._ends[chrom]
        cat_idxs = self._cat_idxs[chrom]

        right = int(np.searchsorted(starts, win_end, side="left"))
        mask  = ends[:right] > win_start
        hits  = cat_idxs[:right][mask]
        for ci in hits:
            counts[ci] += 1
        return counts

    def query_window_with_coords(
        self,
        chrom:     str,
        win_start: int,
        win_end:   int,
    ) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
        """
        Return (count_vector, coords_by_category) for cCREs overlapping window.

        coords_by_category: {cat_name: [(start_bp, end_bp), ...]}
        Sorted by position within each category.
        """
        counts = np.zeros(N_CCRE_CATS, dtype=np.int32)
        coords: Dict[str, List[Tuple[int, int]]] = {c: [] for c in CCRE_CATEGORIES}

        if chrom not in self._starts:
            return counts, coords

        starts   = self._starts[chrom]
        ends     = self._ends[chrom]
        cat_idxs = self._cat_idxs[chrom]

        right = int(np.searchsorted(starts, win_end, side="left"))
        mask  = ends[:right] > win_start
        hit_indices = np.where(mask)[0]

        for i in hit_indices:
            ci   = int(cat_idxs[i])
            s    = int(starts[i])
            e    = int(ends[i])
            counts[ci] += 1
            coords[CCRE_CATEGORIES[ci]].append((s, e))

        # Sort each category by position
        for cat in CCRE_CATEGORIES:
            coords[cat].sort(key=lambda x: x[0])

        return counts, coords

    def chroms(self):
        return list(self._starts.keys())


# ── Annotation vector construction ───────────────────────────────────────────

def counts_to_annot_vector(counts: np.ndarray, win_size_bp: int) -> np.ndarray:
    """
    Convert raw cCRE count vector [N_CCRE_CATS] to full annotation vector [ANNOT_DIM].
    
    win_size_bp: window size in base pairs (for density normalisation).
    """
    v = np.zeros(ANNOT_DIM, dtype=np.float32)
    v[:N_CCRE_CATS] = counts.astype(np.float32)

    total     = counts.sum()
    n_active  = counts[:9].sum()                  # exclude Low-DNase
    density   = total / max(win_size_bp, 1) * 1e6  # per Mb

    # Shannon entropy over active categories (regulatory complexity)
    active_counts = counts[:9].astype(np.float64)
    if active_counts.sum() > 0:
        p      = active_counts / active_counts.sum()
        p_nz   = p[p > 0]
        entropy = float(-np.sum(p_nz * np.log2(p_nz)))
    else:
        entropy = 0.0

    ctcf_density    = float(counts[3]) / max(win_size_bp, 1) * 1e6
    promoter_ratio  = float(counts[0]) / max(n_active, 1)
    enhancer_ratio  = float(counts[1] + counts[2]) / max(n_active, 1)

    v[N_CCRE_CATS + 0] = float(n_active)
    v[N_CCRE_CATS + 1] = entropy
    v[N_CCRE_CATS + 2] = ctcf_density
    v[N_CCRE_CATS + 3] = promoter_ratio
    v[N_CCRE_CATS + 4] = enhancer_ratio
    v[N_CCRE_CATS + 5] = density
    return v


def annotate_window(
    index:      CcreIndex,
    chrom:      str,
    start_bp:   int,
    end_bp:     int,
) -> np.ndarray:
    """Return [ANNOT_DIM] annotation vector for a single genomic window."""
    counts   = index.query_window(chrom, start_bp, end_bp)
    win_size = end_bp - start_bp
    return counts_to_annot_vector(counts, win_size)


# ── Multi-sample annotator ────────────────────────────────────────────────────

class MultiSampleAnnotator:
    """
    Holds one CcreIndex per sample.  Used during ingestion to annotate each
    window with both the reference sample's cCRE profile and the query's.
    
    Also provides differential annotation:
      diff = query_annot - reference_annot  →  which cCRE categories change
    """

    def __init__(self, ccre_registry: Dict[str, str]):
        """
        ccre_registry: {sample_id: path_to_bed_gz}
        """
        self.indices: Dict[str, CcreIndex] = {}
        for sample_id, path in ccre_registry.items():
            if Path(path).exists():
                print(f"[annotator] Loading cCREs for {sample_id}...", flush=True)
                self.indices[sample_id] = CcreIndex(path)
            else:
                print(f"[annotator] WARNING: cCRE file not found for {sample_id}: {path}")

    def annotate(
        self,
        sample_id: str,
        chrom:     str,
        start_bp:  int,
        end_bp:    int,
    ) -> Optional[np.ndarray]:
        """Return [ANNOT_DIM] vector or None if sample not registered."""
        if sample_id not in self.indices:
            return None
        return annotate_window(self.indices[sample_id], chrom, start_bp, end_bp)

    def differential(
        self,
        sample_a:  str,
        sample_b:  str,
        chrom:     str,
        start_bp:  int,
        end_bp:    int,
    ) -> Optional[np.ndarray]:
        """
        Return signed difference vector [ANNOT_DIM]: A - B.
        Positive values → more in sample_a; negative → more in sample_b.
        """
        a = self.annotate(sample_a, chrom, start_bp, end_bp)
        b = self.annotate(sample_b, chrom, start_bp, end_bp)
        if a is None or b is None:
            return None
        return a - b

    def summarise_window(
        self,
        sample_id: str,
        chrom:     str,
        start_bp:  int,
        end_bp:    int,
    ) -> Dict:
        """Human-readable summary dict for one window."""
        v = self.annotate(sample_id, chrom, start_bp, end_bp)
        if v is None:
            return {}
        summary = {}
        for i, name in enumerate(ANNOT_NAMES):
            summary[name] = float(v[i])
        summary["dominant_category"] = CCRE_CATEGORIES[
            int(np.argmax(v[:N_CCRE_CATS]))
        ]
        return summary


# ── Differential annotation report ───────────────────────────────────────────

def differential_report(
    annotator:   MultiSampleAnnotator,
    query_id:    str,
    ref_id:      str,
    chrom:       str,
    start_bp:    int,
    end_bp:      int,
    threshold:   float = 10.0,  # min count difference to flag
) -> List[str]:
    """
    Returns list of human-readable strings describing regulatory differences
    between query_id and ref_id at the given locus.
    
    E.g.: ["PLS: +12 more in query (promoters)", "CA-CTCF: -8 (CTCF anchors lost)"]
    """
    diff = annotator.differential(query_id, ref_id, chrom, start_bp, end_bp)
    if diff is None:
        return []

    lines = []
    for i, cat in enumerate(CCRE_CATEGORIES):
        delta = diff[i]
        if abs(delta) >= threshold:
            direction = "more in query" if delta > 0 else "fewer in query"
            lines.append(f"{cat}: {delta:+.0f} ({direction})")

    # Summarise scalar features
    scalar_diffs = {
        "regulatory_complexity": (diff[N_CCRE_CATS + 1], 0.5),
        "ctcf_density":          (diff[N_CCRE_CATS + 2], 0.1),
        "promoter_ratio":        (diff[N_CCRE_CATS + 3], 0.05),
        "enhancer_ratio":        (diff[N_CCRE_CATS + 4], 0.05),
    }
    for feat, (delta, thr) in scalar_diffs.items():
        if abs(delta) >= thr:
            direction = "higher" if delta > 0 else "lower"
            lines.append(f"{feat}: {delta:+.3f} ({direction} in query)")
    return lines


# ── Permutation test ──────────────────────────────────────────────────────────

def permutation_pvalue(
    query_fp:   np.ndarray,
    ref_fps:    np.ndarray,
    n_perm:     int = 500,
    rng:        Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Non-parametric permutation test for fingerprint similarity.
    
    Null hypothesis: the query fingerprint is no more similar to the reference
    set than a random shuffled fingerprint.
    
    Returns: (observed_mean_cosine, p_value)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    q_norm   = query_fp / (np.linalg.norm(query_fp) + 1e-8)
    r_norm   = ref_fps  / (np.linalg.norm(ref_fps, axis=1, keepdims=True) + 1e-8)
    observed = float(np.mean(r_norm @ q_norm))

    count = 0
    for _ in range(n_perm):
        shuffled    = rng.permutation(query_fp)
        s_norm      = shuffled / (np.linalg.norm(shuffled) + 1e-8)
        null_sim    = float(np.mean(r_norm @ s_norm))
        if null_sim >= observed:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value


# ── Convenience: annotation vector → bytes for DuckDB ────────────────────────

def annot_to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def bytes_to_annot(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()
