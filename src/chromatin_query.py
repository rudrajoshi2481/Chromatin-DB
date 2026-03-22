#!/usr/bin/env python3
"""
chromatin_query.py — Interactive CLI for Hi-C structural fingerprint similarity search.

Commands:
  query    Query a .mcool file against reference fingerprints in the database
  compare  Compare two .mcool files directly (fingerprint + cCRE)
  list     List available reference mcool and cCRE files
  test     End-to-end accuracy test across known cell lines

Usage:
  python src/chromatin_query.py query   --input GM12878.mcool
  python src/chromatin_query.py compare --a GM12878.mcool --b K562.mcool
  python src/chromatin_query.py list
  python src/chromatin_query.py test    --n-pairs 5
"""

import sys
import argparse
import gzip
import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).parent.parent
_MCOOL_DIR   = Path("/data/joshi/Generative_experiment/Chromatin-CLI/data/downloads/mcool")
_CCRE_DIR    = Path("/data/joshi/Generative_experiment/Chromatin-CLI/data/downloads/ccre")
_CKPT        = _ROOT / "runs/67cl_512codes/checkpoints/mqvae_epoch181_best.pt"
_PROCESSED   = Path("/data/joshi/Generative_experiment/Chromatin-CLI/data/processed")

RESOLUTION   = 100_000
TILE_SIZE    = 256
TILE_STEP    = 128
CHROMOSOMES  = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# ── Dependency imports ────────────────────────────────────────────────────────

import numpy as np
import torch

try:
    from tabulate import tabulate
    _HAS_TAB = True
except ImportError:
    _HAS_TAB = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
    console = Console()
except ImportError:
    _HAS_RICH = False
    class _FakeConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = _FakeConsole()

sys.path.insert(0, str(_ROOT / "src"))

# ── cCRE categories ───────────────────────────────────────────────────────────

CCRE_CATEGORIES = ["PLS", "pELS", "dELS", "CA-CTCF", "CA-TF",
                   "CA-H3K4me3", "CA-only", "High-H3K27ac", "High-H3K4me3", "Low-DNase"]
CAT_TO_IDX      = {c: i for i, c in enumerate(CCRE_CATEGORIES)}
N_CATS          = len(CCRE_CATEGORIES)

# ── Accession-stripping regex (same as preprocess.py) ─────────────────────────

_ACCESSION_RE = re.compile(
    r"[_.]?(?:4DNF[A-Z0-9]+|ENCFF[A-Z0-9]+|ENCSR[A-Z0-9]+|GSM\d+|SRR\d+|ERR\d+)$",
    re.IGNORECASE,
)
_REPLICATE_RE = re.compile(
    r"[_.]?(?:rep\d+|replicate\d*|r\d+(?=[_.]|$)|clone[_-]?[A-Z0-9]+|auxin\d*(?:h|hr)?|\d+h(?:r)?(?=[_.]|$)|DMSO|IAA|degron|hom|het)[_.]?.*$",
    re.IGNORECASE,
)

def _cell_name(stem: str) -> str:
    name = _ACCESSION_RE.sub("", stem).strip("_.-")
    name = _REPLICATE_RE.sub("", name).strip("_.-")
    return name or stem


# ═══════════════════════════════════════════════════════════════════════════════
# Model + Fingerprint Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(ckpt_path: Path, device: torch.device):
    """Load MQVAE from checkpoint."""
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    arch  = ckpt.get("arch") or {}
    # Strip non-constructor keys saved in arch dict
    _MODEL_KEYS = {"n_codes", "use_classifier_head", "n_cell_types",
                   "use_masking", "use_film"}
    arch = {k: v for k, v in arch.items() if k in _MODEL_KEYS}
    sys.path.insert(0, str(_ROOT / "src"))
    from model import MQVAE
    model = MQVAE(**arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("epoch", "?")


def _get_cooler(mcool_path: Path):
    """Open .mcool at the target resolution."""
    import cooler
    uri = f"{mcool_path}::/resolutions/{RESOLUTION}"
    return cooler.Cooler(str(uri))


# Import OE computation from preprocess.py to guarantee identical transforms
try:
    from preprocess import compute_oe_matrix as _preprocess_oe
    _HAS_PREPROCESS_OE = True
except Exception:
    _HAS_PREPROCESS_OE = False


def _compute_oe(clr, chrom: str) -> Optional[np.ndarray]:
    """
    Compute log2(OE) matrix — delegates to preprocess.compute_oe_matrix
    so the transform is byte-for-byte identical to training data.
    Falls back to local simple_oe if preprocess import fails.
    """
    if _HAS_PREPROCESS_OE:
        try:
            oe = _preprocess_oe(clr, chrom, nproc=4)
            if oe is None or oe.shape[0] < TILE_SIZE:
                return None
            return oe
        except Exception:
            pass
    # Fallback: simple diagonal-mean log2 OE
    try:
        mat = clr.matrix(balance=True).fetch(chrom)
        if mat is None or mat.shape[0] < TILE_SIZE:
            return None
        mat = np.array(mat, dtype=np.float32)
        mat = np.nan_to_num(mat, nan=0.0)
        n   = mat.shape[0]
        oe  = np.zeros_like(mat)
        for d in range(n):
            diag     = np.diag(mat, d)
            pos      = diag[diag > 0]
            diag_exp = float(pos.mean()) if len(pos) > 0 else 0.0
            if diag_exp > 0:
                oe[np.arange(n - d), np.arange(n - d) + d] = diag / diag_exp
                if d > 0:
                    oe[np.arange(n - d) + d, np.arange(n - d)] = diag / diag_exp
        log_oe = np.log2(oe + 1e-6)
        log_oe = np.nan_to_num(log_oe, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(log_oe, -5.0, 5.0).astype(np.float32)
    except Exception:
        return None


def _tile_oe(oe: np.ndarray, chrom: str) -> List[Dict]:
    """Tile OE matrix into overlapping windows."""
    n     = oe.shape[0]
    tiles = []
    for start in range(0, n - TILE_SIZE + 1, TILE_STEP):
        end      = start + TILE_SIZE
        tile_mat = oe[start:end, start:end].copy()
        # Check near-diagonal bands are non-trivial (±20 bins)
        diag_vals = np.concatenate([np.diag(tile_mat, k) for k in range(-20, 21)])
        valid_frac = float((diag_vals > 0).mean())
        if valid_frac < 0.05:
            continue
        tile_mat = np.nan_to_num(tile_mat, nan=0.0)
        tiles.append({
            "matrix":   tile_mat.astype(np.float32),
            "chr":      chrom,
            "start_bp": int(start * RESOLUTION),
            "end_bp":   int(end   * RESOLUTION),
        })
    return tiles


@torch.no_grad()
def extract_fingerprints(
    mcool_path: Path,
    model,
    device: torch.device,
    assay_id: int = 0,
    max_tiles: int = 2000,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract fingerprints from all tiles in a .mcool file."""
    clr    = _get_cooler(mcool_path)
    avail  = set(clr.chromnames)
    chroms = [c for c in CHROMOSOMES if c in avail]

    fps, meta = [], []
    aid_t = torch.tensor([assay_id], dtype=torch.long, device=device)

    for chrom in chroms:
        oe = _compute_oe(clr, chrom)
        if oe is None:
            continue
        tiles = _tile_oe(oe, chrom)
        for tile in tiles:
            mat_t = torch.from_numpy(tile["matrix"]).unsqueeze(0).unsqueeze(0).to(device)
            fp    = model.encode_fingerprint(mat_t, aid_t).squeeze(0).cpu().float().numpy()
            fps.append(fp)
            meta.append(tile)
            if len(fps) >= max_tiles:
                break
        if len(fps) >= max_tiles:
            break

    if not fps:
        return np.zeros((0, 64), dtype=np.float32), []

    return np.stack(fps).astype(np.float32), meta


# ═══════════════════════════════════════════════════════════════════════════════
# cCRE Loading + Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ccre_counts(bed_gz: Path) -> Dict[str, np.ndarray]:
    """Load cCRE bed.gz → {chr: sorted arrays (starts, ends, cat_idxs)}."""
    raw = defaultdict(list)
    opener = gzip.open if str(bed_gz).endswith(".gz") else open
    with opener(str(bed_gz), "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 10:
                continue
            chrom = parts[0]
            if chrom not in set(CHROMOSOMES):
                continue
            start = int(parts[1])
            end   = int(parts[2])
            cat   = parts[9].strip()
            ci    = CAT_TO_IDX.get(cat, CAT_TO_IDX.get("Low-DNase", 9))
            raw[chrom].append((start, end, ci))
    result = {}
    for chrom, recs in raw.items():
        recs.sort(key=lambda x: x[0])
        arr = np.array(recs, dtype=np.int64)
        result[chrom] = arr
    return result


def _ccre_window_counts(arr: np.ndarray, start_bp: int, end_bp: int) -> np.ndarray:
    """Count cCREs per category overlapping a window."""
    counts = np.zeros(N_CATS, dtype=np.int32)
    if arr is None or len(arr) == 0:
        return counts
    starts   = arr[:, 0]
    ends     = arr[:, 1]
    cat_idxs = arr[:, 2]
    right = int(np.searchsorted(starts, end_bp, side="left"))
    mask  = ends[:right] > start_bp
    for ci in cat_idxs[:right][mask]:
        counts[ci] += 1
    return counts


def compare_ccre(ccre_a: Path, ccre_b: Path) -> Dict:
    """Compare cCRE profiles between two samples (genome-wide summary)."""
    idx_a = _load_ccre_counts(ccre_a)
    idx_b = _load_ccre_counts(ccre_b)

    counts_a = np.zeros(N_CATS, dtype=np.int64)
    counts_b = np.zeros(N_CATS, dtype=np.int64)
    for chrom in CHROMOSOMES:
        if chrom in idx_a:
            for ci in idx_a[chrom][:, 2]:
                counts_a[ci] += 1
        if chrom in idx_b:
            for ci in idx_b[chrom][:, 2]:
                counts_b[ci] += 1

    diff  = counts_a.astype(np.int64) - counts_b.astype(np.int64)
    total_a = counts_a.sum()
    total_b = counts_b.sum()

    return {
        "counts_a": counts_a,
        "counts_b": counts_b,
        "diff":     diff,
        "total_a":  int(total_a),
        "total_b":  int(total_b),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Database fingerprint index (from processed dir)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_reference_index(
    processed_dir: Path,
    model,
    device: torch.device,
    max_per_sample: int = 500,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build in-memory reference fingerprint matrix from processed tiles.
    Returns (fp_matrix [N, D], labels [N]) where label=cell_line_name.
    """
    fps, labels = [], []
    if not processed_dir.exists():
        return np.zeros((0, 64), dtype=np.float32), []

    sample_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
    aid_t = torch.tensor([0], dtype=torch.long, device=device)

    for sample_dir in sample_dirs:
        cell_name = _cell_name(sample_dir.name)
        npz_files = sorted(sample_dir.glob("*.npz"))
        count = 0
        for npz_path in npz_files:
            try:
                data = np.load(str(npz_path), mmap_mode="r")
                mats = data["matrices"]
                n    = min(len(mats), max_per_sample - count)
                for i in range(n):
                    mat = mats[i].astype(np.float32)
                    t   = torch.from_numpy(mat).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        fp = model.encode_fingerprint(t, aid_t).squeeze(0).cpu().float().numpy()
                    fps.append(fp)
                    labels.append(cell_name)
                    count += 1
                data.close()
            except Exception:
                continue
            if count >= max_per_sample:
                break

    if not fps:
        return np.zeros((0, 64), dtype=np.float32), []
    return np.stack(fps).astype(np.float32), labels


# ═══════════════════════════════════════════════════════════════════════════════
# Similarity Computation
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T   # [N_query, N_ref]


def compute_cell_type_similarity(
    query_fps: np.ndarray,
    ref_fps:   np.ndarray,
    ref_labels: List[str],
) -> List[Tuple[str, float, int]]:
    """
    Returns sorted list of (cell_type, mean_similarity, n_matched_loci).
    n_matched_loci = number of query loci where this cell type is the best match.
    """
    sim_mat = cosine_sim_matrix(query_fps, ref_fps)   # [Nq, Nr]

    # Per-query locus: best cell type
    label_arr   = np.array(ref_labels)
    unique_cts  = sorted(set(ref_labels))

    # Mean similarity per cell type (averaged across all query tiles)
    ct_mean_sim = {}
    for ct in unique_cts:
        mask = label_arr == ct
        if mask.sum() == 0:
            continue
        ct_sims = sim_mat[:, mask].max(axis=1)  # per query: best ref of this CT
        ct_mean_sim[ct] = float(ct_sims.mean())

    # Best match per query locus
    best_ct_per_query = []
    for i in range(sim_mat.shape[0]):
        best_j  = int(sim_mat[i].argmax())
        best_ct = ref_labels[best_j]
        best_ct_per_query.append(best_ct)

    from collections import Counter
    match_counts = Counter(best_ct_per_query)

    results = []
    for ct, mean_sim in ct_mean_sim.items():
        results.append((ct, mean_sim, match_counts.get(ct, 0)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_divergent_loci(
    query_fps:   np.ndarray,
    ref_fps:     np.ndarray,
    ref_labels:  List[str],
    query_meta:  List[Dict],
    top_ct:      str,
    top_n:       int = 10,
) -> List[Dict]:
    """Return loci where the query diverges most from the top matching cell type."""
    label_arr = np.array(ref_labels)
    ct_mask   = label_arr == top_ct
    if ct_mask.sum() == 0:
        return []

    ct_fps  = ref_fps[ct_mask]
    sim_mat = cosine_sim_matrix(query_fps, ct_fps)  # [Nq, Nct]
    per_locus_sim = sim_mat.max(axis=1)             # [Nq]

    # Z-score
    mean_s  = per_locus_sim.mean()
    std_s   = per_locus_sim.std() + 1e-8

    sorted_idx = np.argsort(per_locus_sim)[:top_n]

    divergent = []
    for i in sorted_idx:
        m = query_meta[i]
        z = float((per_locus_sim[i] - mean_s) / std_s)
        divergent.append({
            "locus":      f"{m['chr']}:{m['start_bp']//1_000_000:.1f}M-{m['end_bp']//1_000_000:.1f}M",
            "chr":        m["chr"],
            "start_bp":   m["start_bp"],
            "end_bp":     m["end_bp"],
            "similarity": float(per_locus_sim[i]),
            "z_score":    z,
        })
    return divergent


# ═══════════════════════════════════════════════════════════════════════════════
# Table Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _conf_badge(sim: float) -> str:
    if sim >= 0.90: return "VERY HIGH ★★★"
    if sim >= 0.80: return "HIGH     ★★☆"
    if sim >= 0.65: return "MEDIUM   ★☆☆"
    return                   "LOW      ☆☆☆"

def _sim_bar(sim: float, width: int = 16) -> str:
    n = int(round(sim * width))
    return "█" * n + "░" * (width - n)

def _diff_arrow(d: int) -> str:
    if d > 0:  return f"+{d} ▲"
    if d < 0:  return f"{d} ▼"
    return              "  ="

def _print_header(title: str):
    if _HAS_RICH:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print("\n" + "═"*60)
        print(f"  {title}")
        print("═"*60)

def _print_section(title: str):
    if _HAS_RICH:
        console.print(f"\n[bold yellow]{'─'*4} {title} {'─'*(50-len(title))}[/bold yellow]")
    else:
        print(f"\n{'─'*4} {title} {'─'*(50-len(title))}")

def _print_table(rows, headers, fmt="simple"):
    if _HAS_TAB:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print(tabulate(rows, headers=headers) if _HAS_TAB else str(rows))


def _mcool_label(path: Path) -> str:
    return _cell_name(path.stem)


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery helpers
# ═══════════════════════════════════════════════════════════════════════════════

def discover_mcool_files() -> Dict[str, List[Path]]:
    """Group mcool files by canonical cell name."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(_MCOOL_DIR.glob("*.mcool")):
        groups[_cell_name(p.stem)].append(p)
    return dict(groups)


def discover_ccre_files() -> Dict[str, Path]:
    """Map canonical cell name → cCRE bed.gz path."""
    result = {}
    for p in sorted(_CCRE_DIR.glob("*.bed.gz")):
        stem = p.name.replace(".bed.gz", "")
        name = _ACCESSION_RE.sub("", stem).strip("_.-")
        result[name] = p
    return result


def find_ccre_for_cell(cell_name: str, ccre_map: Dict[str, Path]) -> Optional[Path]:
    """Find best matching cCRE file for a cell name."""
    if cell_name in ccre_map:
        return ccre_map[cell_name]
    for k, v in ccre_map.items():
        if cell_name.lower().startswith(k.lower()) or k.lower().startswith(cell_name.lower()):
            return v
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: list
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_list(args):
    _print_header("AVAILABLE REFERENCE FILES")

    mcool_groups = discover_mcool_files()
    ccre_map     = discover_ccre_files()

    _print_section("mcool Files (Hi-C)")
    rows = []
    for cell, paths in sorted(mcool_groups.items()):
        has_ccre = "✓" if find_ccre_for_cell(cell, ccre_map) else "✗"
        total_gb = sum(p.stat().st_size for p in paths) / 1e9
        rows.append([cell, len(paths), f"{total_gb:.2f} GB", has_ccre])
    _print_table(rows, ["Cell Line", "# Files", "Total Size", "cCRE?"])
    print(f"\n  Total: {len(mcool_groups)} cell lines, {sum(len(v) for v in mcool_groups.values())} mcool files")

    _print_section("cCRE Files (ENCODE Regulatory Elements)")
    rows = []
    for cell, path in sorted(ccre_map.items()):
        size_mb = path.stat().st_size / 1e6
        rows.append([cell, path.name, f"{size_mb:.1f} MB"])
    _print_table(rows, ["Cell Line", "File", "Size"])


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: query
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_query(args):
    mcool_path = Path(args.input)
    if not mcool_path.exists():
        # Try in mcool dir
        mcool_path = _MCOOL_DIR / args.input
    if not mcool_path.exists():
        console.print(f"[red]✗ File not found: {args.input}[/red]")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _CKPT
    if not ckpt_path.exists():
        console.print(f"[red]✗ Checkpoint not found: {ckpt_path}[/red]")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_header("CHROMATIN STRUCTURAL QUERY")

    # ── Load model ───────────────────────────────────────────────────────────
    _print_section("Loading Model")
    t0    = time.time()
    model, epoch = _load_model(ckpt_path, device)
    query_cell   = _cell_name(mcool_path.stem)
    file_gb      = mcool_path.stat().st_size / 1e9

    _print_table(
        [
            ["Input File",   mcool_path.name],
            ["Cell Line",    query_cell],
            ["File Size",    f"{file_gb:.2f} GB"],
            ["Model Epoch",  epoch],
            ["Device",       str(device).upper()],
        ],
        ["Parameter", "Value"],
    )

    # ── Extract query fingerprints ───────────────────────────────────────────
    _print_section("Extracting Fingerprints")
    if _HAS_RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Processing chromosomes...", total=None)
            query_fps, query_meta = extract_fingerprints(
                mcool_path, model, device, max_tiles=args.max_tiles, verbose=False
            )
            prog.update(task, description=f"Done — {len(query_fps)} tiles extracted")
    else:
        print("  Extracting fingerprints (this may take a minute)...")
        query_fps, query_meta = extract_fingerprints(
            mcool_path, model, device, max_tiles=args.max_tiles, verbose=True
        )

    if len(query_fps) == 0:
        console.print("[red]✗ No valid tiles extracted. Check resolution or file format.[/red]")
        sys.exit(1)

    print(f"  ✓ Extracted {len(query_fps)} tiles")

    # ── Build reference index ────────────────────────────────────────────────
    _print_section("Loading Reference Database")
    print("  Building reference fingerprint index from processed tiles...")
    ref_fps, ref_labels = _build_reference_index(
        _PROCESSED, model, device, max_per_sample=args.ref_per_sample
    )

    if len(ref_fps) == 0:
        console.print("[red]✗ No reference fingerprints found. Check processed directory.[/red]")
        sys.exit(1)

    unique_refs = len(set(ref_labels))
    print(f"  ✓ Reference: {len(ref_fps)} tiles from {unique_refs} cell lines")

    # ── Compute similarity ───────────────────────────────────────────────────
    _print_section("Computing Structural Similarity")
    results = compute_cell_type_similarity(query_fps, ref_fps, ref_labels)
    top5    = results[:5]
    top_ct, top_sim, _ = results[0] if results else ("unknown", 0.0, 0)

    rows = []
    for rank, (ct, sim, n_match) in enumerate(top5, 1):
        same = "★ MATCH" if ct.lower() == query_cell.lower() else ""
        rows.append([
            rank, ct,
            f"{sim:.4f}",
            _sim_bar(sim),
            _conf_badge(sim),
            n_match,
            same,
        ])
    _print_table(rows, ["#", "Cell Type", "Similarity", "Visual", "Confidence", "Loci Won", ""])

    # ── Divergent loci ───────────────────────────────────────────────────────
    _print_section(f"Divergent Loci (vs. {top_ct})")
    divs = find_divergent_loci(query_fps, ref_fps, ref_labels, query_meta, top_ct, top_n=10)
    if divs:
        div_rows = []
        for d in divs:
            div_rows.append([d["locus"], f"{d['similarity']:.4f}", f"{d['z_score']:.2f}"])
        _print_table(div_rows, ["Locus", "Similarity", "Z-score"])
    else:
        print("  (no divergent loci found)")

    # ── cCRE comparison ──────────────────────────────────────────────────────
    ccre_map = discover_ccre_files()
    query_ccre = find_ccre_for_cell(query_cell, ccre_map)
    ref_ccre   = find_ccre_for_cell(top_ct, ccre_map)

    if query_ccre and ref_ccre and not args.no_ccre:
        _print_section(f"Regulatory Landscape — {query_cell} vs {top_ct} (cCRE)")
        ccre_result = compare_ccre(query_ccre, ref_ccre)
        ccre_rows = []
        for i, cat in enumerate(CCRE_CATEGORIES):
            a   = int(ccre_result["counts_a"][i])
            b   = int(ccre_result["counts_b"][i])
            d   = int(ccre_result["diff"][i])
            if a == 0 and b == 0:
                continue
            ccre_rows.append([cat, a, b, _diff_arrow(d)])
        _print_table(ccre_rows, ["cCRE Category", query_cell, top_ct, "Difference"])
    elif not args.no_ccre:
        _print_section("cCRE Comparison")
        print(f"  ℹ  No cCRE files found for {query_cell!r} or {top_ct!r}")

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_section("Summary")
    overall_sim_mean = np.mean([r[1] for r in results[:3]])
    summary_rows = [
        ["Top Match",        top_ct],
        ["Top Similarity",   f"{top_sim:.4f}"],
        ["Confidence",       _conf_badge(top_sim)],
        ["Tiles Analyzed",   len(query_fps)],
        ["Reference Lines",  unique_refs],
        ["Processing Time",  f"{elapsed:.1f}s"],
    ]
    _print_table(summary_rows, ["Metric", "Value"])

    # ── JSON output ──────────────────────────────────────────────────────────
    if args.output:
        out = {
            "query_file":  str(mcool_path),
            "query_cell":  query_cell,
            "top_matches": [{"cell_type": ct, "similarity": sim, "loci_won": n}
                            for ct, sim, n in results[:10]],
            "divergent_loci": divs,
            "n_tiles":     len(query_fps),
            "elapsed_s":   elapsed,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  ✓ Results saved → {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: compare
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_compare(args):
    path_a = Path(args.a) if Path(args.a).exists() else _MCOOL_DIR / args.a
    path_b = Path(args.b) if Path(args.b).exists() else _MCOOL_DIR / args.b

    for p, name in [(path_a, args.a), (path_b, args.b)]:
        if not p.exists():
            console.print(f"[red]✗ File not found: {name}[/red]")
            sys.exit(1)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _CKPT
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_header("DIRECT STRUCTURAL COMPARISON")

    cell_a = _cell_name(path_a.stem)
    cell_b = _cell_name(path_b.stem)

    info_rows = [
        ["Sample A", path_a.name, cell_a, f"{path_a.stat().st_size/1e9:.2f} GB"],
        ["Sample B", path_b.name, cell_b, f"{path_b.stat().st_size/1e9:.2f} GB"],
    ]
    _print_table(info_rows, ["", "File", "Cell Line", "Size"])

    model, epoch = _load_model(ckpt_path, device)

    _print_section("Extracting Fingerprints")
    print(f"  Processing {cell_a}...")
    fps_a, meta_a = extract_fingerprints(path_a, model, device,
                                          max_tiles=args.max_tiles, verbose=False)
    print(f"  Processing {cell_b}...")
    fps_b, meta_b = extract_fingerprints(path_b, model, device,
                                          max_tiles=args.max_tiles, verbose=False)
    print(f"  ✓ {len(fps_a)} tiles from A,  {len(fps_b)} tiles from B")

    # ── Fingerprint similarity ───────────────────────────────────────────────
    _print_section("Fingerprint Similarity (Structural)")
    sim_mat    = cosine_sim_matrix(fps_a, fps_b)  # [Na, Nb]
    mean_sim   = float(sim_mat.mean())
    max_sim    = float(sim_mat.max())
    per_locus  = sim_mat.max(axis=1)              # best match per A tile
    frac_high  = float((per_locus > 0.85).mean())

    _print_table(
        [
            ["Mean Similarity",    f"{mean_sim:.4f}", _sim_bar(mean_sim),    _conf_badge(mean_sim)],
            ["Max Similarity",     f"{max_sim:.4f}",  _sim_bar(max_sim),     _conf_badge(max_sim)],
            ["Loci > 0.85 match",  f"{frac_high:.1%}", _sim_bar(frac_high),  ""],
        ],
        ["Metric", "Value", "Visual", "Confidence"],
    )

    # ── Per-chromosome breakdown ─────────────────────────────────────────────
    _print_section("Per-Chromosome Similarity")
    chrom_sims: Dict[str, List[float]] = defaultdict(list)
    for i, m in enumerate(meta_a):
        chrom_sims[m["chr"]].append(float(per_locus[i]))

    chr_rows = []
    for chrom in CHROMOSOMES:
        if chrom not in chrom_sims:
            continue
        sims   = chrom_sims[chrom]
        mean_c = float(np.mean(sims))
        chr_rows.append([chrom, f"{mean_c:.4f}", _sim_bar(mean_c, 12), len(sims)])

    _print_table(chr_rows, ["Chrom", "Similarity", "Visual", "# Tiles"])

    # ── cCRE Comparison ──────────────────────────────────────────────────────
    ccre_map   = discover_ccre_files()
    ccre_a     = find_ccre_for_cell(cell_a, ccre_map)
    ccre_b     = find_ccre_for_cell(cell_b, ccre_map)

    if ccre_a and ccre_b and not args.no_ccre:
        _print_section(f"Regulatory Landscape (cCRE): {cell_a} vs {cell_b}")
        print("  Loading cCRE files...")
        ccre_result = compare_ccre(ccre_a, ccre_b)
        rows = []
        for i, cat in enumerate(CCRE_CATEGORIES):
            a = int(ccre_result["counts_a"][i])
            b = int(ccre_result["counts_b"][i])
            d = int(ccre_result["diff"][i])
            if a == 0 and b == 0:
                continue
            rows.append([cat, a, b, _diff_arrow(d)])
        _print_table(rows, ["cCRE Category", cell_a, cell_b, "Difference"])
        print(f"\n  Total cCREs — {cell_a}: {ccre_result['total_a']:,}   {cell_b}: {ccre_result['total_b']:,}")
    else:
        _print_section("cCRE Comparison")
        msg = f"  ℹ  cCRE files available: {cell_a}={'yes' if ccre_a else 'no'}, {cell_b}={'yes' if ccre_b else 'no'}"
        print(msg)

    # ── Divergent loci ───────────────────────────────────────────────────────
    _print_section(f"Most Divergent Loci (A differs from B)")
    z = (per_locus - per_locus.mean()) / (per_locus.std() + 1e-8)
    sorted_div = np.argsort(per_locus)[:10]
    div_rows = []
    for i in sorted_div:
        m = meta_a[i]
        div_rows.append([
            f"{m['chr']}:{m['start_bp']//1_000_000:.1f}M",
            f"{per_locus[i]:.4f}",
            f"{z[i]:.2f}",
        ])
    _print_table(div_rows, ["Locus", "Similarity", "Z-score"])

    # ── Summary ─────────────────────────────────────────────────────────────
    _print_section("Summary")
    verdict = (
        "STRUCTURALLY IDENTICAL" if mean_sim > 0.95 else
        "VERY SIMILAR"           if mean_sim > 0.88 else
        "SIMILAR"                if mean_sim > 0.75 else
        "DIFFERENT"
    )
    _print_table(
        [
            ["Sample A",         cell_a],
            ["Sample B",         cell_b],
            ["Mean Similarity",  f"{mean_sim:.4f}"],
            ["Verdict",          verdict],
        ],
        ["Metric", "Value"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: test
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_test(args):
    _print_header("END-TO-END ACCURACY TEST")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _CKPT
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, epoch = _load_model(ckpt_path, device)
    mcool_groups = discover_mcool_files()
    ccre_map     = discover_ccre_files()

    # Pick cell lines that have cCRE files available (richer test)
    test_cells = [c for c in sorted(mcool_groups.keys()) if find_ccre_for_cell(c, ccre_map)]
    if not test_cells:
        test_cells = sorted(mcool_groups.keys())

    test_cells = test_cells[:args.n_pairs]
    print(f"\n  Testing {len(test_cells)} cell lines: {', '.join(test_cells)}")

    # ── Build reference from processed dir ───────────────────────────────────
    _print_section("Building Reference Index")
    ref_fps, ref_labels = _build_reference_index(
        _PROCESSED, model, device, max_per_sample=200
    )
    if len(ref_fps) == 0:
        console.print("[red]✗ No reference fingerprints found.[/red]")
        sys.exit(1)

    unique_refs = sorted(set(ref_labels))
    print(f"  ✓ {len(ref_fps)} reference tiles from {len(unique_refs)} cell lines")

    # ── Per-cell-line accuracy test ──────────────────────────────────────────
    _print_section("Classification Accuracy Test")
    results_rows = []
    correct = 0
    total   = 0

    for cell_name in test_cells:
        paths = mcool_groups[cell_name]
        # Use first available mcool
        mcool_path = paths[0]

        try:
            fps, _ = extract_fingerprints(
                mcool_path, model, device, max_tiles=300, verbose=False
            )
        except Exception as e:
            results_rows.append([cell_name, mcool_path.name[:30], "ERROR", "—", "—", "✗"])
            total += 1
            continue

        if len(fps) == 0:
            results_rows.append([cell_name, mcool_path.name[:30], "NO TILES", "—", "—", "✗"])
            total += 1
            continue

        ct_results = compute_cell_type_similarity(fps, ref_fps, ref_labels)
        pred_ct, pred_sim, _ = ct_results[0] if ct_results else ("?", 0.0, 0)

        is_correct = (pred_ct.lower() == cell_name.lower())
        if is_correct:
            correct += 1
        total += 1

        status = "✓ CORRECT" if is_correct else f"✗ → {pred_ct}"
        results_rows.append([
            cell_name,
            mcool_path.name[:35],
            pred_ct,
            f"{pred_sim:.4f}",
            _sim_bar(pred_sim, 10),
            status,
        ])

    _print_table(results_rows,
                 ["True Cell", "Query File", "Predicted", "Sim", "Visual", "Result"])

    accuracy = correct / total if total > 0 else 0.0
    _print_section("Accuracy Summary")
    _print_table(
        [
            ["Correct",   f"{correct}/{total}"],
            ["Accuracy",  f"{accuracy:.1%}"],
            ["Model",     f"Epoch {epoch}"],
            ["Status",    "✓ PASS" if accuracy > 0.7 else "✗ FAIL"],
        ],
        ["Metric", "Value"],
    )

    # ── cCRE self-consistency test ───────────────────────────────────────────
    if not args.no_ccre:
        _print_section("cCRE Self-Consistency Test")
        ccre_cells = [(c, find_ccre_for_cell(c, ccre_map))
                      for c in test_cells if find_ccre_for_cell(c, ccre_map)]

        if len(ccre_cells) >= 2:
            print(f"  Comparing cCRE profiles across {len(ccre_cells)} cell lines...")
            ccre_rows = []
            for i, (ca, pa) in enumerate(ccre_cells):
                for cb, pb in ccre_cells[i+1:]:
                    res = compare_ccre(pa, pb)
                    total_diff = int(np.abs(res["diff"]).sum())
                    pct_diff   = total_diff / max(res["total_a"], res["total_b"], 1)
                    ccre_rows.append([ca, cb, res["total_a"], res["total_b"],
                                      total_diff, f"{pct_diff:.1%}"])
            _print_table(ccre_rows, ["Cell A", "Cell B", "cCRE A", "cCRE B",
                                      "Total Diff", "% Diff"])
        else:
            print("  (need ≥2 cells with cCRE files for this test)")

    _print_section("Done")
    print(f"  Accuracy: {accuracy:.1%} | {correct}/{total} correct\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="chromatin-query",
        description="Interactive Hi-C structural similarity query CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/chromatin_query.py list
  python src/chromatin_query.py query --input GM12878_4DNFI2A4OBS9.mcool
  python src/chromatin_query.py compare --a GM12878_4DNFI2A4OBS9.mcool --b K562_4DNFI18UHVRO.mcool
  python src/chromatin_query.py test --n-pairs 5
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── list ─────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List available mcool and cCRE files")

    # ── query ────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Query a mcool file against the reference database")
    p_query.add_argument("--input",          required=True, help="Path or filename of query .mcool")
    p_query.add_argument("--checkpoint",     default=None,  help="Model checkpoint path")
    p_query.add_argument("--max-tiles",      type=int, default=1000, help="Max tiles to extract")
    p_query.add_argument("--ref-per-sample", type=int, default=300,  help="Max tiles per reference sample")
    p_query.add_argument("--no-ccre",        action="store_true",    help="Skip cCRE comparison")
    p_query.add_argument("--output",         default=None,           help="Save JSON results to file")

    # ── compare ──────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Directly compare two mcool files")
    p_cmp.add_argument("--a",            required=True, help="First .mcool file")
    p_cmp.add_argument("--b",            required=True, help="Second .mcool file")
    p_cmp.add_argument("--checkpoint",   default=None)
    p_cmp.add_argument("--max-tiles",    type=int, default=500)
    p_cmp.add_argument("--no-ccre",      action="store_true")

    # ── test ─────────────────────────────────────────────────────────────────
    p_test = sub.add_parser("test", help="End-to-end accuracy test")
    p_test.add_argument("--checkpoint",   default=None)
    p_test.add_argument("--n-pairs",      type=int, default=5, help="Number of cell lines to test")
    p_test.add_argument("--no-ccre",      action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
