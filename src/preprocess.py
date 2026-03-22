"""
preprocess.py — Streamlined Hi-C data preprocessing pipeline.

Loads .mcool files, computes OE matrices, applies cytoband-based quality
filtering to remove non-genomic regions (centromeres, telomeres, gaps),
and tiles everything into 256×256 windows saved as numpy arrays.

No insulation scores or A/B compartment computation — pure structural
fingerprinting focused on maximizing genomic coverage.

Parallelism design
------------------
* Sample-level parallelism only (no intra-sample chrom threads).
* `--workers` total threads are split evenly across `--parallel_samples`
  processes; each process gets ``workers // parallel_samples`` threads for
  cooltools expected_cis (the main bottleneck).
* RAM guard: new sample workers are not spawned if estimated RSS exceeds
  ``--max_ram_gb`` (default 1000 GB).
* Per-sample per-chromosome .npz tiles are written to ``--tmp_dir``
  (default: <project>/trash/tmp).
* As each sample finishes it is immediately merged into the growing combined
  .npz at ``--out_dir/combined.npz``, then the per-sample tmp files are
  optionally removed to free disk space.

Usage:
    python src/preprocess.py --mcool_dir /path/to/mcool
    python src/preprocess.py --mcool_dir /path/to/mcool --parallel_samples 6 --workers 150
    python src/preprocess.py --samples HCT116 HeLa-S3
    python src/preprocess.py --out_dir /path/to/output --min_valid_frac 0.2
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import gc
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import cooler
import cooltools.api.expected as ct_expected
import bioframe

from config import (
    MCOOL_DIR, PROCESSED_DIR, CELL_LINE_REGISTRY, CHROMOSOMES,
    RESOLUTION, TILE_SIZE, TILE_STEP, MIN_VALID_FRAC,
    CYTOBAND_PATH, TRASH_DIR,
)
from cytoband_filter import (
    load_cytoband, create_quality_map, mask_oe_matrix, print_coverage_summary,
)

# ── Cell-line name extraction ────────────────────────────────────────────────
# Strips ENCODE / 4DN accession suffixes AND replicate/clone identifiers so
# that all replicates of the same cell line share one canonical name.
#   GM12878_4DNFI2A4OBS9           → GM12878
#   HeLa-S3_ENCFF960YUI            → HeLa-S3
#   foreskin_fibroblast_4DNFI...   → foreskin_fibroblast
#   HCT116_auxin2h_4DNFI...        → HCT116
#   dTAG-NIPBL_hTERT-RPE-1_clone_A2_ENCFF... → dTAG-NIPBL_hTERT-RPE-1

_ACCESSION_RE = re.compile(
    r"[_.]?(?:4DNF[A-Z0-9]+|ENCFF[A-Z0-9]+|ENCSR[A-Z0-9]+|GSM\d+|SRR\d+|ERR\d+)$",
    re.IGNORECASE,
)
# Replicate/clone/condition suffixes that should be stripped when grouping
_REPLICATE_RE = re.compile(
    r"[_.]?"
    r"(?:"
    r"rep\d+|replicate\d*|r\d+(?=[_.]|$)"       # rep1, replicate2, r1
    r"|clone[_-]?[A-Z0-9]+"                      # clone_A2, cloneB1
    r"|auxin\d*(?:h|hr)?"                         # auxin2h, auxin
    r"|\d+h(?:r)?(?=[_.]|$)"                     # 2h, 48hr
    r"|DMSO|IAA|degron"                          # treatment labels
    r"|hom|het"                                  # genotype
    r")"
    r"[_.]?.*$",
    re.IGNORECASE,
)


def _extract_cell_line_name(stem: str) -> str:
    """
    Extract canonical cell-line name from a mcool filename stem.
    Steps:
      1. Strip ENCODE / 4DN accession suffix
      2. Strip trailing replicate / clone / condition tokens
    e.g.
      'GM12878_4DNFI2A4OBS9'            → 'GM12878'
      'HCT116_auxin2h_4DNFI3PNAYBK'    → 'HCT116'
      'dTAG-NIPBL_hTERT-RPE-1_-_clone_A2_ENCFF123' → 'dTAG-NIPBL_hTERT-RPE-1'
      'foreskin_fibroblast_4DNFIQJQY7PW' → 'foreskin_fibroblast'
      'CyT49_differentiated_to_definitive_endoderm_4DNFINXSKZ4F' → 'CyT49_differentiated_to_definitive_endoderm'
    """
    name = _ACCESSION_RE.sub("", stem).strip("_.-")
    name = _REPLICATE_RE.sub("", name).strip("_.-")
    return name or stem


# ── RAM estimation ─────────────────────────────────────────────────────────────
_BYTES_PER_GB = 1024 ** 3

def _current_rss_gb() -> float:
    """Return current process-tree RSS in GB (Linux /proc only, fallback 0)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        return 0.0


# ── Core math helpers ─────────────────────────────────────────────────────────

def get_cooler(mcool_path: Path, resolution: int = RESOLUTION) -> cooler.Cooler:
    uri = f"{mcool_path}::/resolutions/{resolution}"
    return cooler.Cooler(str(uri))


def compute_oe_matrix(
    clr: cooler.Cooler,
    chrom: str,
    nproc: int = 1,
) -> np.ndarray:
    """
    Compute Observed/Expected (OE) contact matrix for one chromosome.
    Uses cooltools expected_cis for distance normalization.
    Returns log2(OE + eps) clipped to [-5, 5].
    """
    bins = clr.bins().fetch(chrom)
    n    = len(bins)
    if n < 2:
        return np.zeros((n, n), dtype=np.float32)

    mat = clr.matrix(balance=True).fetch(chrom).astype(np.float64)

    try:
        view_df = bioframe.make_viewframe([(chrom, 0, clr.chromsizes[chrom])])
        expected = ct_expected.expected_cis(
            clr, view_df=view_df, nproc=nproc, clr_weight_name="weight",
        )
        exp_vals = expected["balanced.avg"].values

        exp_mat = _fill_diagonals_vectorized(n, exp_vals)

        with np.errstate(divide="ignore", invalid="ignore"):
            oe = np.where(exp_mat > 0, mat / exp_mat, 0.0)
    except Exception:
        oe = _simple_oe(mat)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_oe = np.log2(oe + 1e-6)
    log_oe = np.nan_to_num(log_oe, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(log_oe, -5.0, 5.0).astype(np.float32)


def _fill_diagonals_vectorized(n: int, exp_vals: np.ndarray) -> np.ndarray:
    """
    Fill a symmetric expected matrix using vectorized numpy index operations.
    ~50x faster than the pure-Python diagonal loop for large n.
    """
    exp_mat = np.zeros((n, n), dtype=np.float64)
    n_vals  = min(len(exp_vals), n)
    for d in range(n_vals):
        if exp_vals[d] <= 0:
            continue
        idx = np.arange(n - d)
        exp_mat[idx, idx + d] = exp_vals[d]
        if d > 0:
            exp_mat[idx + d, idx] = exp_vals[d]
    return exp_mat


def _simple_oe(mat: np.ndarray) -> np.ndarray:
    """Fallback OE: divide each diagonal by its mean."""
    n  = mat.shape[0]
    oe = np.zeros_like(mat, dtype=np.float64)
    for d in range(n):
        diag = np.diagonal(mat, offset=d)
        mu   = np.nanmean(diag)
        if mu > 0:
            normed = diag / mu
            idx    = np.arange(len(normed))
            oe[idx, idx + d] = normed
            if d > 0:
                oe[idx + d, idx] = normed
    return oe


def tile_chromosome(
    oe_matrix:    np.ndarray,
    quality_map:  np.ndarray,
    chrom:        str,
    tile_size:    int   = TILE_SIZE,
    step:         int   = TILE_STEP,
    min_valid_frac: float = MIN_VALID_FRAC,
    resolution:   int   = RESOLUTION,
) -> List[Dict]:
    """
    Tile a chromosome OE matrix into overlapping tile_size×tile_size windows.
    Uses cytoband quality_map to skip tiles dominated by non-genomic regions.

    Returns list of dicts with matrix and coordinates (no boundary/compartment).
    """
    n     = oe_matrix.shape[0]
    tiles = []

    for start in range(0, n - tile_size + 1, step):
        end      = start + tile_size
        tile_mat = oe_matrix[start:end, start:end].copy()

        tile_quality = quality_map[start:end]
        valid_frac   = tile_quality.mean()

        if valid_frac < min_valid_frac:
            continue

        tile_mat = np.nan_to_num(tile_mat, nan=0.0)

        tiles.append({
            "matrix":    tile_mat.astype(np.float32),   # [tile_size, tile_size]
            "chr":       chrom,
            "start_bin": int(start),
            "end_bin":   int(end),
            "start_bp":  int(start * resolution),
            "end_bp":    int(end   * resolution),
        })

    return tiles


# ── Per-sample worker (runs in a subprocess) ──────────────────────────────────

def preprocess_sample(
    sample_id:        str,
    mcool_path:       Path,
    out_dir:          Path,
    cytoband_df:      pd.DataFrame,
    chroms:           List[str]  = CHROMOSOMES,
    resolution:       int        = RESOLUTION,
    tile_size:        int        = TILE_SIZE,
    tile_step:        int        = TILE_STEP,
    min_valid_frac:   float      = MIN_VALID_FRAC,
    nproc_per_sample: int        = 6,
    cell_line_idx:    int        = 0,
    cell_line_name:   str        = "",
    verbose:          bool       = True,
) -> Dict:
    """
    Full preprocessing pipeline for one sample.
    Chromosomes are processed sequentially within this process.
    ``nproc_per_sample`` threads are handed to cooltools expected_cis.
    Per-chrom .npz tiles written to ``out_dir/<sample_id>/``.
    Each tile npz stores cell_line_idx (int) and cell_line_name (str) for
    the classifier and metadata retrieval.
    """
    t_start  = time.time()
    samp_dir = out_dir / sample_id
    samp_dir.mkdir(parents=True, exist_ok=True)

    try:
        clr = get_cooler(mcool_path, resolution)
    except Exception as e:
        return {"sample_id": sample_id, "status": "error",
                "error": str(e), "tiles": 0, "elapsed_s": 0,
                "out_dir": str(samp_dir)}

    available_chroms = [c for c in chroms if c in set(clr.chromnames)]
    chrom_summary    = {}

    for chrom in available_chroms:
        npz_path = samp_dir / f"{chrom}.npz"
        if npz_path.exists():
            data    = np.load(str(npz_path), allow_pickle=True)
            n_tiles = data["matrices"].shape[0]
            if verbose:
                print(f"  [{sample_id}] {chrom}: {n_tiles} tiles (cached)", flush=True)
            chrom_summary[chrom] = {"tiles": n_tiles, "cached": True}
            continue

        if verbose:
            print(f"  [{sample_id}] {chrom}: computing OE (nproc={nproc_per_sample})...",
                  flush=True)

        try:
            quality_map = create_quality_map(clr, chrom, resolution, cytoband_df)
            oe          = compute_oe_matrix(clr, chrom, nproc=nproc_per_sample)
            oe          = mask_oe_matrix(oe, quality_map)
            tiles       = tile_chromosome(
                oe, quality_map, chrom,
                tile_size=tile_size, step=tile_step,
                min_valid_frac=min_valid_frac, resolution=resolution,
            )

            if tiles:
                n = len(tiles)
                np.savez_compressed(
                    str(npz_path),
                    matrices        = np.stack([t["matrix"]   for t in tiles]),
                    chroms          = np.array([t["chr"]       for t in tiles]),
                    start_bins      = np.array([t["start_bin"] for t in tiles]),
                    end_bins        = np.array([t["end_bin"]   for t in tiles]),
                    start_bps       = np.array([t["start_bp"]  for t in tiles]),
                    end_bps         = np.array([t["end_bp"]    for t in tiles]),
                    cell_line_idxs  = np.full(n, cell_line_idx,  dtype=np.int32),
                    cell_line_names = np.full(n, cell_line_name or sample_id, dtype=object),
                    sample_ids      = np.full(n, sample_id,      dtype=object),
                )
            if verbose:
                print(f"  [{sample_id}] {chrom}: {len(tiles)} tiles saved", flush=True)
            chrom_summary[chrom] = {"tiles": len(tiles), "cached": False}

        except Exception as exc:
            print(f"  [{sample_id}] {chrom}: ERROR {exc}", flush=True)
            chrom_summary[chrom] = {"tiles": 0, "cached": False}
            import traceback; traceback.print_exc()

    elapsed     = time.time() - t_start
    total_tiles = sum(c["tiles"] for c in chrom_summary.values())
    file_mb     = mcool_path.stat().st_size / 1e6
    if verbose:
        print(f"  [{sample_id}] DONE: {total_tiles} tiles | {file_mb:.0f} MB | {elapsed:.1f}s",
              flush=True)

    return {
        "sample_id":       sample_id,
        "cell_line_idx":   cell_line_idx,
        "cell_line_name":  cell_line_name or sample_id,
        "status":          "ok",
        "tiles":           total_tiles,
        "elapsed_s":       elapsed,
        "file_mb":         file_mb,
        "chrom_summary":   chrom_summary,
        "out_dir":         str(samp_dir),
    }


def _worker(args: Dict) -> Dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    return preprocess_sample(**args)


# ── Incremental combine removed — tiles written directly to out_dir/<sample_id>/ ──


# ── Main orchestrator ─────────────────────────────────────────────────────────

def preprocess_all(
    cell_lines:        Optional[List[str]] = None,
    mcool_dir:         Path                = MCOOL_DIR,
    out_dir:           Path                = PROCESSED_DIR,
    cytoband_path:     Path                = CYTOBAND_PATH,
    parallel_samples:  int                 = 6,
    total_workers:     int                 = 200,
    chroms:            List[str]           = CHROMOSOMES,
    resolution:        int                 = RESOLUTION,
    tile_size:         int                 = TILE_SIZE,
    tile_step:         int                 = TILE_STEP,
    min_valid_frac:    float               = MIN_VALID_FRAC,
    max_ram_gb:        float               = 1024.0,
    verbose:           bool                = True,
    timing_log:        Optional[Path]      = None,
) -> Dict[str, Dict]:
    """
    Preprocess ALL replicates in mcool_dir in parallel.
    Each replicate gets its own ``out_dir/<sample_id>/`` subdirectory.
    ``cell_line_name`` is the canonical name shared across replicates
    (ENCODE accession + replicate tokens stripped).
    The integer classifier label is keyed on ``cell_line_name``.

    Args:
        cell_lines:       Filter to these sample IDs (raw stems). None = all.
        mcool_dir:        Directory containing .mcool files.
        out_dir:          Output root; per-sample tiles go to out_dir/<sample_id>/.
        cytoband_path:    Path to cytoBand.hg38.txt (auto-downloaded if missing).
        parallel_samples: Number of sample processes running simultaneously.
        total_workers:    Total threads split across parallel_samples processes.
        chroms:           Chromosomes to process.
        resolution:       Hi-C resolution in bp.
        tile_size:        Tile size in bins.
        tile_step:        Step between tiles.
        min_valid_frac:   Minimum valid fraction per tile.
        max_ram_gb:       Soft RSS ceiling — pause spawning above this.
        verbose:          Per-chromosome progress lines.
        timing_log:       Optional TSV with per-sample timing.

    Returns:
        Dict mapping sample_id → result summary.
    """
    import json as _json

    # ── Discover ALL .mcool files (all replicates kept) ──────────────────────
    mcool_files = sorted(mcool_dir.glob("*.mcool"))
    if not mcool_files:
        print(f"[preprocess] No .mcool files found in {mcool_dir}")
        return {}

    # sample_id = full stem (unique per file); cell_line_name = canonical group
    # Use stem as sample_id so every replicate gets its own output directory
    all_samples: List[Tuple[str, Path, str]] = []  # (sample_id, path, cell_line_name)
    for p in mcool_files:
        sid  = p.stem                          # e.g. GM12878_4DNFI2A4OBS9
        cln  = _extract_cell_line_name(p.stem) # e.g. GM12878
        all_samples.append((sid, p, cln))

    # Optional filter by exact sample_id or cell_line_name
    if cell_lines is not None:
        cl_set = set(cell_lines)
        filtered = [(s, p, c) for s, p, c in all_samples
                    if s in cl_set or c in cl_set]
        if not filtered:
            print(f"[preprocess] WARNING: none of {cell_lines} matched, skipping")
            return {}
        all_samples = filtered

    if not all_samples:
        print("[preprocess] No valid samples found.")
        return {}

    # ── Build cell-line label map keyed on canonical cell_line_name ──────────
    # Sorted for reproducibility — all replicates of the same line get same int
    unique_cell_lines = sorted(set(cln for _, _, cln in all_samples))
    cell_line_map: Dict[str, int] = {cln: i for i, cln in enumerate(unique_cell_lines)}

    out_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = out_dir / "cell_line_label_map.json"
    with open(str(label_map_path), "w") as _f:
        _json.dump(cell_line_map, _f, indent=2)

    print(f"[preprocess] {len(all_samples)} total samples  "
          f"({len(unique_cell_lines)} unique cell lines)  →  {mcool_dir}", flush=True)
    print(f"[preprocess] Cell-line label map → {label_map_path}", flush=True)

    # Print cell-line grouping summary
    from collections import Counter
    cln_counts = Counter(cln for _, _, cln in all_samples)
    multi = {k: v for k, v in cln_counts.items() if v > 1}
    if multi:
        top = sorted(multi.items(), key=lambda x: -x[1])[:10]
        print(f"[preprocess] Cell lines with replicates (top 10): "
              + "  ".join(f"{k}:{v}" for k, v in top), flush=True)

    # ── Cytoband ─────────────────────────────────────────────────────────────
    cytoband_df = load_cytoband(cytoband_path)
    print(f"[preprocess] Loaded cytoband: {len(cytoband_df)} bands", flush=True)

    # ── Worker budget ─────────────────────────────────────────────────────────────
    n_parallel       = min(parallel_samples, len(all_samples))
    nproc_per_sample = max(1, total_workers // n_parallel)
    print(f"[preprocess] parallel_samples={n_parallel}  "
          f"nproc_per_sample={nproc_per_sample}  "
          f"(total threads≈{n_parallel * nproc_per_sample})", flush=True)
    print(f"[preprocess] out  → {out_dir}", flush=True)
    print(f"[preprocess] RAM ceiling: {max_ram_gb:.0f} GB", flush=True)

    t_total  = time.time()
    results: Dict[str, Dict] = {}
    n_total  = len(all_samples)
    n_done   = 0
    n_errors = 0

    # Build job arg list
    job_args = [
        {
            "sample_id":        sid,
            "mcool_path":       mcool_path,
            "out_dir":          out_dir,
            "cytoband_df":      cytoband_df,
            "chroms":           chroms,
            "resolution":       resolution,
            "tile_size":        tile_size,
            "tile_step":        tile_step,
            "min_valid_frac":   min_valid_frac,
            "nproc_per_sample": nproc_per_sample,
            "cell_line_idx":    cell_line_map[cln],
            "cell_line_name":   cln,
            "verbose":          verbose,
        }
        for sid, mcool_path, cln in all_samples
    ]

    # ── Progress bar ─────────────────────────────────────────────────────────────
    pbar = (
        _tqdm(total=n_total, desc="Preprocessing samples",
              unit="sample", dynamic_ncols=True, position=0)
        if _HAS_TQDM
        else None
    )

    def _pbar_update(r: Dict):
        nonlocal n_done, n_errors
        status = r.get("status", "ok")
        if status == "ok":
            n_done += 1
        else:
            n_errors += 1
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                done=n_done, errors=n_errors,
                remaining=n_total - n_done - n_errors,
                tiles=sum(x.get("tiles", 0) for x in results.values()),
                refresh=True,
            )
        else:
            pct = (n_done + n_errors) / n_total * 100
            print(f"[progress] {n_done + n_errors}/{n_total} ({pct:.1f}%)  "
                  f"done={n_done}  errors={n_errors}  "
                  f"remaining={n_total - n_done - n_errors}", flush=True)

    # ── Process pool ─────────────────────────────────────────────────────────────
    pending = list(job_args)
    active_futures: Dict = {}

    with ProcessPoolExecutor(max_workers=n_parallel,
                             mp_context=mp.get_context("spawn")) as executor:
        # Seed initial batch
        while pending and len(active_futures) < n_parallel:
            args = pending.pop(0)
            fut  = executor.submit(_worker, args)
            active_futures[fut] = args["sample_id"]

        while active_futures:
            done_fut = next(as_completed(active_futures))
            sid      = active_futures.pop(done_fut)

            try:
                r = done_fut.result()
                results[r["sample_id"]] = r
                _pbar_update(r)
                if not verbose:
                    print(f"[preprocess] ✓ {r['sample_id']}  "
                          f"cell_line={r['cell_line_name']}  "
                          f"{r['tiles']} tiles  {r['elapsed_s']:.1f}s", flush=True)

            except Exception as e:
                err_r = {"sample_id": sid, "status": "error", "error": str(e), "tiles": 0}
                results[sid] = err_r
                _pbar_update(err_r)
                print(f"[preprocess] ✗ {sid}: {e}", flush=True)

            # RAM guard
            if pending:
                rss = _current_rss_gb()
                if rss < max_ram_gb:
                    args = pending.pop(0)
                    fut  = executor.submit(_worker, args)
                    active_futures[fut] = args["sample_id"]
                else:
                    print(f"[preprocess] RAM guard: {rss:.1f} GB ≥ {max_ram_gb:.0f} GB — "
                          f"waiting before spawning next sample", flush=True)

    if pbar is not None:
        pbar.close()

    total_elapsed = time.time() - t_total
    total_tiles   = sum(r.get("tiles", 0) for r in results.values())
    print(f"\n[preprocess] COMPLETE: {n_done}/{n_total} samples  "
          f"{total_tiles} total tiles  {total_elapsed:.1f}s", flush=True)
    print(f"[preprocess] Output  → {out_dir}", flush=True)
    print(f"[preprocess] Labels  → {label_map_path}", flush=True)

    # ── Timing log ────────────────────────────────────────────────────────────
    if timing_log is not None:
        timing_log.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "sample_id": r.get("sample_id", "?"),
                "status":    r.get("status", "?"),
                "file_mb":   f"{r.get('file_mb', 0):.1f}",
                "tiles":     r.get("tiles", 0),
                "elapsed_s": f"{r.get('elapsed_s', 0):.1f}",
            }
            for r in results.values()
        ]
        with open(str(timing_log), "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sample_id", "status", "file_mb", "tiles", "elapsed_s"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[preprocess] Timing log → {timing_log}", flush=True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hi-C preprocessing: OE matrix + cytoband filtering + combine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcool_dir",
                        type=Path, default=MCOOL_DIR,
                        help="Folder containing .mcool files (all are auto-discovered)")
    parser.add_argument("--out_dir",
                        type=Path, default=PROCESSED_DIR,
                        help="Output root; per-sample tiles go to out_dir/<sample_id>/")
    parser.add_argument("--cytoband",
                        type=Path, default=CYTOBAND_PATH,
                        help="cytoBand.hg38.txt (auto-downloaded if missing)")
    parser.add_argument("--samples",
                        nargs="+", default=None,
                        help="Specific sample names to process (default: all in mcool_dir)")
    parser.add_argument("--parallel_samples",
                        type=int, default=6,
                        help="Number of sample processes running simultaneously")
    parser.add_argument("--workers",
                        type=int, default=200,
                        help="Total threads distributed across sample processes "
                             "(nproc_per_sample = workers // parallel_samples)")
    parser.add_argument("--resolution",
                        type=int, default=RESOLUTION)
    parser.add_argument("--tile_size",
                        type=int, default=TILE_SIZE)
    parser.add_argument("--min_valid_frac",
                        type=float, default=MIN_VALID_FRAC)
    parser.add_argument("--max_ram_gb",
                        type=float, default=1024.0,
                        help="Soft RAM ceiling in GB; pause spawning new jobs above this")
    parser.add_argument("--timing_log",
                        type=Path, default=None,
                        help="Write timing summary to this TSV file")
    parser.add_argument("--quiet",
                        action="store_true")
    args = parser.parse_args()

    tile_step = int(args.tile_size * 0.5)

    results = preprocess_all(
        cell_lines       = args.samples,
        mcool_dir        = args.mcool_dir,
        out_dir          = args.out_dir,
        cytoband_path    = args.cytoband,
        parallel_samples = args.parallel_samples,
        total_workers    = args.workers,
        resolution       = args.resolution,
        tile_size        = args.tile_size,
        tile_step        = tile_step,
        min_valid_frac   = args.min_valid_frac,
        max_ram_gb       = args.max_ram_gb,
        verbose          = not args.quiet,
        timing_log       = args.timing_log,
    )

    print("\n{:<50} {:>10} {:>10} {:>10}".format("Sample", "FileMB", "Tiles", "Time(s)"))
    print("-" * 85)
    for r in results.values():
        print("{:<50} {:>10} {:>10} {:>10}".format(
            r.get("sample_id", "?")[:50],
            f"{r.get('file_mb', 0):.0f}",
            r.get("tiles", 0),
            f"{r.get('elapsed_s', 0):.1f}",
        ))


if __name__ == "__main__":
    main()
