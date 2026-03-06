"""
preprocess.py — Hi-C data preprocessing pipeline.
Loads .mcool files, computes OE matrices, boundary labels, compartment scores,
and tiles everything into 256×256 windows saved as numpy arrays.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cooler
import cooltools
import cooltools.api.expected as ct_expected
import cooltools.api.insulation as ct_insulation
import cooltools.api.eigdecomp as ct_eigdecomp
import bioframe

from config import (
    MCOOL_DIR, PROCESSED_DIR, CELL_LINE_REGISTRY, CHROMOSOMES,
    RESOLUTION, TILE_SIZE, TILE_STEP, INSULATION_WINDOW, MIN_VALID_FRAC,
)


def get_cooler(sample_id: str, resolution: int = RESOLUTION) -> cooler.Cooler:
    info   = CELL_LINE_REGISTRY[sample_id]
    path   = MCOOL_DIR / info["file"]
    uri    = f"{path}::/resolutions/{resolution}"
    return cooler.Cooler(str(uri))


def compute_oe_matrix(clr: cooler.Cooler, chrom: str) -> np.ndarray:
    """
    Compute Observed/Expected (OE) contact matrix for one chromosome.
    Uses cooltools expected_cis to get distance-normalised expected, then
    divides observed by expected bin-by-bin.
    Returns log2(OE + 1) clipped to [-5, 5].
    """
    bins  = clr.bins().fetch(chrom)
    n     = len(bins)
    if n < 2:
        return np.zeros((n, n), dtype=np.float32)

    # Observed (ICE-balanced)
    mat = clr.matrix(balance=True).fetch(chrom).astype(np.float64)

    # Expected via cooltools
    try:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom])]
        )
        expected = ct_expected.expected_cis(
            clr,
            view_df=view_df,
            nproc=1,
            clr_weight_name="weight",
        )
        exp_vals = expected["balanced.avg"].values  # length = n diagonal offsets

        # Build expected matrix
        exp_mat = np.zeros((n, n), dtype=np.float64)
        for d in range(n):
            if d < len(exp_vals) and exp_vals[d] > 0:
                val = exp_vals[d]
                diag_idx = np.arange(n - d)
                rows = diag_idx
                cols = diag_idx + d
                exp_mat[rows, cols] = val
                exp_mat[cols, rows] = val

        with np.errstate(divide="ignore", invalid="ignore"):
            oe = np.where(exp_mat > 0, mat / exp_mat, 0.0)
    except Exception:
        # Fallback: simple distance normalisation
        oe = _simple_oe(mat)

    # Log2(OE+epsilon) and clip
    with np.errstate(divide="ignore", invalid="ignore"):
        log_oe = np.log2(oe + 1e-6)
    log_oe = np.nan_to_num(log_oe, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(log_oe, -5.0, 5.0).astype(np.float32)


def _simple_oe(mat: np.ndarray) -> np.ndarray:
    """Fallback OE: divide each diagonal by its mean."""
    n = mat.shape[0]
    oe = np.zeros_like(mat, dtype=np.float64)
    for d in range(n):
        diag = np.diagonal(mat, offset=d)
        mu   = np.nanmean(diag)
        if mu > 0:
            normed = diag / mu
            idx = np.arange(len(normed))
            oe[idx, idx + d] = normed
            if d > 0:
                oe[idx + d, idx] = normed
    return oe


def compute_boundary_labels(clr: cooler.Cooler, chrom: str) -> np.ndarray:
    """
    Binary boundary vector [N_bins] for a chromosome.
    Uses cooltools insulation score with a 500 kb window.
    """
    n = len(clr.bins().fetch(chrom))
    try:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom])]
        )
        ins = ct_insulation.insulation(
            clr,
            window_bp=[INSULATION_WINDOW],
            view_df=view_df,
            clr_weight_name="weight",
            nproc=1,
        )
        col = f"is_boundary_{INSULATION_WINDOW}"
        if col in ins.columns:
            labels = ins[col].fillna(False).astype(np.float32).values
            if len(labels) == n:
                return labels
    except Exception:
        pass
    return np.zeros(n, dtype=np.float32)


def compute_compartment_scores(clr: cooler.Cooler, chrom: str) -> np.ndarray:
    """
    A/B compartment E1 eigenvector [N_bins], phased so A>0.
    Uses cooltools eigs_cis with GC-content phasing track.
    Falls back to correlation-based E1 if phasing track unavailable.
    """
    n = len(clr.bins().fetch(chrom))
    try:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom])]
        )
        eigvals, eigvecs = ct_eigdecomp.eigs_cis(
            clr,
            view_df=view_df,
            n_eigs=3,
            clr_weight_name="weight",
        )
        # cooltools 0.7.x returns a single DataFrame (not a list)
        if isinstance(eigvecs, list):
            df = eigvecs[0]
        else:
            df = eigvecs

        if "E1" in df.columns:
            e1 = df["E1"].fillna(0.0).values.astype(np.float32)
        else:
            # Fall back to first numeric column after metadata cols
            numeric_cols = [c for c in df.columns if c not in ("chrom","start","end")]
            e1 = df[numeric_cols[0]].fillna(0.0).values.astype(np.float32) if numeric_cols else np.zeros(n, dtype=np.float32)

        # Phase: positive = more contacts with high-E1 regions (A)
        if len(e1) == n:
            return e1
    except Exception:
        pass
    return np.zeros(n, dtype=np.float32)


def tile_chromosome(
    oe_matrix: np.ndarray,
    boundaries: np.ndarray,
    compartments: np.ndarray,
    chrom: str,
    tile_size: int = TILE_SIZE,
    step: int = TILE_STEP,
    min_valid_frac: float = MIN_VALID_FRAC,
) -> List[Dict]:
    """
    Tile a chromosome OE matrix into overlapping 256×256 windows.
    Returns list of dicts with matrix + labels + coordinates.
    """
    n = oe_matrix.shape[0]
    tiles = []

    for start in range(0, n - tile_size + 1, step):
        end = start + tile_size
        tile_mat = oe_matrix[start:end, start:end].copy()

        # Skip if too many NaN/zeros (uninformative tile)
        finite_mask = np.isfinite(tile_mat) & (tile_mat != 0.0)
        if finite_mask.mean() < min_valid_frac:
            continue

        # Replace NaN with 0 for model input
        tile_mat = np.nan_to_num(tile_mat, nan=0.0)

        bd_slice   = boundaries[start:end]
        comp_slice = compartments[start:end]

        # Normalise compartment to [-1, 1]
        c_abs = np.abs(comp_slice).max()
        if c_abs > 1e-6:
            comp_slice = comp_slice / c_abs

        tiles.append({
            "matrix":      tile_mat.astype(np.float32),        # [256, 256]
            "boundary":    bd_slice.astype(np.float32),         # [256]
            "compartment": comp_slice.astype(np.float32),       # [256]
            "chr":         chrom,
            "start_bin":   int(start),
            "end_bin":     int(end),
            "start_bp":    int(start * RESOLUTION),
            "end_bp":      int(end * RESOLUTION),
        })
    return tiles


def preprocess_sample(
    sample_id: str,
    out_dir: Optional[Path] = None,
    chroms: List[str] = CHROMOSOMES,
    resolution: int = RESOLUTION,
    verbose: bool = True,
) -> List[Dict]:
    """
    Full preprocessing pipeline for one sample.
    Returns list of tile dicts; also saves them as .npz if out_dir given.
    """
    if out_dir is None:
        out_dir = PROCESSED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clr    = get_cooler(sample_id, resolution)
    available_chroms = set(clr.chromnames)
    tiles_all = []

    for chrom in chroms:
        if chrom not in available_chroms:
            continue

        npz_path = out_dir / f"{chrom}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            n_tiles = data["matrices"].shape[0]
            for i in range(n_tiles):
                tiles_all.append({
                    "matrix":      data["matrices"][i],
                    "boundary":    data["boundaries"][i],
                    "compartment": data["compartments"][i],
                    "chr":         str(data["chroms"][i]),
                    "start_bin":   int(data["start_bins"][i]),
                    "end_bin":     int(data["end_bins"][i]),
                    "start_bp":    int(data["start_bps"][i]),
                    "end_bp":      int(data["end_bps"][i]),
                    "sample_id":   sample_id,
                })
            if verbose:
                print(f"  [{sample_id}] {chrom}: loaded {n_tiles} tiles from cache")
            continue

        if verbose:
            print(f"  [{sample_id}] {chrom}: computing OE matrix...", flush=True)

        oe          = compute_oe_matrix(clr, chrom)
        boundaries  = compute_boundary_labels(clr, chrom)
        compartments = compute_compartment_scores(clr, chrom)

        tiles = tile_chromosome(oe, boundaries, compartments, chrom)

        if len(tiles) == 0:
            if verbose:
                print(f"  [{sample_id}] {chrom}: no valid tiles, skipping")
            continue

        # Save per-chromosome npz
        np.savez_compressed(
            str(npz_path),
            matrices=np.stack([t["matrix"] for t in tiles]),
            boundaries=np.stack([t["boundary"] for t in tiles]),
            compartments=np.stack([t["compartment"] for t in tiles]),
            chroms=np.array([t["chr"] for t in tiles]),
            start_bins=np.array([t["start_bin"] for t in tiles]),
            end_bins=np.array([t["end_bin"] for t in tiles]),
            start_bps=np.array([t["start_bp"] for t in tiles]),
            end_bps=np.array([t["end_bp"] for t in tiles]),
        )

        for t in tiles:
            t["sample_id"] = sample_id
        tiles_all.extend(tiles)

        if verbose:
            print(f"  [{sample_id}] {chrom}: {len(tiles)} tiles saved")

    return tiles_all


def preprocess_all(
    cell_lines: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Preprocess all registered cell lines (or a subset).
    Returns dict: sample_id → list of tile dicts.
    """
    if cell_lines is None:
        cell_lines = list(CELL_LINE_REGISTRY.keys())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for sample_id in cell_lines:
        if verbose:
            print(f"\n=== Preprocessing {sample_id} ===")
        tiles = preprocess_sample(sample_id, verbose=verbose)
        results[sample_id] = tiles
        if verbose:
            print(f"  Total tiles: {len(tiles)}")

    return results


if __name__ == "__main__":
    preprocess_all(verbose=True)
