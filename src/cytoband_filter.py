"""
cytoband_filter.py — Cytoband-based genomic region filtering for Hi-C preprocessing.

Downloads UCSC cytoBand.txt for hg38 if not present.
Identifies non-genomic regions (centromeres, telomeres, heterochromatin, gaps)
and produces a boolean quality map per chromosome.
"""

import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Set, Optional
import cooler

# UCSC hg38 cytoband download URL
CYTOBAND_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz"
CYTOBAND_URL_PLAIN = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt"

# Cytoband stain types to mask (non-genomic / poorly mappable regions)
# acen  = centromere
# gvar  = heterochromatin (variable staining)
# stalk = short arm stalks (acrocentric chromosomes)
BAD_STAINS = {"acen", "gvar", "stalk"}

# Number of telomeric bins to mask at each chromosome end
TELOMERE_BINS = 3


def download_cytoband(dest: Path) -> None:
    """Download hg38 cytoBand.txt from UCSC to dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cytoband] Downloading cytoBand.hg38.txt from UCSC → {dest}")
    try:
        # Try plain text first
        urllib.request.urlretrieve(CYTOBAND_URL_PLAIN, str(dest))
        print(f"[cytoband] Downloaded successfully ({dest.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[cytoband] Plain download failed ({e}), trying gzipped...")
        try:
            import gzip
            import shutil
            gz_dest = dest.with_suffix(".txt.gz")
            urllib.request.urlretrieve(CYTOBAND_URL, str(gz_dest))
            with gzip.open(str(gz_dest), "rb") as f_in:
                with open(str(dest), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_dest.unlink(missing_ok=True)
            print(f"[cytoband] Downloaded and decompressed successfully")
        except Exception as e2:
            raise RuntimeError(
                f"[cytoband] Failed to download cytoBand file: {e2}\n"
                f"Please manually download from {CYTOBAND_URL_PLAIN} and save to {dest}"
            )


def load_cytoband(path: Path) -> pd.DataFrame:
    """
    Load cytoBand.txt file into a DataFrame.
    Handles both plain text and gzip-compressed files transparently.
    Columns: chrom, start, end, name, gieStain
    """
    if not path.exists():
        download_cytoband(path)

    # Detect gzip by magic bytes
    import gzip as _gzip
    with open(str(path), "rb") as f:
        magic = f.read(2)
    is_gz = magic == b"\x1f\x8b"

    opener = _gzip.open if is_gz else open
    with opener(str(path), "rt") as f:
        df = pd.read_csv(
            f,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "name", "gieStain"],
            dtype={"start": int, "end": int},
        )
    return df


def get_bad_bins(
    chrom: str,
    chrom_size: int,
    resolution: int,
    cytoband_df: pd.DataFrame,
    telomere_bins: int = TELOMERE_BINS,
) -> Set[int]:
    """
    Return set of bin indices that should be masked for a chromosome.

    Masks:
    - Centromeres (acen)
    - Heterochromatin (gvar)
    - Stalks (stalk)
    - Telomeric ends (first and last N bins)
    """
    n_bins = (chrom_size + resolution - 1) // resolution
    bad = set()

    # Mask cytoband bad stain regions
    chrom_bands = cytoband_df[cytoband_df["chrom"] == chrom]
    bad_bands = chrom_bands[chrom_bands["gieStain"].isin(BAD_STAINS)]

    for _, row in bad_bands.iterrows():
        start_bin = row["start"] // resolution
        end_bin   = (row["end"] + resolution - 1) // resolution
        bad.update(range(max(0, start_bin), min(end_bin + 1, n_bins)))

    # Mask telomeric ends
    bad.update(range(0, min(telomere_bins, n_bins)))
    bad.update(range(max(0, n_bins - telomere_bins), n_bins))

    return bad


def create_quality_map(
    clr: cooler.Cooler,
    chrom: str,
    resolution: int,
    cytoband_df: pd.DataFrame,
) -> np.ndarray:
    """
    Create a boolean quality map for a chromosome.

    Returns:
        np.ndarray of shape [n_bins], dtype bool
        True  = valid genomic bin (keep)
        False = problematic region (mask)
    """
    chrom_size = clr.chromsizes[chrom]
    n_bins     = len(clr.bins().fetch(chrom))

    # Start with all bins valid
    quality_map = np.ones(n_bins, dtype=bool)

    # Get bad bins from cytoband
    bad_bins = get_bad_bins(chrom, chrom_size, resolution, cytoband_df)

    for b in bad_bins:
        if 0 <= b < n_bins:
            quality_map[b] = False

    return quality_map


def mask_oe_matrix(
    oe_matrix: np.ndarray,
    quality_map: np.ndarray,
) -> np.ndarray:
    """
    Zero out rows and columns corresponding to bad bins in the OE matrix.
    This ensures that non-genomic regions contribute 0 to any tile that
    overlaps them, rather than NaN or spurious signal.
    """
    masked = oe_matrix.copy()
    bad_bins = np.where(~quality_map)[0]
    if len(bad_bins) > 0:
        masked[bad_bins, :] = 0.0
        masked[:, bad_bins] = 0.0
    return masked


def print_coverage_summary(chrom: str, quality_map: np.ndarray, resolution: int) -> None:
    """Print a short coverage summary for a chromosome."""
    n_total = len(quality_map)
    n_good  = int(quality_map.sum())
    pct     = 100.0 * n_good / max(n_total, 1)
    span_mb = n_total * resolution / 1e6
    good_mb = n_good  * resolution / 1e6
    print(f"  [{chrom}] coverage: {n_good}/{n_total} bins ({pct:.1f}%) "
          f"= {good_mb:.1f}/{span_mb:.1f} Mb usable")
