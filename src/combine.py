"""
combine.py — Batch-combine preprocessed .npz tiles into a .dat memmap file.

Design: RAM-safe streaming, maximum DataLoader throughput.
  - Produces TWO files:
      tiles.dat   — flat float32 binary  [N, 256, 256]  (numpy memmap)
      tiles_meta.npz — compact metadata arrays + train/val index arrays
  - Why .dat instead of HDF5:
      HDF5 gzip decompresses every read → CPU-bound bottleneck per worker.
      .dat is a raw mmap: the OS page-cache serves tiles directly with
      zero decompression, enabling full NVMe bandwidth to all workers.
  - RAM-safe: processes BATCH_SIZE samples at a time (default 10).
      Peak RAM = batch_size × avg_tiles_per_sample × 256 KB ≈ 500 MB.
  - Stratified train/val split: each sample contributes proportionally
      so the classifier always sees every cell type in both train and val.

Usage:
    python src/combine.py --processed_dir data/processed --out data/tiles.dat
    python src/combine.py --processed_dir trash/test_run/processed \\
                          --out trash/test_run/tiles.dat \\
                          --batch_size 10
"""

import argparse
import sys
import time
import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil

# Add src to path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, CELL_LINE_REGISTRY, ASSAY_TYPES, CHROMOSOMES, SEED

TILE_SIZE = 256


def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _sample_list(processed_dir: Path, cell_lines: Optional[List[str]]) -> List[str]:
    """Return sorted list of sample names that exist in processed_dir."""
    if cell_lines:
        return [s for s in cell_lines if (processed_dir / s).is_dir()]
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    return sorted(d.name for d in processed_dir.iterdir() if d.is_dir())


def _count_tiles_for_sample(sample_dir: Path, chroms: List[str]) -> int:
    """Fast count of tiles without loading matrices."""
    total = 0
    for chrom in chroms:
        npz = sample_dir / f"{chrom}.npz"
        if not npz.exists():
            continue
        d = np.load(str(npz), mmap_mode="r")
        total += d["matrices"].shape[0]
        d.close()
    return total


def _count_total_tiles(processed_dir: Path, samples: List[str], chroms: List[str]) -> int:
    total = 0
    for s in samples:
        total += _count_tiles_for_sample(processed_dir / s, chroms)
    return total


def combine(
    processed_dir: Path,
    out_path:      Path,
    cell_lines:    Optional[List[str]] = None,
    chroms:        Optional[List[str]] = None,
    batch_size:    int   = 10,
    val_frac:      float = 0.1,
    seed:          int   = SEED,
    verbose:       bool  = True,
) -> Path:
    """
    Stream all preprocessed tiles into TWO files:
      <out_path>          — flat float32 binary [N, 256, 256] (numpy memmap)
      <out_path>_meta.npz — metadata arrays + stratified train/val indices

    Peak RAM = batch_size × avg_tiles_per_sample × 256 KB
    For batch_size=10, ~200 tiles/sample → ~500 MB peak.

    Why .dat not HDF5:
      numpy memmap is a raw OS page-cache mmap — no decompression,
      no Python overhead per read.  DataLoader workers read tiles via
      direct OS page faults, saturating NVMe bandwidth.
    """
    processed_dir = Path(processed_dir)
    out_path      = Path(str(out_path).replace(".h5", ".dat"))  # ensure .dat ext
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(out_path) + "_meta.npz")

    if chroms is None:
        chroms = CHROMOSOMES

    samples = _sample_list(processed_dir, cell_lines)
    if not samples:
        raise RuntimeError(f"No sample directories found in {processed_dir}")

    cell_idx_map = {sid: i for i, sid in enumerate(sorted(samples))}

    t_start = time.time()
    if verbose:
        print(f"[combine] Scanning tile counts for {len(samples)} samples...")
    n_total = _count_total_tiles(processed_dir, samples, chroms)
    if n_total == 0:
        raise RuntimeError("No tiles found.")

    if verbose:
        dat_gb = n_total * TILE_SIZE * TILE_SIZE * 4 / 1e9
        print(f"[combine] {n_total} tiles × {TILE_SIZE}² float32 = {dat_gb:.2f} GB")
        print(f"[combine] Samples: {len(samples)}  Batch: {batch_size}  "
              f"RAM free: {_available_ram_gb():.1f} GB")
        print(f"[combine] Output: {out_path}")

    # ── Pre-allocate the full .dat memmap ─────────────────────────────────────
    # This creates the file immediately at full size (sparse on most filesystems)
    mmap = np.memmap(str(out_path), dtype="float32", mode="w+",
                     shape=(n_total, TILE_SIZE, TILE_SIZE))

    # Metadata arrays (small — always fit in RAM)
    meta_sids  = np.empty(n_total, dtype=object)
    meta_chrs  = np.empty(n_total, dtype=object)
    meta_sbps  = np.zeros(n_total, dtype=np.int64)
    meta_ebps  = np.zeros(n_total, dtype=np.int64)
    meta_aids  = np.zeros(n_total, dtype=np.int32)
    meta_cidxs = np.zeros(n_total, dtype=np.int32)

    cursor    = 0
    n_batches = (len(samples) + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        batch_samples = samples[b_idx * batch_size : (b_idx + 1) * batch_size]
        t_batch = time.time()

        b_matrices, b_sids, b_chrs = [], [], []
        b_sbps, b_ebps, b_aids, b_cidxs = [], [], [], []

        for sample_id in batch_samples:
            reg_entry  = CELL_LINE_REGISTRY.get(sample_id, {})
            assay_type = reg_entry.get("assay", "bulk_hic")
            assay_id   = int(ASSAY_TYPES.get(assay_type, 0))
            cell_idx   = cell_idx_map[sample_id]
            sample_dir = processed_dir / sample_id

            for chrom in chroms:
                npz_path = sample_dir / f"{chrom}.npz"
                if not npz_path.exists():
                    continue
                d   = np.load(str(npz_path), allow_pickle=True)
                n   = d["matrices"].shape[0]
                mat = d["matrices"][:].astype(np.float32)
                chs = [str(d["chroms"][i]) for i in range(n)]
                sbp = d["start_bps"][:].astype(np.int64)
                ebp = d["end_bps"][:].astype(np.int64)
                d.close()

                b_matrices.append(mat)
                b_sids.extend([sample_id] * n)
                b_chrs.extend(chs)
                b_sbps.append(sbp)
                b_ebps.append(ebp)
                b_aids.extend([assay_id] * n)
                b_cidxs.extend([cell_idx] * n)

        if not b_matrices:
            continue

        batch_mat  = np.concatenate(b_matrices, axis=0)
        batch_sbps = np.concatenate(b_sbps, axis=0)
        batch_ebps = np.concatenate(b_ebps, axis=0)
        n_new  = len(batch_mat)
        end    = cursor + n_new

        # Write directly into pre-allocated memmap — no resize needed
        mmap[cursor:end] = batch_mat

        meta_sids[cursor:end]  = b_sids
        meta_chrs[cursor:end]  = b_chrs
        meta_sbps[cursor:end]  = batch_sbps
        meta_ebps[cursor:end]  = batch_ebps
        meta_aids[cursor:end]  = b_aids
        meta_cidxs[cursor:end] = b_cidxs
        cursor = end

        # Flush this batch to disk and free RAM
        mmap.flush()
        del batch_mat, b_matrices, b_sbps, b_ebps

        if verbose:
            ram = psutil.Process().memory_info().rss / 1e9
            print(f"  [batch {b_idx+1}/{n_batches}] "
                  f"{batch_samples} → {n_new} tiles  "
                  f"(total={cursor}/{n_total})  RAM={ram:.1f}GB  "
                  f"[{time.time()-t_batch:.1f}s]")

    del mmap  # close memmap

    # ── Stratified train/val split ────────────────────────────────────────────
    train_idx = val_idx = np.array([], dtype=np.int64)
    if val_frac > 0.0:
        rng          = np.random.default_rng(seed)
        unique_cells = np.unique(meta_cidxs[:cursor])
        train_parts, val_parts = [], []
        for cid in unique_cells:
            cls_idx = np.where(meta_cidxs[:cursor] == cid)[0].astype(np.int64)
            cls_idx = rng.permutation(cls_idx)
            n_v     = max(1, int(len(cls_idx) * val_frac))
            val_parts.append(cls_idx[:n_v])
            train_parts.append(cls_idx[n_v:])
        train_idx = rng.permutation(np.concatenate(train_parts)).astype(np.int64)
        val_idx   = rng.permutation(np.concatenate(val_parts)).astype(np.int64)

        if verbose:
            from collections import Counter
            tc = Counter(int(meta_cidxs[i]) for i in train_idx)
            vc = Counter(int(meta_cidxs[i]) for i in val_idx)
            print(f"[combine] Stratified split: train={len(train_idx)} {dict(tc)} "
                  f"| val={len(val_idx)} {dict(vc)}")

    # ── Save metadata .npz ────────────────────────────────────────────────────
    np.savez(
        str(meta_path),
        sample_ids = meta_sids[:cursor].astype(str),
        chroms     = meta_chrs[:cursor].astype(str),
        start_bps  = meta_sbps[:cursor],
        end_bps    = meta_ebps[:cursor],
        assay_ids  = meta_aids[:cursor],
        cell_idxs  = meta_cidxs[:cursor],
        train_idx  = train_idx,
        val_idx    = val_idx,
        n_tiles    = np.array(cursor),
        tile_size  = np.array(TILE_SIZE),
        samples    = np.array(sorted(samples)),
    )

    elapsed = time.time() - t_start
    dat_gb  = out_path.stat().st_size / 1e9
    meta_mb = meta_path.stat().st_size / 1e6
    if verbose:
        print(f"\n[combine] DONE — {cursor} tiles | "
              f"{dat_gb:.2f} GB dat + {meta_mb:.1f} MB meta | {elapsed:.1f}s")
        print(f"[combine] dat:  {out_path}")
        print(f"[combine] meta: {meta_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Combine preprocessed .npz tiles into a fast .dat memmap file"
    )
    p.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out",           type=Path, default=None,
                   help="Output .dat file (default: processed_dir/../tiles.dat)")
    p.add_argument("--cell_lines",    nargs="+", default=None)
    p.add_argument("--batch_size",    type=int,  default=10,
                   help="Samples per RAM batch (default=10 ≈ 500 MB peak)")
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--no_split",      action="store_true")
    p.add_argument("--chroms",        nargs="+", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processed_dir = args.processed_dir
    out_path = args.out or (processed_dir.parent / "tiles.dat")
    combine(
        processed_dir = processed_dir,
        out_path      = out_path,
        cell_lines    = args.cell_lines,
        chroms        = args.chroms,
        batch_size    = args.batch_size,
        val_frac      = 0.0 if args.no_split else args.val_frac,
        verbose       = True,
    )
