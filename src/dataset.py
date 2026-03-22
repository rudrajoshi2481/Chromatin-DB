"""
dataset.py — PyTorch Dataset and DataLoader for MQ-VAE Hi-C training.
Reads preprocessed .npz tile files from disk.

RAM-safe design for 500+ samples:
  - __init__ uses mmap_mode='r' so metadata arrays (chroms, start_bps, end_bps)
    are read via memory-mapping without loading tile matrices into RAM.
  - _get_npz uses a bounded LRU cache (NPZ_CACHE_SIZE files) so at most
    NPZ_CACHE_SIZE * npz_file_size bytes of npz handles stay in RAM.
  - num_workers and pin_memory are auto-tuned based on available RAM so
    worker processes don't collectively exhaust memory.
"""

import json
import os
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import h5py
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import (
    ASSAY_TYPES, BATCH_SIZE, CELL_LINE_REGISTRY, CHROMOSOMES,
    NUM_WORKERS, PROCESSED_DIR, SEED, TILE_SIZE,
)


def load_cell_line_label_map(processed_dir: Path) -> Dict[str, int]:
    """
    Load the deterministic cell-line → integer label map written by preprocess.py.
    Falls back to building one on the fly from the registry if the file is absent.
    """
    map_path = Path(processed_dir) / "cell_line_label_map.json"
    if map_path.exists():
        with open(str(map_path)) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    # Fallback: derive from whatever sample dirs exist
    dirs = sorted(d.name for d in Path(processed_dir).iterdir() if d.is_dir())
    return {sid: i for i, sid in enumerate(dirs)}

# Maximum number of open npz file handles kept in the LRU cache per worker.
# Each npz holds ~metadata + one chr worth of tiles.  At 23 chroms/sample and
# 500 samples the full set would be 11 500 files; we cap at 64 so each worker
# holds at most ~64 open mmaps regardless of dataset size.
NPZ_CACHE_SIZE = 64


def _available_ram_gb() -> float:
    """Return available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def _safe_num_workers(requested: int, n_tiles: int, batch_size: int) -> int:
    """
    Cap num_workers so that worker processes don't collectively exhaust RAM.
    Each worker pre-fetches ~2 batches; each tile is ~256KB (256×256×float32).
    We leave at least 8 GB free for the model and OS.
    """
    avail_gb   = _available_ram_gb()
    headroom   = max(0.0, avail_gb - 8.0)          # keep 8 GB free
    tile_mb    = TILE_SIZE * TILE_SIZE * 4 / 1e6    # bytes → MB
    prefetch   = 2                                   # DataLoader default prefetch factor
    mb_per_worker = batch_size * prefetch * tile_mb
    max_workers_by_ram = max(0, int(headroom * 1024 / mb_per_worker))
    safe = min(requested, max_workers_by_ram)
    if safe < requested:
        print(f"[dataset] RAM-safe num_workers capped: {requested} → {safe} "
              f"(avail={avail_gb:.1f}GB, {mb_per_worker:.0f}MB/worker)")
    return max(0, safe)


class _LRUNpzCache:
    """
    Thread-unsafe (single-process) LRU cache for open npz file handles.
    Bounded to `maxsize` entries; oldest handle is closed when evicted.
    Safe because each DataLoader worker gets its own copy (fork semantics).
    """

    def __init__(self, maxsize: int = NPZ_CACHE_SIZE):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, np.lib.npyio.NpzFile] = OrderedDict()

    def get(self, path: str) -> np.lib.npyio.NpzFile:
        if path in self._cache:
            self._cache.move_to_end(path)          # mark as recently used
            return self._cache[path]
        # Open with mmap_mode='r': matrices stay on disk, accessed on demand
        fh = np.load(path, allow_pickle=True, mmap_mode="r")
        self._cache[path] = fh
        self._cache.move_to_end(path)
        if len(self._cache) > self._maxsize:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        return fh


class HiCTileDataset(Dataset):
    """
    Loads preprocessed 256×256 Hi-C tiles from disk on demand.

    RAM usage is O(n_records * ~200 bytes) for the record list regardless of
    dataset size.  Tile matrices are only loaded when __getitem__ is called,
    and at most NPZ_CACHE_SIZE npz files are held open per worker at once.
    """

    def __init__(
        self,
        cell_lines:    Optional[List[str]] = None,
        chroms:        Optional[List[str]] = None,
        processed_dir: Path = PROCESSED_DIR,
        augment:       bool = True,
    ):
        processed_dir = Path(processed_dir)

        if cell_lines is None:
            if processed_dir.exists():
                cell_lines = sorted(
                    d.name for d in processed_dir.iterdir() if d.is_dir()
                )
            else:
                cell_lines = list(CELL_LINE_REGISTRY.keys())

        if chroms is None:
            chroms = CHROMOSOMES

        self.augment       = augment
        self.processed_dir = processed_dir
        self.records: List[Dict] = []

        # Load the deterministic label map written by preprocess.py
        self.cell_idx_map: Dict[str, int] = load_cell_line_label_map(processed_dir)

        for sample_id in cell_lines:
            reg_entry  = CELL_LINE_REGISTRY.get(sample_id, {})
            assay_type = reg_entry.get("assay", "bulk_hic")
            assay_id   = ASSAY_TYPES.get(assay_type, 0)
            cell_idx   = self.cell_idx_map.get(sample_id, 0)
            sample_dir = processed_dir / sample_id

            if not sample_dir.exists():
                print(f"[dataset] Warning: {sample_dir} not found, skipping")
                continue

            for chrom in chroms:
                npz_path = sample_dir / f"{chrom}.npz"
                if not npz_path.exists():
                    continue

                # mmap_mode='r': only the small metadata arrays are touched
                # here; the large 'matrices' array stays on disk.
                meta = np.load(str(npz_path), allow_pickle=True, mmap_mode="r")
                n = meta["matrices"].shape[0]
                chroms_arr    = meta["chroms"]
                start_bps_arr = meta["start_bps"]
                end_bps_arr   = meta["end_bps"]

                # Prefer labels baked into the npz (written by new preprocess.py)
                if "cell_line_idxs" in meta:
                    cl_idxs  = meta["cell_line_idxs"]
                    cl_names = meta["cell_line_names"]
                else:
                    cl_idxs  = None
                    cl_names = None

                for i in range(n):
                    self.records.append({
                        "npz_path":        str(npz_path),
                        "idx":             i,
                        "assay_id":        assay_id,
                        "cell_idx":        int(cl_idxs[i])  if cl_idxs  is not None else cell_idx,
                        "cell_line_name":  str(cl_names[i]) if cl_names is not None else sample_id,
                        "sample_id":       sample_id,
                        "chr":             str(chroms_arr[i]),
                        "start_bp":        int(start_bps_arr[i]),
                        "end_bp":          int(end_bps_arr[i]),
                    })
                # Close the mmap immediately — we only needed the metadata
                meta.close()

        # LRU cache created lazily per-worker (populated in __getitem__)
        self._cache: Optional[_LRUNpzCache] = None

        n_samples    = len(set(r["sample_id"]       for r in self.records))
        n_cell_lines = len(set(r["cell_line_name"]  for r in self.records))
        self.n_cell_lines = n_cell_lines
        print(f"[dataset] {len(self.records)} tiles from {n_samples} samples "
              f"({n_cell_lines} cell lines) "
              f"| npz cache size: {NPZ_CACHE_SIZE} files per worker")

    def _get_cache(self) -> _LRUNpzCache:
        """Lazily create the LRU cache (once per worker process after fork)."""
        if self._cache is None:
            self._cache = _LRUNpzCache(maxsize=NPZ_CACHE_SIZE)
        return self._cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec  = self.records[idx]
        data = self._get_cache().get(rec["npz_path"])
        i    = rec["idx"]

        # mmap slice: only this one tile's float32 data is paged in
        matrix = np.array(data["matrices"][i], dtype=np.float32)  # [256, 256]

        if self.augment and random.random() > 0.5:
            matrix = matrix.T

        contact = torch.from_numpy(matrix).unsqueeze(0)  # [1, 256, 256]

        return {
            "contact":        contact,
            "assay_id":       torch.tensor(rec["assay_id"], dtype=torch.long),
            "cell_idx":       torch.tensor(rec["cell_idx"], dtype=torch.long),
            "cell_line_name": rec["cell_line_name"],
            "sample_id":      rec["sample_id"],
            "chr":            rec["chr"],
            "start_bp":       torch.tensor(rec["start_bp"], dtype=torch.long),
            "end_bp":         torch.tensor(rec["end_bp"],   dtype=torch.long),
        }


def build_dataloaders(
    cell_lines:    Optional[List[str]] = None,
    val_frac:      float = 0.1,
    batch_size:    int   = BATCH_SIZE,
    num_workers:   int   = NUM_WORKERS,
    seed:          int   = SEED,
    processed_dir: Path  = PROCESSED_DIR,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders with a RAM-safe, reproducible split.
    num_workers is automatically capped to avoid OOM with large datasets.
    """
    dataset = HiCTileDataset(
        cell_lines=cell_lines, augment=True, processed_dir=processed_dir
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Run preprocess.py first to generate tiles."
        )

    if len(dataset) < 4:
        train_ds = dataset
        val_ds   = dataset
    else:
        n_val   = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
        val_ds.dataset.augment = False  # no augmentation during validation

    # Cap workers based on available RAM
    safe_workers = _safe_num_workers(num_workers, len(dataset), batch_size)

    # Disable pin_memory if RAM is tight (< 16 GB free)
    use_pin = torch.cuda.is_available() and _available_ram_gb() > 16.0

    n_train_eff = len(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size        = min(batch_size, n_train_eff),
        shuffle           = True,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )

    print(f"[dataset] train={len(train_ds)} | val={len(val_ds)} | "
          f"workers={safe_workers} | pin_memory={use_pin} | "
          f"RAM free={_available_ram_gb():.1f}GB")
    return train_loader, val_loader


class DatTileDataset(Dataset):
    """
    Reads tiles from a .dat memmap file produced by combine.py.

    Why this is fast:
      - tiles.dat is a raw float32 binary mmap [N, 256, 256]
      - numpy.memmap lets the OS page-cache serve tiles with zero
        decompression — each __getitem__ is a single array slice
      - The same mmap object is shared across all workers (read-only)
        via fork semantics so there is only ONE copy in page-cache
      - pin_memory=True lets CUDA DMA directly from page-cache → GPU
        with no intermediate CPU copy

    RAM: metadata arrays (~N × 50 bytes) + page-cache tiles on demand.
    """

    def __init__(
        self,
        dat_path:  Path,
        indices:   Optional[np.ndarray] = None,
        augment:   bool = True,
    ):
        dat_path  = Path(dat_path)
        meta_path = Path(str(dat_path) + "_meta.npz")

        if not dat_path.exists():
            raise FileNotFoundError(f"dat file not found: {dat_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"meta file not found: {meta_path}")

        self.augment  = augment
        self.dat_path = str(dat_path)

        # Load compact metadata into RAM (never the matrices)
        meta = np.load(str(meta_path), allow_pickle=True)
        n_total = int(meta["n_tiles"])
        tile_sz = int(meta["tile_size"])

        self._sample_ids = meta["sample_ids"]
        self._chroms     = meta["chroms"]
        self._start_bps  = meta["start_bps"]
        self._end_bps    = meta["end_bps"]
        self._assay_ids  = meta["assay_ids"].astype(np.int32)
        self._cell_idxs  = meta["cell_idxs"].astype(np.int32)

        self._indices = (
            indices.astype(np.int64)
            if indices is not None
            else np.arange(n_total, dtype=np.int64)
        )

        # Open the memmap ONCE here — shared read-only across workers via fork.
        # Shape: (N, tile_sz, tile_sz)  dtype: float32
        self._mmap = np.memmap(self.dat_path, dtype="float32", mode="r",
                               shape=(n_total, tile_sz, tile_sz))

        n_samp = len(set(self._sample_ids[self._indices].tolist()))
        print(f"[DatDataset] {len(self._indices)} tiles from {n_samp} samples "
              f"| {Path(dat_path).name}  ({n_total*tile_sz*tile_sz*4/1e9:.2f} GB)")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        row    = int(self._indices[pos])
        # Single slice from the OS page-cache — zero decompression
        matrix = self._mmap[row].copy().astype(np.float32)   # [256, 256]

        if self.augment and random.random() > 0.5:
            matrix = matrix.T

        return {
            "contact":        torch.from_numpy(matrix).unsqueeze(0),   # [1, 256, 256]
            "assay_id":       torch.tensor(int(self._assay_ids[row]),  dtype=torch.long),
            "cell_idx":       torch.tensor(int(self._cell_idxs[row]),  dtype=torch.long),
            "cell_line_name": str(self._sample_ids[row]),
            "sample_id":      str(self._sample_ids[row]),
            "chr":            str(self._chroms[row]),
            "start_bp":       torch.tensor(int(self._start_bps[row]),  dtype=torch.long),
            "end_bp":         torch.tensor(int(self._end_bps[row]),    dtype=torch.long),
        }


def build_dataloaders_dat(
    dat_path:    Path,
    batch_size:  int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed:        int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from a .dat memmap + _meta.npz pair.
    Uses the pre-built stratified split stored in _meta.npz by combine.py.
    """
    dat_path  = Path(dat_path)
    meta_path = Path(str(dat_path) + "_meta.npz")
    meta      = np.load(str(meta_path), allow_pickle=True)

    train_idx = meta["train_idx"].astype(np.int64)
    val_idx   = meta["val_idx"].astype(np.int64)
    n_total   = int(meta["n_tiles"])

    if len(train_idx) == 0:
        rng       = np.random.default_rng(seed)
        indices   = rng.permutation(n_total).astype(np.int64)
        n_val     = max(1, int(n_total * 0.1))
        train_idx = indices[:-n_val]
        val_idx   = indices[-n_val:]

    print(f"[DatDataset] Split: train={len(train_idx)} | val={len(val_idx)}")

    train_ds = DatTileDataset(dat_path, indices=train_idx, augment=True)
    val_ds   = DatTileDataset(dat_path, indices=val_idx,   augment=False)

    safe_workers = _safe_num_workers(num_workers, n_total, batch_size)
    use_pin      = torch.cuda.is_available() and _available_ram_gb() > 16.0

    train_loader = DataLoader(
        train_ds,
        batch_size        = min(batch_size, len(train_ds)),
        shuffle           = True,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )

    print(f"[DatDataset] workers={safe_workers} | pin_memory={use_pin} | "
          f"RAM free={_available_ram_gb():.1f}GB")
    return train_loader, val_loader


class HiCH5Dataset(Dataset):
    """
    Reads tiles from a combined HDF5 file produced by combine.py.

    RAM usage: O(1) — only metadata (sample_ids, chroms, start_bps, end_bps)
    is loaded into RAM as small 1-D arrays.  The large 'matrices' dataset is
    accessed via HDF5 chunked I/O — one chunk per __getitem__ call.

    The HDF5 file is opened ONCE per worker (lazily, after fork) so multiple
    workers each hold their own file handle, with no shared state.

    Supports pre-built train/val split indices stored as /train_idx and
    /val_idx by combine.py.
    """

    def __init__(
        self,
        h5_path:  Path,
        indices:  Optional[np.ndarray] = None,   # row indices to use; None=all
        augment:  bool = True,
    ):
        self.h5_path = str(h5_path)
        self.augment = augment

        # Load only the lightweight metadata into RAM (not matrices)
        with h5py.File(self.h5_path, "r") as hf:
            n_total = hf["matrices"].shape[0]
            self._sample_ids = hf["sample_ids"][:].astype(str)
            self._chroms     = hf["chroms"][:].astype(str)
            self._start_bps  = hf["start_bps"][:]
            self._end_bps    = hf["end_bps"][:]
            self._assay_ids  = hf["assay_ids"][:]
            self._cell_idxs  = hf["cell_idxs"][:]

        # Use provided index subset or full range
        if indices is not None:
            self._indices = indices.astype(np.int64)
        else:
            self._indices = np.arange(n_total, dtype=np.int64)

        # Per-worker HDF5 file handle (None until first __getitem__ after fork)
        self._h5: Optional[h5py.File] = None

        n_samples = len(set(self._sample_ids[self._indices]))
        print(f"[H5dataset] {len(self._indices)} tiles from {n_samples} samples "
              f"| source: {Path(h5_path).name}")

    def _get_h5(self) -> h5py.File:
        """Open HDF5 lazily once per worker process (safe after fork)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        row = int(self._indices[pos])
        hf  = self._get_h5()

        # Reads exactly one HDF5 chunk (CHUNK_TILES tiles) from disk
        matrix = hf["matrices"][row].astype(np.float32)  # [256, 256]

        if self.augment and random.random() > 0.5:
            matrix = matrix.T

        contact = torch.from_numpy(matrix).unsqueeze(0)  # [1, 256, 256]

        return {
            "contact":        contact,
            "assay_id":       torch.tensor(int(self._assay_ids[row]), dtype=torch.long),
            "cell_idx":       torch.tensor(int(self._cell_idxs[row]), dtype=torch.long),
            "cell_line_name": self._sample_ids[row],
            "sample_id":      self._sample_ids[row],
            "chr":            self._chroms[row],
            "start_bp":       torch.tensor(int(self._start_bps[row]), dtype=torch.long),
            "end_bp":         torch.tensor(int(self._end_bps[row]),   dtype=torch.long),
        }

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def build_dataloaders_h5(
    h5_path:     Path,
    batch_size:  int   = BATCH_SIZE,
    num_workers: int   = NUM_WORKERS,
    seed:        int   = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from a combined HDF5 file.

    If the file contains /train_idx and /val_idx (written by combine.py),
    those pre-built splits are used directly.  Otherwise, a random 90/10
    split is created at runtime.
    """
    h5_path = Path(h5_path)

    # Load split indices (tiny — just int64 arrays)
    with h5py.File(str(h5_path), "r") as hf:
        n_total = hf["matrices"].shape[0]
        has_split = "train_idx" in hf and "val_idx" in hf
        if has_split:
            train_idx = hf["train_idx"][:]
            val_idx   = hf["val_idx"][:]
            print(f"[H5dataset] Using pre-built split: "
                  f"train={len(train_idx)} | val={len(val_idx)}")
        else:
            rng     = np.random.default_rng(seed)
            indices = rng.permutation(n_total).astype(np.int64)
            n_val   = max(1, int(n_total * 0.1))
            train_idx = indices[:-n_val]
            val_idx   = indices[-n_val:]
            print(f"[H5dataset] Random split: train={len(train_idx)} | val={len(val_idx)}")

    train_ds = HiCH5Dataset(h5_path, indices=train_idx, augment=True)
    val_ds   = HiCH5Dataset(h5_path, indices=val_idx,   augment=False)

    safe_workers = _safe_num_workers(num_workers, n_total, batch_size)
    use_pin = torch.cuda.is_available() and _available_ram_gb() > 16.0

    train_loader = DataLoader(
        train_ds,
        batch_size        = min(batch_size, len(train_ds)),
        shuffle           = True,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = safe_workers,
        pin_memory        = use_pin,
        drop_last         = False,
        persistent_workers= (safe_workers > 0),
        prefetch_factor   = 2 if safe_workers > 0 else None,
    )

    print(f"[H5dataset] workers={safe_workers} | pin_memory={use_pin} | "
          f"RAM free={_available_ram_gb():.1f}GB")
    return train_loader, val_loader


def collate_fn_inference(batch):
    """
    Custom collate for inference — keeps string fields as lists.
    """
    contact    = torch.stack([b["contact"]  for b in batch])
    assay_id   = torch.stack([b["assay_id"] for b in batch])
    sample_ids = [b["sample_id"] for b in batch]
    chroms     = [b["chr"]       for b in batch]
    start_bps  = torch.stack([b["start_bp"] for b in batch])
    end_bps    = torch.stack([b["end_bp"]   for b in batch])

    return {
        "contact":    contact,
        "assay_id":   assay_id,
        "sample_ids": sample_ids,
        "chroms":     chroms,
        "start_bps":  start_bps,
        "end_bps":    end_bps,
    }


def build_inference_loader(
    cell_lines:    Optional[List[str]] = None,
    batch_size:    int  = 16,
    num_workers:   int  = NUM_WORKERS,
    processed_dir: Path = PROCESSED_DIR,
) -> DataLoader:
    """
    DataLoader for fingerprint extraction (no augmentation, no shuffle).
    """
    dataset = HiCTileDataset(cell_lines=cell_lines, augment=False, processed_dir=processed_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_inference,
    )
