"""
dataset.py — PyTorch Dataset and DataLoader for MQ-VAE Hi-C training.
Reads preprocessed .npz tile files from disk.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import random

from config import (
    PROCESSED_DIR, CELL_LINE_REGISTRY, ASSAY_TYPES,
    TILE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
    CHROMOSOMES,
)


class HiCTileDataset(Dataset):
    """
    Loads all preprocessed 256×256 Hi-C tiles from disk.
    Each item: {contact, boundary, compartment, assay_id, sample_id, chr, start_bp}
    """

    def __init__(
        self,
        cell_lines: Optional[List[str]] = None,
        chroms: Optional[List[str]] = None,
        processed_dir: Path = PROCESSED_DIR,
        augment: bool = True,
    ):
        if cell_lines is None:
            cell_lines = list(CELL_LINE_REGISTRY.keys())
        if chroms is None:
            chroms = CHROMOSOMES

        self.augment = augment
        self.records: List[Dict] = []

        for sample_id in cell_lines:
            assay_type = CELL_LINE_REGISTRY[sample_id]["assay"]
            assay_id   = ASSAY_TYPES.get(assay_type, 0)
            sample_dir = processed_dir / sample_id

            if not sample_dir.exists():
                print(f"[dataset] Warning: {sample_dir} not found, skipping {sample_id}")
                continue

            for chrom in chroms:
                npz_path = sample_dir / f"{chrom}.npz"
                if not npz_path.exists():
                    continue

                data = np.load(str(npz_path), allow_pickle=True)
                n = data["matrices"].shape[0]
                for i in range(n):
                    self.records.append({
                        "npz_path":    str(npz_path),
                        "idx":         i,
                        "assay_id":    assay_id,
                        "sample_id":   sample_id,
                        "chr":         str(data["chroms"][i]),
                        "start_bp":    int(data["start_bps"][i]),
                        "end_bp":      int(data["end_bps"][i]),
                    })

        # Cache for opened npz files (avoid repeated disk seeks)
        self._npz_cache: Dict[str, np.lib.npyio.NpzFile] = {}

        print(f"[dataset] Loaded {len(self.records)} tiles from "
              f"{len(cell_lines)} cell lines")

    def _get_npz(self, path: str) -> np.lib.npyio.NpzFile:
        if path not in self._npz_cache:
            self._npz_cache[path] = np.load(path, allow_pickle=True)
        return self._npz_cache[path]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec  = self.records[idx]
        data = self._get_npz(rec["npz_path"])
        i    = rec["idx"]

        matrix      = data["matrices"][i].astype(np.float32)       # [256, 256]
        boundary    = data["boundaries"][i].astype(np.float32)      # [256]
        compartment = data["compartments"][i].astype(np.float32)    # [256]

        # Random augmentation: transpose (Hi-C matrices are symmetric)
        if self.augment and random.random() > 0.5:
            matrix = matrix.T

        contact = torch.from_numpy(matrix).unsqueeze(0)             # [1, 256, 256]

        return {
            "contact":     contact,
            "boundary":    torch.from_numpy(boundary),
            "compartment": torch.from_numpy(compartment),
            "assay_id":    torch.tensor(rec["assay_id"], dtype=torch.long),
            "sample_id":   rec["sample_id"],
            "chr":         rec["chr"],
            "start_bp":    torch.tensor(rec["start_bp"], dtype=torch.long),
            "end_bp":      torch.tensor(rec["end_bp"], dtype=torch.long),
        }


def build_dataloaders(
    cell_lines: Optional[List[str]] = None,
    val_frac: float = 0.1,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders with reproducible split.
    Split is done at tile level (not sample level) for simplicity.
    """
    dataset = HiCTileDataset(cell_lines=cell_lines, augment=True)

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Run preprocess.py first:\n"
            "  cd /app/tmp/DATABASE_CONCEPT && python src/preprocess.py"
        )

    if len(dataset) < 4:
        # Too few tiles to split — use all for both train and val
        from torch.utils.data import Subset
        train_ds = dataset
        val_ds   = dataset
        val_ds.augment = False
    else:
        n_val   = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
        val_ds.dataset.augment = False

    n_train_effective = len(train_ds) if hasattr(train_ds, '__len__') else len(dataset)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, n_train_effective),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    print(f"[dataset] train={len(train_ds)} tiles | val={len(val_ds)} tiles")
    return train_loader, val_loader


def collate_fn_inference(batch):
    """
    Custom collate for inference — keeps string fields as lists.
    """
    contact     = torch.stack([b["contact"] for b in batch])
    assay_id    = torch.stack([b["assay_id"] for b in batch])
    boundary    = torch.stack([b["boundary"] for b in batch])
    compartment = torch.stack([b["compartment"] for b in batch])
    sample_ids  = [b["sample_id"] for b in batch]
    chroms      = [b["chr"] for b in batch]
    start_bps   = torch.stack([b["start_bp"] for b in batch])
    end_bps     = torch.stack([b["end_bp"] for b in batch])

    return {
        "contact":     contact,
        "boundary":    boundary,
        "compartment": compartment,
        "assay_id":    assay_id,
        "sample_ids":  sample_ids,
        "chroms":      chroms,
        "start_bps":   start_bps,
        "end_bps":     end_bps,
    }


def build_inference_loader(
    cell_lines: Optional[List[str]] = None,
    batch_size: int = 16,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    DataLoader for fingerprint extraction (no augmentation, no shuffle).
    """
    dataset = HiCTileDataset(cell_lines=cell_lines, augment=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_inference,
    )
