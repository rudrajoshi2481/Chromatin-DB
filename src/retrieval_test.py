"""
retrieval_test.py — Evaluate fingerprint quality via Recall@k retrieval metrics.

This tests the actual quality of embeddings for nearest-neighbor retrieval,
which is what matters for database applications. PCA/UMAP visualization
can be misleading; Recall@k is the ground truth.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import MQVAE
import config


def compute_recall_at_k(sim_matrix: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    Compute Recall@k: for each sample, check if any of the k nearest neighbors
    share the same label (cell type).
    
    Args:
        sim_matrix: [N, N] cosine similarity matrix
        labels: [N] cell type labels (integers or strings)
        k: number of nearest neighbors to check
    
    Returns:
        Recall@k (fraction of queries with at least 1 same-type neighbor in top-k)
    """
    n = len(labels)
    correct = 0
    
    for i in range(n):
        # Exclude self-similarity
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        
        # Get top-k indices (excluding self)
        top_k_idx = np.argsort(sims)[-k:]
        
        # Check if any neighbor has same label
        if any(labels[j] == labels[i] for j in top_k_idx):
            correct += 1
    
    return correct / n


def extract_fingerprints(model, loader, device, max_batches: int = None):
    """
    Extract fingerprints and cell type labels from dataloader.
    
    Returns:
        fingerprints: [N, FP_DIM] numpy array
        labels: [N] list of cell type strings
    """
    model.eval()
    fingerprints = []
    labels = []
    cell_idx_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            contact = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            
            # Get fingerprints
            fp = model.encode_fingerprint(contact, assay_id).cpu().numpy()
            fingerprints.append(fp)
            
            # Get labels
            batch_labels = batch.get("sample_id", ["unknown"] * len(fp))
            labels.extend(batch_labels)
            
            # Also get cell_idx for numeric comparison
            if "cell_idx" in batch:
                cell_idx_list.extend(batch["cell_idx"].cpu().numpy())
            else:
                cell_idx_list.extend([0] * len(fp))
            
            if max_batches and batch_idx >= max_batches:
                break
    
    fingerprints = np.concatenate(fingerprints, axis=0)
    cell_indices = np.array(cell_idx_list)
    
    return fingerprints, labels, cell_indices


def evaluate_checkpoint(checkpoint_path: str, split: str = "val", max_samples: int = 1000):
    """
    Load a checkpoint and evaluate fingerprint retrieval quality.
    
    Args:
        checkpoint_path: path to model checkpoint (.pt file)
        split: "train" or "val" to evaluate on
        max_samples: maximum number of samples to evaluate (for speed)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"[retrieval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Determine n_cell_types from checkpoint or config
    cell_lines = list(config.CELL_LINE_REGISTRY.keys())
    n_cell_types = len(cell_lines)
    
    # Initialize model with same architecture
    model = MQVAE(
        n_codes=ckpt.get("n_codes", config.N_CODES),
        code_dim=ckpt.get("code_dim", config.CODE_DIM),
        fp_dim=ckpt.get("fp_dim", config.FP_DIM),
        use_boundary_head=False,
        use_compartment_head=False,
        use_classifier_head=True,
        n_cell_types=n_cell_types,
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"[retrieval] Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Build dataloader
    train_loader, val_loader = build_dataloaders(batch_size=config.BATCH_SIZE)
    loader = val_loader if split == "val" else train_loader
    
    # Extract fingerprints
    print(f"[retrieval] Extracting fingerprints from {split} set...")
    fingerprints, labels, cell_indices = extract_fingerprints(
        model, loader, device, max_batches=max_samples // config.BATCH_SIZE
    )
    
    print(f"[retrieval] Extracted {len(fingerprints)} fingerprints ({config.FP_DIM}D each)")
    
    # Compute similarity matrix
    print("[retrieval] Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(fingerprints)
    
    # Convert string labels to integers for evaluation
    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[l] for l in labels])
    
    # Compute Recall@k for different k values
    print("\n" + "=" * 60)
    print("Retrieval Metrics (higher = better)")
    print("=" * 60)
    
    for k in [1, 3, 5, 10]:
        recall = compute_recall_at_k(sim_matrix, int_labels, k=k)
        print(f"Recall@{k:2d}:  {recall:.3f}  ({recall*100:.1f}%)")
    
    # Baseline: random chance
    n_classes = len(unique_labels)
    random_recall_5 = 1 - (1 - 1/n_classes) ** 5
    print(f"\nRandom baseline Recall@5: {random_recall_5:.3f} ({random_recall_5*100:.1f}%)")
    
    # Per-cell-type analysis
    print("\n" + "=" * 60)
    print("Per-Cell-Type Recall@5")
    print("=" * 60)
    
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        if mask.sum() < 5:
            continue
        
        # Get sub-matrix for this cell type
        sub_sim = sim_matrix[mask][:, mask]
        sub_labels = int_labels[mask]
        
        # Compute Recall@5 within this cell type
        recall_5 = compute_recall_at_k(sub_sim, sub_labels, k=5)
        n_samples = mask.sum()
        print(f"{label:30s}: {recall_5:.3f} ({n_samples:3d} samples)")
    
    # Quality assessment
    recall_5 = compute_recall_at_k(sim_matrix, int_labels, k=5)
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    if recall_5 >= 0.7:
        print(f"✓ EXCELLENT: Recall@5 = {recall_5:.3f} (≥0.7)")
        print("  Fingerprints are publication-quality for retrieval")
    elif recall_5 >= 0.5:
        print(f"○ GOOD: Recall@5 = {recall_5:.3f} (0.5-0.7)")
        print("  Fingerprints work for retrieval but could be improved")
    elif recall_5 >= 0.3:
        print(f"△ FAIR: Recall@5 = {recall_5:.3f} (0.3-0.5)")
        print("  Fingerprints have some structure but need improvement")
    else:
        print(f"✗ POOR: Recall@5 = {recall_5:.3f} (<0.3)")
        print("  Fingerprints lack discriminative structure")
    
    print(f"\nComparison to classifier accuracy:")
    if "classifier_acc" in ckpt:
        print(f"  Classifier accuracy: {ckpt['classifier_acc']:.3f}")
        print(f"  Recall@5:            {recall_5:.3f}")
        gap = abs(ckpt["classifier_acc"] - recall_5)
        if gap > 0.2:
            print(f"  ⚠ Large gap ({gap:.2f}) suggests embeddings don't cluster geometrically")
        else:
            print(f"  ✓ Small gap ({gap:.2f}) suggests embeddings capture cell-type structure")
    
    print("=" * 60)
    
    return {
        "recall@1": compute_recall_at_k(sim_matrix, int_labels, k=1),
        "recall@3": compute_recall_at_k(sim_matrix, int_labels, k=3),
        "recall@5": compute_recall_at_k(sim_matrix, int_labels, k=5),
        "recall@10": compute_recall_at_k(sim_matrix, int_labels, k=10),
        "n_samples": len(fingerprints),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fingerprint retrieval quality")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    results = evaluate_checkpoint(args.checkpoint, args.split, args.max_samples)
    
    print(f"\n[retrieval] Evaluation complete!")
