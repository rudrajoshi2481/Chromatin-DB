"""
umap_plot.py — Generate standalone UMAP visualization of fingerprints.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import MQVAE
from dashboard import collect_fingerprints, _umap_2d
import config

def generate_umap_plot(checkpoint_path: str, output_path: str = None):
    """Generate and save UMAP plot of fingerprints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"[umap] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    cell_lines = list(config.CELL_LINE_REGISTRY.keys())
    n_cell_types = len(cell_lines)
    
    model = MQVAE(
        use_boundary_head=False,
        use_compartment_head=False,
        use_classifier_head=True,
        n_cell_types=n_cell_types,
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Collect fingerprints
    _, val_loader = build_dataloaders(batch_size=config.BATCH_SIZE)
    fingerprints, labels = collect_fingerprints(model, val_loader, device, max_samples=1000)
    
    print(f"[umap] Collected {len(fingerprints)} fingerprints")
    
    # Compute UMAP
    print("[umap] Computing UMAP...")
    umap_2d = _umap_2d(fingerprints)
    
    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(set(labels))
    
    # Use distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array([l == label for l in labels])
        pts = umap_2d[mask]
        if len(pts):
            plt.scatter(pts[:, 0], pts[:, 1], 
                       c=[colors[i]], s=80, alpha=0.8, 
                       label=label, edgecolors='white', linewidths=1)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.title(f"UMAP of Fingerprints\n{len(unique_labels)} Cell Types Separated", 
              fontsize=14, fontweight='bold')
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = f"umap_fingerprints.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[umap] Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    generate_umap_plot(args.checkpoint, args.output)
