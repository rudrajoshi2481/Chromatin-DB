"""
ablation.py — Ablation study runner for MQ-VAE.

Runs 10 ablation configs, saves results to /app/tmp/DATABASE_CONCEPT/trash/
Outputs:
  trash/ablation_results.json       — raw metrics per config
  trash/ablation_summary.txt        — formatted comparison table
  trash/ablation_dim_sweep.json     — fingerprint dimensionality AUC results
  trash/ablation_plots/             — optional matplotlib figures

Usage:
    cd /app/tmp/DATABASE_CONCEPT
    python src/ablation.py
    python src/ablation.py --epochs 5   # fast smoke-test
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TRASH_DIR, CHECKPOINTS_DIR, DATA_DIR,
    ABLATION_EPOCHS, CELL_LINE_REGISTRY, FP_DIM,
)
from train import train
from dataset import build_inference_loader
from model import MQVAE


# ── Ablation configuration table ──────────────────────────────────────────────

ABLATION_CONFIGS = [
    {
        "id":    1,
        "name":  "baseline_recon_only",
        "desc":  "Recon-only, no auxiliary heads",
        "kwargs": dict(use_boundary_head=False, use_compartment_head=False,
                       use_masking=True, use_film=True, n_codes=512),
    },
    {
        "id":    2,
        "name":  "boundary_head_only",
        "desc":  "+ Boundary head only",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=False,
                       use_masking=True, use_film=True, n_codes=512),
    },
    {
        "id":    3,
        "name":  "compartment_head_only",
        "desc":  "+ Compartment head only",
        "kwargs": dict(use_boundary_head=False, use_compartment_head=True,
                       use_masking=True, use_film=True, n_codes=512),
    },
    {
        "id":    4,
        "name":  "full_v4",
        "desc":  "Full v4: both heads + masking + FiLM",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=True,
                       use_masking=True, use_film=True, n_codes=512),
    },
    {
        "id":    6,
        "name":  "no_masking",
        "desc":  "No masking (all tokens visible)",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=True,
                       use_masking=False, use_film=True, n_codes=512),
    },
    {
        "id":    7,
        "name":  "no_film",
        "desc":  "Additive assay inject vs FiLM (ablate FiLM)",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=True,
                       use_masking=True, use_film=False, n_codes=512),
    },
    {
        "id":    10,
        "name":  "codebook_256",
        "desc":  "256-code codebook",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=True,
                       use_masking=True, use_film=True, n_codes=256),
    },
    {
        "id":    11,
        "name":  "codebook_1024",
        "desc":  "1024-code codebook",
        "kwargs": dict(use_boundary_head=True, use_compartment_head=True,
                       use_masking=True, use_film=True, n_codes=1024),
    },
]


# ── MAP@K metric ──────────────────────────────────────────────────────────────

def compute_map_at_k(
    fingerprints: np.ndarray,   # [N, D]
    labels:       np.ndarray,   # [N]  cell-type label (int)
    k:            int = 5,
) -> float:
    """
    Mean Average Precision @ K over all fingerprints.
    A retrieval is 'relevant' if it has the same cell-type label.
    """
    N     = len(fingerprints)
    fps_n = fingerprints / (np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8)
    sims  = fps_n @ fps_n.T                    # [N, N]
    np.fill_diagonal(sims, -1.0)               # exclude self

    aps = []
    for i in range(N):
        ranked = np.argsort(sims[i])[::-1][:k]
        hits   = (labels[ranked] == labels[i]).astype(float)
        if hits.sum() == 0:
            aps.append(0.0)
            continue
        precisions = np.cumsum(hits) / (np.arange(k) + 1)
        aps.append(float((precisions * hits).sum() / min(k, hits.sum())))

    return float(np.mean(aps))


# ── Fingerprint extraction for a trained model ────────────────────────────────

def extract_fingerprints_for_ablation(
    model:     MQVAE,
    device:    torch.device,
    cell_lines: Optional[List[str]] = None,
) -> tuple:
    """
    Run model over all tiles, return (fingerprints [N, D], cell_type_labels [N]).
    """
    if cell_lines is None:
        cell_lines = list(CELL_LINE_REGISTRY.keys())

    label_map = {s: i for i, s in enumerate(cell_lines)}
    loader    = build_inference_loader(cell_lines=cell_lines, batch_size=16, num_workers=0)

    fps_list    = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            contact  = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            fp       = model.encode_fingerprint(contact, assay_id).cpu().float().numpy()
            fps_list.append(fp)
            labels_list.extend([label_map.get(s, -1) for s in batch["sample_ids"]])

    if not fps_list:
        return np.zeros((0, FP_DIM)), np.zeros(0, dtype=int)

    fingerprints = np.concatenate(fps_list, axis=0)
    labels       = np.array(labels_list, dtype=int)
    return fingerprints, labels


# ── Ablation #5: fingerprint dimensionality sweep via PCA ─────────────────────

def ablation_dim_sweep(
    fingerprints: np.ndarray,
    labels:       np.ndarray,
    dims:         List[int] = [8, 16, 24, 32],
    k:            int = 5,
) -> Dict[int, float]:
    """
    Evaluate MAP@K after projecting fingerprints to lower dimensions via PCA.
    """
    from sklearn.decomposition import PCA

    results = {}
    for d in dims:
        if d >= fingerprints.shape[1]:
            reduced = fingerprints
        else:
            pca     = PCA(n_components=d, random_state=42)
            reduced = pca.fit_transform(fingerprints)
        results[d] = compute_map_at_k(reduced, labels, k)
        print(f"  PCA dim={d:2d}  MAP@{k}={results[d]:.4f}")
    return results


# ── Main ablation runner ───────────────────────────────────────────────────────

def run_ablations(
    epochs:     int = ABLATION_EPOCHS,
    cell_lines: Optional[List[str]] = None,
    device_str: str = "auto",
    configs:    Optional[List[int]] = None,
) -> List[Dict]:
    """
    Run all ablation configs and collect metrics.
    Returns list of result dicts.
    """
    TRASH_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if cell_lines is None:
        cell_lines = list(CELL_LINE_REGISTRY.keys())

    active_configs = [c for c in ABLATION_CONFIGS
                      if configs is None or c["id"] in configs]

    all_results = []
    full_v4_fps    = None
    full_v4_labels = None

    for cfg in active_configs:
        print(f"\n{'='*60}")
        print(f"Ablation #{cfg['id']}: {cfg['desc']}")
        print(f"{'='*60}")

        run_name = f"ablation_{cfg['name']}"
        ckpt_dir = TRASH_DIR / "checkpoints" / run_name

        # Train
        train_results = train(
            cell_lines           = cell_lines,
            epochs               = epochs,
            run_name             = run_name,
            ckpt_dir             = ckpt_dir,
            device_str           = device_str,
            **cfg["kwargs"],
        )

        # Load best checkpoint
        best_ckpts = sorted(ckpt_dir.glob("*_best.pt"))
        if not best_ckpts:
            best_ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not best_ckpts:
            print(f"  No checkpoint found for {run_name}, skipping eval")
            continue

        ckpt  = torch.load(str(best_ckpts[-1]), map_location=device)
        model = MQVAE(**cfg["kwargs"]).to(device)
        model.load_state_dict(ckpt["model"])

        # Extract fingerprints + compute MAP@5
        fps, labels = extract_fingerprints_for_ablation(model, device, cell_lines)
        map_at_5    = compute_map_at_k(fps, labels, k=5) if len(fps) > 5 else 0.0

        # Codebook perplexity from final step
        active_codes = int((model.vq.usage_count > 0).sum().item())

        result = {
            "ablation_id":     cfg["id"],
            "name":            cfg["name"],
            "desc":            cfg["desc"],
            "epochs":          epochs,
            "best_val_loss":   train_results["best_val_loss"],
            "best_epoch":      train_results["best_epoch"],
            "map_at_5":        round(map_at_5, 4),
            "active_codes":    active_codes,
            **cfg["kwargs"],
        }

        # Extract per-metric from train history
        history = train_results.get("history", [])
        if history:
            last = history[-1]
            result.update({
                "final_val_recon":      last.get("val_recon", 0),
                "final_val_boundary_f1": last.get("val_boundary_f1", 0),
                "final_val_compartment_r": last.get("val_compartment_r", 0),
            })

        all_results.append(result)
        print(f"  ✓ MAP@5={map_at_5:.4f}  active_codes={active_codes}  val_loss={train_results['best_val_loss']:.4f}")

        # Stash full_v4 fingerprints for dim sweep (ablation #5)
        if cfg["id"] == 4:
            full_v4_fps    = fps
            full_v4_labels = labels

    # ── Ablation #5: dimension sweep on full_v4 fingerprints ─────────────────
    dim_results = {}
    if full_v4_fps is not None and len(full_v4_fps) > 5:
        print("\n── Ablation #5: Fingerprint Dimensionality Sweep ──")
        dim_results = ablation_dim_sweep(full_v4_fps, full_v4_labels)

        dim_path = TRASH_DIR / "ablation_dim_sweep.json"
        with open(dim_path, "w") as f:
            json.dump({str(k): v for k, v in dim_results.items()}, f, indent=2)
        print(f"  Dim sweep saved to {dim_path}")

    # ── Save full results ─────────────────────────────────────────────────────
    results_path = TRASH_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ablation] Results saved to {results_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary_table(all_results, dim_results)

    return all_results


def _print_summary_table(results: List[Dict], dim_results: Dict) -> None:
    """Print and save a formatted ablation comparison table."""
    header = (
        f"{'#':<4} {'Name':<30} {'MAP@5':>6} {'Val Loss':>9} "
        f"{'Bdry F1':>8} {'Comp r':>7} {'Codes':>6}"
    )
    sep = "─" * len(header)

    lines = [
        "\n" + "═" * len(header),
        "  ABLATION STUDY RESULTS",
        "═" * len(header),
        header,
        sep,
    ]

    for r in results:
        lines.append(
            f"{r['ablation_id']:<4} {r['name']:<30} "
            f"{r['map_at_5']:>6.4f} {r['best_val_loss']:>9.4f} "
            f"{r.get('final_val_boundary_f1', 0):>8.3f} "
            f"{r.get('final_val_compartment_r', 0):>7.3f} "
            f"{r['active_codes']:>6}"
        )

    lines.append(sep)

    if dim_results:
        lines.append("\n  Ablation #5: Fingerprint Dimensionality (PCA on Full v4)")
        lines.append(f"  {'Dim':<8} {'MAP@5':>6}")
        for d, auc in sorted(dim_results.items()):
            lines.append(f"  {d:<8} {auc:>6.4f}")

    # Decision points
    baseline_map = next((r["map_at_5"] for r in results if r["ablation_id"] == 1), None)
    full_map     = next((r["map_at_5"] for r in results if r["ablation_id"] == 4), None)
    lines.append("\n  Decision Points:")
    if baseline_map is not None and full_map is not None:
        improvement = full_map - baseline_map
        lines.append(f"  Full v4 vs Baseline: +{improvement:.4f} MAP@5 "
                     f"({'significant' if improvement > 0.02 else 'marginal'})")
    if dim_results:
        best_dim  = max(dim_results, key=dim_results.get)
        worst_dim = min(dim_results, key=dim_results.get)
        drop = dim_results.get(32, 0) - dim_results.get(16, 0)
        lines.append(f"  Best dim: {best_dim} | d=16 vs d=32 drop: {drop:.4f} "
                     f"({'keep d=32' if drop > 0.03 else 'd=16 sufficient'})")

    table_str = "\n".join(lines)
    print(table_str)

    summary_path = TRASH_DIR / "ablation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(table_str + "\n")
    print(f"[ablation] Summary saved to {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run MQ-VAE ablation studies")
    p.add_argument("--epochs",      type=int,  default=ABLATION_EPOCHS)
    p.add_argument("--device",      type=str,  default="auto")
    p.add_argument("--cell_lines",  nargs="+", default=None)
    p.add_argument("--configs",     nargs="+", type=int, default=None,
                   help="Ablation IDs to run (default: all). E.g., --configs 1 4 6")
    args = p.parse_args()

    run_ablations(
        epochs     = args.epochs,
        cell_lines = args.cell_lines,
        device_str = args.device,
        configs    = args.configs,
    )
