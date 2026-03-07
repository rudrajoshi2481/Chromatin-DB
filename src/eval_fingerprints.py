"""
eval_fingerprints.py — Comprehensive fingerprint quality evaluation.

Tests:
  1. Retrieval metrics: Recall@k, mAP, mean cosine intra/inter distance
  2. Per cell-type silhouette score breakdown
  3. Classifier accuracy on fingerprint space (kNN)
  4. Publication-quality figures:
     a. UMAP colored by cell type (window-level, cosine)
     b. Recall@k bar chart
     c. Per cell-type silhouette heatmap
     d. Intra vs inter class distance matrix
     e. Codebook usage heatmap

Usage:
    python src/eval_fingerprints.py --checkpoint checkpoints/rebalanced_v2/mqvae_epoch037_best.pt
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import MQVAE
from dashboard import _umap_2d, _palette_pub
import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_fingerprints(model, loader, device, max_samples=4000):
    """Extract window-level fingerprints + labels from a dataloader."""
    model.eval()
    fps, labels, chrs, starts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            contact  = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            fp = model.encode_fingerprint(contact, assay_id).cpu().float().numpy()
            fps.append(fp)
            labels.extend(batch.get("sample_id", ["unknown"] * len(fp)))
            chrs.extend(batch.get("chr", ["?"] * len(fp)))
            starts.extend(batch.get("start_bp", [0] * len(fp)))
            if sum(len(f) for f in fps) >= max_samples:
                break
    fps = np.concatenate(fps, axis=0)[:max_samples]
    labels = labels[:max_samples]
    return fps, labels, chrs[:max_samples], starts[:max_samples]


def recall_at_k(sim: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Recall@k: fraction of queries with ≥1 same-class neighbor in top-k."""
    n = len(labels)
    correct = 0
    for i in range(n):
        row = sim[i].copy()
        row[i] = -2  # exclude self
        top_k = np.argsort(row)[-k:]
        if any(labels[j] == labels[i] for j in top_k):
            correct += 1
    return correct / n


def mean_ap(sim: np.ndarray, labels: np.ndarray) -> float:
    """Mean Average Precision for cell-type retrieval."""
    n = len(labels)
    aps = []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -2
        order = np.argsort(row)[::-1]
        same = (labels[order] == labels[i]).astype(float)
        n_pos = same.sum()
        if n_pos == 0:
            continue
        cumsum = np.cumsum(same)
        ranks  = np.arange(1, n + 1)
        precisions = (cumsum / ranks) * same
        aps.append(precisions.sum() / n_pos)
    return float(np.mean(aps)) if aps else 0.0


def knn_accuracy(fps_norm: np.ndarray, labels: list, k: int = 5) -> float:
    """kNN classification accuracy on fingerprint space."""
    label_arr = np.array(labels)
    unique = np.unique(label_arr)
    if len(unique) < 2 or len(fps_norm) < k + 1:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(fps_norm, label_arr)
    preds = knn.predict(fps_norm)
    # leave-one-out is expensive, use training set (upper bound, still informative)
    return float((preds == label_arr).mean())


# ── Figure routines ───────────────────────────────────────────────────────────

def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdc3c7")
    ax.spines["bottom"].set_color("#bdc3c7")
    ax.tick_params(colors="#555", labelsize=9)
    if title:   ax.set_title(title, fontsize=11, color="#2c3e50", pad=8, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def fig_umap(fps_norm, labels, sil, out_path):
    """Figure 1: UMAP with per-cell-type silhouette in title."""
    print("[eval] Computing UMAP...")
    emb = _umap_2d(fps_norm)

    unique_labels = sorted(set(labels))
    palette = _palette_pub(len(unique_labels))
    lbl2col = {l: palette[i] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        pts  = emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[lbl2col[lbl]], s=50, alpha=0.75,
                   label=lbl, edgecolors="white", linewidths=0.6)

    ax.set_title(
        f"Fingerprint UMAP  —  window-level, cosine metric  |  Silhouette = {sil:.3f}",
        fontsize=13, fontweight="bold", color="#2c3e50", pad=10,
    )
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.legend(fontsize=8, frameon=True, loc="center left",
              bbox_to_anchor=(1.01, 0.5), ncol=1, borderaxespad=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return emb


def fig_retrieval_summary(metrics_dict, random_recall5, out_path):
    """Figure 2: Recall@k + mAP + kNN bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Fingerprint Retrieval Quality", fontsize=14, fontweight="bold",
                 color="#2c3e50", y=1.02)

    # --- Recall@k
    ax = axes[0]
    ks     = [1, 3, 5, 10, 20]
    recalls = [metrics_dict[f"recall@{k}"] for k in ks]
    bars = ax.bar([str(k) for k in ks], recalls, color="#3498db", edgecolor="white", linewidth=1.5)
    ax.axhline(random_recall5, color="#e74c3c", ls="--", lw=1.5, label=f"Random @5 = {random_recall5:.3f}")
    ax.axhline(0.7, color="#27ae60", ls=":", lw=1.2, label="Target 0.70")
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, frameon=True)
    _style(ax, title="Recall@k", xlabel="k", ylabel="Recall")

    # --- mAP
    ax = axes[1]
    ax.barh(["mAP"], [metrics_dict["mAP"]], color="#9b59b6", edgecolor="white")
    ax.barh(["kNN-5 Acc"], [metrics_dict["knn5_acc"]], color="#e67e22", edgecolor="white")
    ax.barh(["Silhouette"], [max(0, metrics_dict["silhouette"])], color="#1abc9c", edgecolor="white")
    ax.set_xlim(0, 1.05)
    for i, (name, val) in enumerate([("mAP", metrics_dict["mAP"]),
                                     ("kNN-5 Acc", metrics_dict["knn5_acc"]),
                                     ("Silhouette", metrics_dict["silhouette"])]):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    _style(ax, title="Summary Metrics", xlabel="Score")

    # --- Per cell-type silhouette
    ax = axes[2]
    per_ct = metrics_dict["per_celltype_silhouette"]
    names  = list(per_ct.keys())
    vals   = list(per_ct.values())
    colors = ["#27ae60" if v > 0.3 else "#f39c12" if v > 0 else "#e74c3c" for v in vals]
    bars = ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="#2c3e50", lw=0.8)
    ax.axvline(0.3, color="#27ae60", ls="--", lw=1.2, alpha=0.7, label="Target 0.3")
    ax.set_xlim(-0.5, 1.0)
    ax.legend(fontsize=8)
    _style(ax, title="Per Cell-Type Silhouette (cosine)", xlabel="Silhouette score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig_distance_matrix(fps_norm, labels, out_path):
    """Figure 3: Mean cosine similarity heat-map (cell-type × cell-type)."""
    unique_labels = sorted(set(labels))
    label_arr     = np.array(labels)
    sim = cosine_similarity(fps_norm)

    n = len(unique_labels)
    mean_sim = np.zeros((n, n))
    for i, li in enumerate(unique_labels):
        mi = label_arr == li
        for j, lj in enumerate(unique_labels):
            mj = label_arr == lj
            mean_sim[i, j] = sim[np.ix_(mi, mj)].mean()

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.65)))
    cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f8f9fa", "#27ae60"])
    im = ax.imshow(mean_sim, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean cosine similarity")

    ax.set_xticks(range(n)); ax.set_xticklabels(unique_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(unique_labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mean_sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black" if 0.3 < mean_sim[i, j] < 0.8 else "white")

    ax.set_title("Mean Cosine Similarity — Cell-Type vs Cell-Type\n"
                 "Diagonal = intra-class; off-diagonal = inter-class",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig_intra_inter(fps_norm, labels, out_path):
    """Figure 4: Intra vs inter cosine similarity distributions per cell type."""
    unique_labels = sorted(set(labels))
    label_arr     = np.array(labels)
    sim = cosine_similarity(fps_norm)
    np.fill_diagonal(sim, np.nan)

    palette = _palette_pub(len(unique_labels))
    n_types = len(unique_labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Intra vs Inter Cell-Type Cosine Similarity",
                 fontsize=13, fontweight="bold", y=1.02)

    # Left: violin / box per cell type (intra)
    ax = axes[0]
    intra_data, intra_labels_plot = [], []
    for lbl in unique_labels:
        mask = label_arr == lbl
        sub  = sim[np.ix_(mask, mask)]
        vals = sub[~np.isnan(sub)].flatten()
        intra_data.append(vals)
        intra_labels_plot.append(lbl)

    bp = ax.boxplot(intra_data, vert=False, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, col in zip(bp["boxes"], palette):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_yticks(range(1, n_types + 1))
    ax.set_yticklabels(intra_labels_plot, fontsize=8)
    ax.axvline(0, color="#e74c3c", ls="--", lw=1)
    _style(ax, title="Intra-Class Similarity (each cell type vs self)",
           xlabel="Cosine similarity")

    # Right: global intra vs inter density
    ax = axes[1]
    all_intra, all_inter = [], []
    for lbl in unique_labels:
        mask = label_arr == lbl
        sub_intra = sim[np.ix_(mask, mask)]
        all_intra.extend(sub_intra[~np.isnan(sub_intra)].flatten())
        sub_inter = sim[np.ix_(mask, ~mask)]
        all_inter.extend(sub_inter[~np.isnan(sub_inter)].flatten())

    bins = np.linspace(-0.2, 1.0, 80)
    ax.hist(all_intra, bins=bins, density=True, alpha=0.65,
            color="#27ae60", label=f"Intra (μ={np.mean(all_intra):.3f})")
    ax.hist(all_inter, bins=bins, density=True, alpha=0.65,
            color="#e74c3c", label=f"Inter (μ={np.mean(all_inter):.3f})")
    ax.legend(fontsize=9, frameon=True)
    sep = np.mean(all_intra) - np.mean(all_inter)
    ax.set_title(f"Intra vs Inter Similarity Distributions\nSeparation gap = {sep:.3f}",
                 fontsize=11, fontweight="bold", pad=8)
    _style(ax, xlabel="Cosine similarity", ylabel="Density")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return float(np.mean(all_intra)), float(np.mean(all_inter))


def fig_codebook(model, out_path):
    """Figure 5: Codebook usage sorted bar chart."""
    usage = model.vq.usage_count.cpu().float().numpy()
    active = int((usage > 0).sum())
    n_codes = len(usage)

    sorted_usage = np.sort(usage)[::-1]
    cumulative   = np.cumsum(sorted_usage) / sorted_usage.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = ["#27ae60" if u > 0 else "#e74c3c" for u in sorted_usage]
    ax.bar(range(n_codes), sorted_usage, color=colors, width=1.0, edgecolor="none")
    ax.set_title(f"Codebook Usage  —  {active}/{n_codes} active codes\n"
                 f"({active/n_codes*100:.1f}% utilization)",
                 fontsize=11, fontweight="bold", pad=8)
    _style(ax, xlabel="Code rank (sorted by usage)", ylabel="Usage count")

    ax = axes[1]
    ax.plot(range(n_codes), cumulative * 100, color="#3498db", lw=2)
    ax.axhline(80, color="#e74c3c", ls="--", lw=1.2, label="80% threshold")
    ax.axhline(95, color="#f39c12", ls="--", lw=1.2, label="95% threshold")
    n80 = int(np.searchsorted(cumulative, 0.80)) + 1
    n95 = int(np.searchsorted(cumulative, 0.95)) + 1
    ax.text(n80, 82, f"{n80} codes →", fontsize=8, color="#e74c3c")
    ax.text(n95, 97, f"{n95} codes →", fontsize=8, color="#f39c12")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    _style(ax, title="Cumulative Usage (Pareto curve)",
           xlabel="Top-k codes", ylabel="Cumulative usage (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return {"active_codes": active, "n_codes": n_codes,
            "codes_for_80pct": n80, "codes_for_95pct": n95}


def fig_combined(fps_norm, labels, emb, metrics_dict, sil, intra_mu, inter_mu, out_path):
    """Figure 6: One-page combined summary for presentation."""
    unique_labels = sorted(set(labels))
    palette = _palette_pub(len(unique_labels))
    lbl2col = {l: palette[i] for i, l in enumerate(unique_labels)}

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("MQ-VAE Fingerprint Quality Report  —  rebalanced_v2",
                 fontsize=16, fontweight="bold", color="#2c3e50", y=1.01)

    # UMAP (top row, spans 2 cols)
    ax = fig.add_subplot(gs[0, :2])
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[lbl2col[lbl]],
                   s=50, alpha=0.75, label=lbl, edgecolors="white", linewidths=0.5)
    ax.set_title(f"Fingerprint UMAP  |  Silhouette={sil:.3f}", fontsize=11,
                 fontweight="bold", pad=6)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    ax.set_facecolor("#f8f9fa"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Recall@k (top right)
    ax = fig.add_subplot(gs[0, 2])
    ks = [1, 3, 5, 10, 20]
    vals = [metrics_dict[f"recall@{k}"] for k in ks]
    bars = ax.bar([str(k) for k in ks], vals, color="#3498db", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    _style(ax, title="Recall@k", xlabel="k", ylabel="Recall")

    # Per-CT silhouette (middle left)
    ax = fig.add_subplot(gs[1, 0])
    per_ct = metrics_dict["per_celltype_silhouette"]
    names  = list(per_ct.keys())
    vals2  = list(per_ct.values())
    cols   = ["#27ae60" if v > 0.3 else "#f39c12" if v > 0 else "#e74c3c" for v in vals2]
    ax.barh(names, vals2, color=cols, edgecolor="white")
    ax.axvline(0, color="#2c3e50", lw=0.8)
    _style(ax, title="Silhouette per Cell Type", xlabel="Score")

    # Intra/inter dist (middle centre)
    ax = fig.add_subplot(gs[1, 1])
    label_arr = np.array(labels)
    sim = cosine_similarity(fps_norm)
    np.fill_diagonal(sim, np.nan)
    all_intra, all_inter = [], []
    for lbl in unique_labels:
        mask = label_arr == lbl
        all_intra.extend(sim[np.ix_(mask, mask)][~np.isnan(sim[np.ix_(mask, mask)])].flatten())
        all_inter.extend(sim[np.ix_(mask, ~mask)].flatten())
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(all_intra, bins=bins, density=True, alpha=0.65, color="#27ae60",
            label=f"Intra μ={intra_mu:.3f}")
    ax.hist(all_inter, bins=bins, density=True, alpha=0.65, color="#e74c3c",
            label=f"Inter μ={inter_mu:.3f}")
    ax.legend(fontsize=8)
    _style(ax, title="Similarity Distributions", xlabel="Cosine similarity")

    # Summary text (middle right)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary = (
        f"Retrieval Quality Summary\n"
        f"{'─'*32}\n"
        f"Recall@1:       {metrics_dict['recall@1']:.3f}\n"
        f"Recall@3:       {metrics_dict['recall@3']:.3f}\n"
        f"Recall@5:       {metrics_dict['recall@5']:.3f}\n"
        f"Recall@10:      {metrics_dict['recall@10']:.3f}\n"
        f"Recall@20:      {metrics_dict['recall@20']:.3f}\n"
        f"mAP:            {metrics_dict['mAP']:.3f}\n"
        f"kNN-5 Acc:      {metrics_dict['knn5_acc']:.3f}\n"
        f"Silhouette:     {sil:.3f}\n"
        f"Intra-sim μ:    {intra_mu:.3f}\n"
        f"Inter-sim μ:    {inter_mu:.3f}\n"
        f"Gap:            {intra_mu - inter_mu:.3f}\n"
        f"N windows:      {len(labels)}\n"
        f"N cell types:   {len(unique_labels)}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9,
                      edgecolor="#34495e", lw=2), color="#2c3e50")

    # Bottom row: distance matrix
    ax = fig.add_subplot(gs[2, :])
    unique_labels_list = sorted(set(labels))
    n = len(unique_labels_list)
    mean_sim = np.zeros((n, n))
    for i, li in enumerate(unique_labels_list):
        mi = label_arr == li
        for j, lj in enumerate(unique_labels_list):
            mj = label_arr == lj
            block = sim[np.ix_(mi, mj)]
            block = block[~np.isnan(block)]
            mean_sim[i, j] = block.mean() if len(block) > 0 else 0.0

    cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f8f9fa", "#27ae60"])
    im = ax.imshow(mean_sim, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean cosine similarity", orientation="vertical")
    ax.set_xticks(range(n)); ax.set_xticklabels(unique_labels_list, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(unique_labels_list, fontsize=8)
    for i in range(n):
        for j in range(n):
            c = "white" if mean_sim[i, j] < 0.3 or mean_sim[i, j] > 0.75 else "black"
            ax.text(j, i, f"{mean_sim[i,j]:.2f}", ha="center", va="center", fontsize=7, color=c)
    ax.set_title("Cell-Type Mean Cosine Similarity Matrix  (diagonal = intra-class)",
                 fontsize=11, fontweight="bold", pad=6)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eval(checkpoint_path: str, out_dir: Path, max_samples: int = 4000):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    n_cell_types = arch.get("n_cell_types", len(list(config.CELL_LINE_REGISTRY.keys())))

    model = MQVAE(
        use_boundary_head    = False,
        use_compartment_head = False,
        use_classifier_head  = True,
        n_cell_types         = n_cell_types,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"[eval] Model: {sum(p.numel() for p in model.parameters()):,} params  |  device: {device}")

    # ── Extract fingerprints (train + val combined for richer UMAP/retrieval) ─
    print("[eval] Building dataloaders...")
    train_loader, val_loader = build_dataloaders(batch_size=64, num_workers=4)

    print("[eval] Extracting fingerprints (val set)...")
    fps_val, labels_val, _, _ = extract_fingerprints(model, val_loader, device, max_samples)

    print("[eval] Extracting fingerprints (train set)...")
    fps_train, labels_train, _, _ = extract_fingerprints(model, train_loader, device, max_samples)

    # Combine both splits for full-database analysis
    fps_all    = np.concatenate([fps_train, fps_val], axis=0)
    labels_all = labels_train + labels_val
    print(f"[eval] Total fingerprints: {len(fps_all)}  ({len(set(labels_all))} cell types)")

    # L2-normalise for cosine ops
    fps_norm = sk_normalize(fps_all, norm="l2")
    label_arr = np.array(labels_all)

    # ── Retrieval metrics ─────────────────────────────────────────────────────
    print("[eval] Computing cosine similarity matrix...")
    sim = cosine_similarity(fps_norm)

    print("[eval] Computing Recall@k and mAP...")
    unique_labels = sorted(set(labels_all))
    int_labels    = np.array([unique_labels.index(l) for l in labels_all])

    metrics = {}
    for k in [1, 3, 5, 10, 20]:
        metrics[f"recall@{k}"] = recall_at_k(sim, int_labels, k)
        print(f"  Recall@{k:2d} = {metrics[f'recall@{k}']:.4f}")

    metrics["mAP"]      = mean_ap(sim, int_labels)
    metrics["knn5_acc"] = knn_accuracy(fps_norm, labels_all, k=5)
    print(f"  mAP       = {metrics['mAP']:.4f}")
    print(f"  kNN-5 acc = {metrics['knn5_acc']:.4f}")

    # ── Silhouette ────────────────────────────────────────────────────────────
    sil_global = float(silhouette_score(fps_norm, label_arr, metric="cosine"))
    sil_samples = silhouette_samples(fps_norm, label_arr, metric="cosine")
    per_ct_sil = {}
    for lbl in unique_labels:
        mask = label_arr == lbl
        per_ct_sil[lbl] = float(sil_samples[mask].mean())
    metrics["silhouette"]               = sil_global
    metrics["per_celltype_silhouette"]  = per_ct_sil
    print(f"  Silhouette (global) = {sil_global:.4f}")
    for lbl, v in per_ct_sil.items():
        print(f"    {lbl:30s}: {v:.4f}")

    # ── Random baseline ────────────────────────────────────────────────────────
    n_classes = len(unique_labels)
    random_recall5 = 1 - (1 - 1/n_classes) ** 5
    metrics["random_recall5"] = random_recall5

    # ── Print quality verdict ─────────────────────────────────────────────────
    r5 = metrics["recall@5"]
    print("\n" + "=" * 55)
    print("  FINGERPRINT QUALITY VERDICT")
    print("=" * 55)
    if r5 >= 0.8:
        verdict = "EXCELLENT — publication ready"
    elif r5 >= 0.65:
        verdict = "GOOD — suitable for retrieval"
    elif r5 >= 0.45:
        verdict = "FAIR — usable but needs improvement"
    else:
        verdict = "POOR — needs more training"
    print(f"  Recall@5  = {r5:.3f}   ({verdict})")
    print(f"  mAP       = {metrics['mAP']:.3f}")
    print(f"  Silhouette= {sil_global:.3f}")
    print(f"  Random R@5= {random_recall5:.3f}")
    print("=" * 55)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[eval] Generating figures...")

    emb = fig_umap(fps_norm, labels_all, sil_global, out_dir / "fig1_umap.png")
    fig_retrieval_summary(metrics, random_recall5,                 out_dir / "fig2_retrieval.png")
    fig_distance_matrix(fps_norm, labels_all,                      out_dir / "fig3_distance_matrix.png")
    intra_mu, inter_mu = fig_intra_inter(fps_norm, labels_all,     out_dir / "fig4_intra_inter.png")
    cb_stats = fig_codebook(model,                                  out_dir / "fig5_codebook.png")
    fig_combined(fps_norm, labels_all, emb, metrics, sil_global,
                 intra_mu, inter_mu,                                out_dir / "fig6_combined.png")

    metrics["intra_sim_mean"] = intra_mu
    metrics["inter_sim_mean"] = inter_mu
    metrics["separation_gap"] = intra_mu - inter_mu
    metrics.update(cb_stats)
    metrics["n_windows"] = len(fps_all)
    metrics["n_cell_types"] = n_classes
    metrics["checkpoint"] = str(checkpoint_path)

    # Save JSON results
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[eval] Results saved → {results_path}")
    print(f"[eval] All figures → {out_dir}/")

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir",    type=str, default=None)
    p.add_argument("--max_samples", type=int, default=4000)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir   = Path(args.out_dir) if args.out_dir else \
                Path("/media/rudhra/ChenLabData1/Joshi/exp/Chromatin-DB/trash/eval") / ckpt_path.stem

    run_eval(str(ckpt_path), out_dir, args.max_samples)
