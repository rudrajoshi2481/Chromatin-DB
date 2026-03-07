"""
dashboard.py — Live training dashboard.

Single PNG file updated every epoch showing:
  Row 1: Loss curves (train/val total, recon, vq)
  Row 2: Task metrics (boundary F1, compartment R)
  Row 3: Codebook usage (active codes, histogram)
  Row 4: Fingerprint PCA (coloured by cell type/replicate group)

Usage:
    from dashboard import TrainingDashboard
    dash = TrainingDashboard(run_name="full", out_dir=TRASH_DIR)
    dash.update(epoch, train_metrics, val_metrics, model, dataloader)

Output:
    trash/dashboard_<run_name>.png   — updated every epoch
    trash/dashboard_<run_name>.json  — full history for later analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import torch
from sklearn.preprocessing import normalize as sk_normalize
try:
    from sklearn.metrics import silhouette_score as _silhouette_score
    _HAS_SILHOUETTE = True
except ImportError:
    _HAS_SILHOUETTE = False


class TrainingDashboard:
    """
    Single persistent dashboard PNG updated every epoch.
    Keeps full history in memory and rewrites the figure each time.
    """

    def __init__(self, run_name: str, out_dir: Path):
        self.run_name = run_name
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.png_path  = self.out_dir / f"dashboard_{run_name}.png"
        self.json_path = self.out_dir / f"dashboard_{run_name}.json"

        # History
        self.epochs:        List[int]   = []
        self.train_total:   List[float] = []
        self.train_recon:   List[float] = []
        self.train_vq:      List[float] = []
        self.val_total:     List[float] = []
        self.val_recon:     List[float] = []
        self.val_vq:        List[float] = []
        self.compartment_r: List[float] = []
        self.classifier_acc: List[float] = []
        self.silhouette:     List[float] = []
        self.active_codes:  List[int]   = []
        self.tau_f:         List[float] = []

        # PCA snapshots — list of {epoch, embeddings [N,2], labels [N]}
        self.pca_snapshots: List[Dict] = []

        # Codebook usage snapshots — list of {epoch, usage [N_CODES]}
        self.codebook_snapshots: List[Dict] = []

        print(f"[dashboard] Output: {self.png_path}")

    def update(
        self,
        epoch:          int,
        train_metrics:  Dict[str, float],
        val_metrics:    Dict[str, float],
        model           = None,
        device          = None,
        fp_embeddings:  Optional[np.ndarray] = None,  # [N, FP_DIM]
        fp_labels:      Optional[List[str]]  = None,  # [N] cell type names
        codebook_usage: Optional[np.ndarray] = None,  # [N_CODES] usage counts
        tau_f:          float = 0.0,
    ):
        """Update history and redraw dashboard. Called once per epoch."""
        self.epochs.append(epoch)
        self.train_total.append(train_metrics.get("total", 0))
        self.train_recon.append(train_metrics.get("recon", 0))
        self.train_vq.append(train_metrics.get("vq", 0))
        self.val_total.append(val_metrics.get("total", 0))
        self.val_recon.append(val_metrics.get("recon", 0))
        self.val_vq.append(val_metrics.get("vq", 0))
        self.compartment_r.append(val_metrics.get("compartment_r", 0))
        self.classifier_acc.append(val_metrics.get("classifier_acc", 0))
        self.silhouette.append(val_metrics.get("silhouette", 0.0))
        self.active_codes.append(val_metrics.get("active_codes",
                                  train_metrics.get("active_codes", 0)))
        self.tau_f.append(tau_f)

        if fp_embeddings is not None and fp_labels is not None:
            umap_2d = _umap_2d(fp_embeddings)
            self.pca_snapshots.append({
                "epoch":      epoch,
                "embeddings": umap_2d.tolist(),
                "labels":     fp_labels,
            })

        if codebook_usage is not None:
            self.codebook_snapshots.append({
                "epoch": epoch,
                "usage": codebook_usage.tolist(),
            })

        self._draw()
        self._save_json()

    def _draw(self):
        # Publication-quality figure
        fig = plt.figure(figsize=(20, 16), facecolor="white")
        fig.suptitle(
            f"MQ-VAE Training Dashboard — {self.run_name}  (Epoch {self.epochs[-1]})",
            fontsize=18, color="#2c3e50", fontweight="bold", y=0.97,
            family="sans-serif",
        )

        gs = gridspec.GridSpec(
            4, 3,
            figure=fig,
            hspace=0.45, wspace=0.38,
            left=0.07, right=0.96, top=0.93, bottom=0.05,
            height_ratios=[1, 1, 1.3, 1],
        )

        ep = self.epochs

        # ── Row 0, Col 0: Total loss ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        _style_pub(ax)
        ax.plot(ep, self.train_total, color="#3498db", lw=2.5, label="Train", marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax.plot(ep, self.val_total,   color="#e74c3c", lw=2.5, label="Validation", ls="--", marker='s', markersize=4, markevery=max(1, len(ep)//10))
        ax.set_title("Total Loss\nShould decrease steadily; val tracks train", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
        _annotate_last(ax, ep, self.val_total, "#e74c3c")

        # ── Row 0, Col 1: Recon loss ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        _style_pub(ax)
        ax.plot(ep, self.train_recon, color="#3498db", lw=2.5, label="Train", marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax.plot(ep, self.val_recon,   color="#e74c3c", lw=2.5, label="Validation", ls="--", marker='s', markersize=4, markevery=max(1, len(ep)//10))
        ax.set_title("Reconstruction Loss\nMSE between input and decoded contact map", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MSE Loss", fontsize=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
        _annotate_last(ax, ep, self.val_recon, "#e74c3c")

        # ── Row 0, Col 2: VQ loss ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        _style_pub(ax)
        ax.plot(ep, self.train_vq, color="#3498db", lw=2.5, label="Train", marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax.plot(ep, self.val_vq,   color="#e74c3c", lw=2.5, label="Validation", ls="--", marker='s', markersize=4, markevery=max(1, len(ep)//10))
        ax.set_title("VQ Commitment Loss\nCodebook learning; stabilizes after warmup", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Commitment Loss", fontsize=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
        _annotate_last(ax, ep, self.val_vq, "#e74c3c")

        # ── Row 1, Col 0: Active codes + tau ─────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        _style_pub(ax)
        ax2 = ax.twinx()
        ax.plot(ep, self.active_codes, color="#f39c12", lw=2.5, label="Active Codes", marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax2.plot(ep, self.tau_f, color="#7f8c8d", lw=2, ls="--", label="Temperature τ", marker='s', markersize=3, markevery=max(1, len(ep)//10))
        ax.set_title("Codebook Utilization\nMore active codes = better diversity", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Active Codes", color="#f39c12", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Gumbel Temperature τ", color="#7f8c8d", fontsize=10, fontweight='bold')
        _style_pub(ax2)
        ax2.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=9)
        ax.tick_params(axis="y", labelcolor="#f39c12", labelsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=9, frameon=True, fancybox=True, shadow=True, loc='upper right')

        # ── Row 1, Col 1: Classifier Accuracy ────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        _style_pub(ax)
        ax.plot(ep, self.classifier_acc, color="#e67e22", lw=2.5, marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax.axhline(0.5, color="#95a5a6", ls="--", lw=1.5, label="Target ≥ 0.5", alpha=0.7)
        ax.axhline(0.9, color="#27ae60", ls="--", lw=1.5, label="Target ≥ 0.9", alpha=0.7)
        ax.set_title("Cell-Type Classifier Accuracy\nHigher = better cell-type separation", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
        _annotate_last(ax, ep, self.classifier_acc, "#e67e22")

        # ── Row 1, Col 2: Active codes + tau ─────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        _style_pub(ax)
        ax2 = ax.twinx()
        ax.plot(ep, self.active_codes, color="#f39c12", lw=2.5, label="Active Codes", marker='o', markersize=4, markevery=max(1, len(ep)//10))
        ax2.plot(ep, self.tau_f, color="#7f8c8d", lw=2, ls="--", label="Temperature τ", marker='s', markersize=3, markevery=max(1, len(ep)//10))
        ax.set_title("Codebook Utilization\nMore active codes = better diversity", fontsize=11, color="#2c3e50", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Active Codes", color="#f39c12", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Gumbel Temperature τ", color="#7f8c8d", fontsize=10, fontweight='bold')
        _style_pub(ax2)
        ax2.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=9)
        ax.tick_params(axis="y", labelcolor="#f39c12", labelsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=9, frameon=True, fancybox=True, shadow=True, loc='upper right')

        # ── Row 2, Col 0-2: UMAP of fingerprints (full width) ──────────────────
        ax = fig.add_subplot(gs[2, :])
        _style_pub(ax)
        sil_str = f"  |  Silhouette={self.silhouette[-1]:.3f}" if self.silhouette else ""
        ax.set_title(
            f"Fingerprint Embedding (UMAP, window-level, cosine metric){sil_str}",
            fontsize=11, color="#2c3e50", pad=6,
        )
        if self.pca_snapshots:
            snap   = self.pca_snapshots[-1]
            emb    = np.array(snap["embeddings"])
            labels = snap["labels"]
            unique_labels = sorted(set(labels))
            palette = _palette_pub(len(unique_labels))
            label_to_color = {l: palette[i] for i, l in enumerate(unique_labels)}

            for lbl in unique_labels:
                mask = np.array([l == lbl for l in labels])
                pts  = emb[mask]
                if len(pts):
                    ax.scatter(pts[:, 0], pts[:, 1],
                               c=[label_to_color[lbl]], s=60, alpha=0.72,
                               label=lbl, edgecolors='white', linewidths=0.8)

            margin = 0.08
            x_range = emb[:, 0].max() - emb[:, 0].min()
            y_range = emb[:, 1].max() - emb[:, 1].min()
            if x_range > 0 and y_range > 0:
                ax.set_xlim(emb[:, 0].min() - margin * x_range, emb[:, 0].max() + margin * x_range)
                ax.set_ylim(emb[:, 1].min() - margin * y_range, emb[:, 1].max() + margin * y_range)

            # Legend outside the axes — right side, never overlaps clusters
            ax.legend(
                fontsize=9, frameon=True, fancybox=True,
                markerscale=1.3, ncol=1,
                loc='center left', bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0,
            )
            ax.set_xlabel("UMAP 1", fontsize=10, fontweight='bold')
            ax.set_ylabel("UMAP 2", fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No UMAP data yet\n(fingerprints will appear after first epoch)",
                    ha="center", va="center", color="#95a5a6", fontsize=12,
                    transform=ax.transAxes, style='italic')

        # ── Row 3, Col 0: Codebook histogram ──────────────────────────────────
        ax = fig.add_subplot(gs[3, 0])
        _style_pub(ax)
        if self.codebook_snapshots:
            usage = np.array(self.codebook_snapshots[-1]["usage"])
            n_dead = int((usage == 0).sum())
            n_active = len(usage) - n_dead
            ax.bar(np.arange(len(usage)), np.sort(usage)[::-1],
                   color="#27ae60", width=1.0, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_title(f"Codebook Usage Distribution\n{n_active}/{len(usage)} codes active ({n_dead} dead)",
                         fontsize=11, color="#2c3e50", pad=10)
            ax.set_xlabel("Code Rank (sorted by usage)", fontsize=10)
            ax.set_ylabel("Usage Count", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No codebook data yet\n(will appear after first epoch)",
                    ha="center", va="center", color="#95a5a6", fontsize=11,
                    transform=ax.transAxes, style='italic')
            ax.set_title("Codebook Usage Distribution", fontsize=11, color="#2c3e50", pad=10)

        # ── Row 3, Col 1-2: Training summary stats ────────────────────────────
        ax = fig.add_subplot(gs[3, 1:])
        ax.axis('off')
        if ep:
            sil_val = self.silhouette[-1] if self.silhouette else 0.0
            summary_text = (
                f"Training Summary (Epoch {ep[-1]})\n\n"
                f"• Total Loss:        {self.val_total[-1]:.4f}  (train: {self.train_total[-1]:.4f})\n"
                f"• Reconstruction:    {self.val_recon[-1]:.4f}\n"
                f"• VQ Commitment:     {self.val_vq[-1]:.4f}\n"
                f"• Cell Classifier:   {self.classifier_acc[-1]:.3f}\n"
                f"• Silhouette (cos):  {sil_val:.3f}\n"
                f"• Active Codes:      {self.active_codes[-1]}/{512 if self.codebook_snapshots else '?'}\n"
                f"• Temperature τ:     {self.tau_f[-1]:.3f}\n\n"
                f"Best Validation Loss: {min(self.val_total):.4f} (epoch {self.val_total.index(min(self.val_total))})\n"
            )
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, edgecolor='#34495e', linewidth=2),
                    color='#2c3e50')

        plt.savefig(self.png_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor='none')
        plt.close(fig)

    def _save_json(self):
        data = {
            "run_name":       self.run_name,
            "epochs":         self.epochs,
            "train_total":    self.train_total,
            "val_total":      self.val_total,
            "train_recon":    self.train_recon,
            "val_recon":      self.val_recon,
            "train_vq":       self.train_vq,
            "val_vq":         self.val_vq,
            "classifier_acc": self.classifier_acc,
            "silhouette":     self.silhouette,
            "compartment_r":  self.compartment_r,
            "active_codes":   self.active_codes,
            "tau_f":          self.tau_f,
        }
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)


# ── Helper functions ──────────────────────────────────────────────────────────

def _style_pub(ax):
    """Publication-quality axis styling."""
    ax.set_facecolor("#f8f9fa")
    ax.tick_params(colors="#2c3e50", labelsize=9, width=1.5)
    ax.xaxis.label.set_color("#2c3e50")
    ax.yaxis.label.set_color("#2c3e50")
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color="#bdc3c7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#7f8c8d")
        spine.set_linewidth(1.5)


def _annotate_last(ax, epochs, values, color):
    """Annotate the last value on a plot."""
    if epochs and values:
        ax.annotate(
            f"{values[-1]:.4f}",
            xy=(epochs[-1], values[-1]),
            color=color, fontsize=9, fontweight='bold',
            xytext=(5, 5), textcoords="offset points",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, alpha=0.8, linewidth=1.5),
        )


def _palette_pub(n: int):
    """Publication-quality color palette."""
    if n <= 1:
        return ["#3498db"]
    # Use distinct, colorblind-friendly colors
    base_colors = [
        "#e74c3c",  # red
        "#3498db",  # blue
        "#2ecc71",  # green
        "#f39c12",  # orange
        "#9b59b6",  # purple
        "#1abc9c",  # teal
        "#e67e22",  # dark orange
        "#34495e",  # dark gray
        "#16a085",  # dark teal
        "#c0392b",  # dark red
        "#2980b9",  # dark blue
        "#27ae60",  # dark green
        "#8e44ad",  # dark purple
        "#d35400",  # burnt orange
        "#2c3e50",  # midnight blue
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    # Fall back to colormap for many categories
    cmap = plt.cm.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Fast PCA to 2D using SVD."""
    X = embeddings - embeddings.mean(axis=0)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((X.shape[0], 2))
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[:2].T
    except Exception:
        return np.zeros((X.shape[0], 2))


def _umap_2d(embeddings: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.05) -> np.ndarray:
    """UMAP to 2D with cosine metric — aligns with retrieval geometry."""
    try:
        import umap
        n_pts = embeddings.shape[0]
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, max(2, n_pts - 1)),
            min_dist=min_dist,
            random_state=42,
            metric='cosine',
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        print("[dashboard] UMAP not installed, falling back to PCA")
        return _pca_2d(embeddings)
    except Exception as e:
        print(f"[dashboard] UMAP failed: {e}, falling back to PCA")
        return _pca_2d(embeddings)


def compute_silhouette(embeddings: np.ndarray, labels: List[str]) -> float:
    """Silhouette score on L2-normalised embeddings with cosine distance."""
    if not _HAS_SILHOUETTE or len(embeddings) < 4:
        return 0.0
    unique = set(labels)
    if len(unique) < 2:
        return 0.0
    try:
        fps_norm = sk_normalize(embeddings, norm='l2')
        label_arr = np.array(labels)
        return float(_silhouette_score(fps_norm, label_arr, metric='cosine'))
    except Exception:
        return 0.0


def collect_fingerprints(model, loader, device, max_samples: int = 2000):
    """
    Collect window-level fingerprints + cell line labels from a dataloader.
    Returns ALL windows (not averaged per sample) — one point per tile.
    Labels are sample_id strings (e.g. "K562", "IMR-90") for color coding.
    Returns (embeddings [N, FP_DIM], labels [N]).
    """
    model.eval()
    fps    = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            contact  = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            fp = model.encode_fingerprint(contact, assay_id).cpu().float().numpy()
            fps.append(fp)
            batch_labels = batch.get("sample_id", ["unknown"] * len(fp))
            labels.extend(batch_labels)
            if sum(len(f) for f in fps) >= max_samples:
                break
    if not fps:
        return np.zeros((0, 32)), []
    return np.concatenate(fps, axis=0)[:max_samples], labels[:max_samples]


def collect_codebook_usage(model) -> np.ndarray:
    """Return usage counts per codebook entry."""
    try:
        usage = model.vq.usage_count.cpu().numpy().copy()  # correct buffer name
        return usage
    except AttributeError:
        return np.array([])
