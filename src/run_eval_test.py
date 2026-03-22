"""
run_eval_test.py — Self-contained evaluation using current MQVAE model.
Saves all results to /data/joshi/Generative_experiment/Chromatin-DB/trash/eval_test/
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from model import MQVAE
from dashboard import _umap_2d, _palette_pub, collect_fingerprints
import config

OUT_DIR = Path("/data/joshi/Generative_experiment/Chromatin-DB/trash/eval_test")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    print(f"[eval] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = MQVAE(
        n_codes             = arch.get("n_codes",             config.N_CODES),
        use_classifier_head = arch.get("use_classifier_head", True),
        n_cell_types        = arch.get("n_cell_types",        config.N_CELL_TYPES),
        use_masking         = arch.get("use_masking",         True),
        use_film            = arch.get("use_film",            True),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] Loaded epoch={ckpt.get('epoch','?')}  params={n_params:,}")
    return model, ckpt


def extract_fps(model, loader, device, max_samples=3000):
    model.eval()
    fps, labels = [], []
    with torch.no_grad():
        for batch in loader:
            contact  = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            fp = model.encode_fingerprint(contact, assay_id).cpu().float().numpy()
            fps.append(fp)
            # use cell_line_name if available, else sample_id, else cell_idx
            if "cell_line_name" in batch:
                lbl = batch["cell_line_name"]
            elif "sample_id" in batch:
                lbl = batch["sample_id"] if isinstance(batch["sample_id"], list) else [str(x) for x in batch["sample_id"]]
            else:
                lbl = [str(x.item()) for x in batch["cell_idx"]]
            labels.extend(lbl)
            if sum(len(f) for f in fps) >= max_samples:
                break
    fps = np.concatenate(fps, axis=0)[:max_samples]
    labels = labels[:max_samples]
    return fps, labels


def recall_at_k(sim, int_labels, k):
    n = len(int_labels)
    correct = 0
    for i in range(n):
        row = sim[i].copy(); row[i] = -2
        top_k = np.argsort(row)[-k:]
        if any(int_labels[j] == int_labels[i] for j in top_k):
            correct += 1
    return correct / n


def mean_ap(sim, int_labels):
    n = len(int_labels)
    aps = []
    for i in range(n):
        row = sim[i].copy(); row[i] = -2
        order = np.argsort(row)[::-1]
        same = (int_labels[order] == int_labels[i]).astype(float)
        n_pos = same.sum()
        if n_pos == 0: continue
        cumsum = np.cumsum(same)
        precisions = (cumsum / np.arange(1, n+1)) * same
        aps.append(precisions.sum() / n_pos)
    return float(np.mean(aps)) if aps else 0.0


def knn_acc(fps_norm, labels, k=5):
    arr = np.array(labels)
    if len(np.unique(arr)) < 2: return 0.0
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(fps_norm, arr)
    return float((knn.predict(fps_norm) == arr).mean())


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_umap(fps_norm, labels, sil, out_path):
    print("[eval] Computing UMAP...")
    emb = _umap_2d(fps_norm)
    unique = sorted(set(labels))
    palette = _palette_pub(len(unique))
    lbl2col = {l: palette[i] for i, l in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#f8f9fa")
    for lbl in unique:
        mask = np.array([l == lbl for l in labels])
        pts = emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[lbl2col[lbl]], s=30, alpha=0.7,
                   label=lbl, edgecolors="white", linewidths=0.4)
    ax.set_title(f"Fingerprint UMAP  |  Silhouette={sil:.3f}  |  {len(unique)} cell types",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return emb


def fig_retrieval(metrics, random_r5, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fingerprint Retrieval Quality", fontsize=14, fontweight="bold")

    # Recall@k bar
    ax = axes[0]
    ks = [1, 3, 5, 10, 20]
    vals = [metrics[f"recall@{k}"] for k in ks]
    bars = ax.bar([str(k) for k in ks], vals, color="#3498db", edgecolor="white")
    ax.axhline(random_r5, color="#e74c3c", ls="--", lw=1.5, label=f"Random@5={random_r5:.3f}")
    ax.axhline(0.7, color="#27ae60", ls=":", lw=1.2, label="Target 0.70")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8)
    ax.set_title("Recall@k"); ax.set_xlabel("k"); ax.set_ylabel("Recall")

    # Summary bar
    ax = axes[1]
    names = ["mAP", "kNN-5 Acc", "Recall@5"]
    vals2 = [metrics["mAP"], metrics["knn5_acc"], metrics["recall@5"]]
    colors = ["#9b59b6", "#e67e22", "#3498db"]
    bars2 = ax.bar(names, vals2, color=colors, edgecolor="white")
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Summary Metrics")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig_similarity_matrix(fps_norm, labels, out_path):
    unique = sorted(set(labels))
    n = len(unique)
    label_arr = np.array(labels)
    sim = cosine_similarity(fps_norm)
    mean_sim = np.zeros((n, n))
    for i, li in enumerate(unique):
        mi = label_arr == li
        for j, lj in enumerate(unique):
            mj = label_arr == lj
            block = sim[np.ix_(mi, mj)]
            mean_sim[i, j] = block.mean() if block.size > 0 else 0.0

    fig, ax = plt.subplots(figsize=(max(10, n//2), max(8, n//2)))
    im = ax.imshow(mean_sim, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.5, label="Mean cosine similarity")
    ax.set_xticks(range(n)); ax.set_xticklabels(unique, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(unique, fontsize=7)
    ax.set_title("Cell-Type Mean Cosine Similarity Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig_per_celltype_silhouette(sil_samples, labels, out_path):
    unique = sorted(set(labels))
    label_arr = np.array(labels)
    per_ct = {lbl: float(sil_samples[label_arr == lbl].mean()) for lbl in unique}

    fig, ax = plt.subplots(figsize=(10, max(6, len(unique)//2)))
    names = list(per_ct.keys())
    vals = list(per_ct.values())
    colors = ["#27ae60" if v > 0.3 else "#f39c12" if v > 0 else "#e74c3c" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="#2c3e50", lw=0.8)
    ax.axvline(0.3, color="#27ae60", ls="--", lw=1.2, alpha=0.7, label="Target 0.3")
    ax.set_title("Per Cell-Type Silhouette Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette score")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return per_ct


def fig_intra_inter(fps_norm, labels, out_path):
    sim = cosine_similarity(fps_norm)
    label_arr = np.array(labels)
    intra, inter = [], []
    for lbl in set(labels):
        mask = label_arr == lbl
        intra.extend(sim[np.ix_(mask, mask)].flatten())
        inter.extend(sim[np.ix_(mask, ~mask)].flatten())
    intra_mu = np.mean(intra); inter_mu = np.mean(inter)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(intra, bins=bins, density=True, alpha=0.65, color="#27ae60",
            label=f"Intra-class μ={intra_mu:.3f}")
    ax.hist(inter, bins=bins, density=True, alpha=0.65, color="#e74c3c",
            label=f"Inter-class μ={inter_mu:.3f}")
    ax.legend(fontsize=9)
    ax.set_title(f"Similarity Distribution  |  Gap={intra_mu-inter_mu:.3f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return intra_mu, inter_mu


def fig_codebook(model, out_path):
    cb = model.vq.codebook.detach().cpu().float().numpy()
    usage = model.vq.ema_count.detach().cpu().float().numpy()
    usage_norm = usage / (usage.sum() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    ax.bar(range(len(usage_norm)), np.sort(usage_norm)[::-1], color="#3498db", width=1.0)
    ax.set_title(f"Codebook EMA Usage (sorted)  |  {(usage > 0.1).sum()} active codes",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Code rank"); ax.set_ylabel("EMA count (normalized)")

    ax = axes[1]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    cb2d = pca.fit_transform(cb)
    sc = ax.scatter(cb2d[:, 0], cb2d[:, 1], c=usage_norm, cmap="viridis", s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Usage (normalized)")
    ax.set_title("Codebook PCA (colored by usage)", fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eval(ckpt_path: str, out_dir: Path, processed_dir: str, max_samples: int = 3000):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(ckpt_path, device)

    print("[eval] Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        processed_dir=Path(processed_dir),
        batch_size=64,
        num_workers=8,
    )

    print("[eval] Extracting fingerprints (val)...")
    fps_val, lbl_val = extract_fps(model, val_loader, device, max_samples // 2)
    print("[eval] Extracting fingerprints (train)...")
    fps_trn, lbl_trn = extract_fps(model, train_loader, device, max_samples // 2)

    fps_all   = np.concatenate([fps_val, fps_trn], axis=0)
    lbl_all   = lbl_val + lbl_trn
    fps_norm  = sk_normalize(fps_all, norm="l2")
    label_arr = np.array(lbl_all)
    unique    = sorted(set(lbl_all))
    int_lbl   = np.array([unique.index(l) for l in lbl_all])

    print(f"[eval] Total: {len(fps_all)} fingerprints  |  {len(unique)} cell types")

    print("[eval] Computing similarity matrix...")
    sim = cosine_similarity(fps_norm)

    metrics = {}
    for k in [1, 3, 5, 10, 20]:
        metrics[f"recall@{k}"] = recall_at_k(sim, int_lbl, k)
        print(f"  Recall@{k:2d} = {metrics[f'recall@{k}']:.4f}")

    metrics["mAP"]      = mean_ap(sim, int_lbl)
    metrics["knn5_acc"] = knn_acc(fps_norm, lbl_all, k=5)
    print(f"  mAP       = {metrics['mAP']:.4f}")
    print(f"  kNN-5     = {metrics['knn5_acc']:.4f}")

    sil_global  = float(silhouette_score(fps_norm, label_arr, metric="cosine"))
    sil_samples = silhouette_samples(fps_norm, label_arr, metric="cosine")
    metrics["silhouette"] = sil_global
    print(f"  Silhouette= {sil_global:.4f}")

    n_classes = len(unique)
    random_r5 = 1 - (1 - 1/n_classes) ** 5
    metrics["random_recall5"] = random_r5

    r5 = metrics["recall@5"]
    print("\n" + "=" * 55)
    print("  FINGERPRINT QUALITY VERDICT")
    print("=" * 55)
    if   r5 >= 0.8:  verdict = "EXCELLENT — publication ready"
    elif r5 >= 0.65: verdict = "GOOD — suitable for retrieval"
    elif r5 >= 0.45: verdict = "FAIR — usable but needs improvement"
    else:            verdict = "POOR — needs more training"
    print(f"  Recall@5   = {r5:.3f}  ({verdict})")
    print(f"  mAP        = {metrics['mAP']:.3f}")
    print(f"  kNN-5 Acc  = {metrics['knn5_acc']:.3f}")
    print(f"  Silhouette = {sil_global:.3f}")
    print(f"  Random R@5 = {random_r5:.3f}  (baseline)")
    print("=" * 55)

    print("\n[eval] Generating figures...")
    emb = fig_umap(fps_norm, lbl_all, sil_global, out_dir / "fig1_umap.png")
    fig_retrieval(metrics, random_r5,              out_dir / "fig2_retrieval.png")
    fig_similarity_matrix(fps_norm, lbl_all,       out_dir / "fig3_sim_matrix.png")
    fig_per_celltype_silhouette(sil_samples, lbl_all, out_dir / "fig4_silhouette.png")
    intra_mu, inter_mu = fig_intra_inter(fps_norm, lbl_all, out_dir / "fig5_intra_inter.png")
    fig_codebook(model,                            out_dir / "fig6_codebook.png")

    metrics["intra_sim_mean"]  = intra_mu
    metrics["inter_sim_mean"]  = inter_mu
    metrics["separation_gap"]  = intra_mu - inter_mu
    metrics["n_windows"]       = len(fps_all)
    metrics["n_cell_types"]    = n_classes
    metrics["checkpoint"]      = str(ckpt_path)
    metrics["epoch"]           = ckpt.get("epoch", "?")

    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({k: (v if not isinstance(v, np.floating) else float(v)) for k, v in metrics.items()}, f, indent=2)
    print(f"\n[eval] JSON  → {results_path}")
    print(f"[eval] Plots → {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--processed_dir",  type=str,
                   default="/data/joshi/Generative_experiment/Chromatin-CLI/data/processed")
    p.add_argument("--out_dir",        type=str,
                   default="/data/joshi/Generative_experiment/Chromatin-DB/trash/eval_test")
    p.add_argument("--max_samples",    type=int, default=3000)
    args = p.parse_args()
    run_eval(args.checkpoint, Path(args.out_dir), args.processed_dir, args.max_samples)
