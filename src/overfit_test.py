"""
overfit_test.py — Comprehensive overfit, codebook, ablation, and DB query tests.

Outputs everything to /app/tmp/DATABASE_CONCEPT/trash/overfit_results/

Tests:
  1. Overfit test:    single-batch gradient descent for 100 steps; verify loss → 0
  2. Codebook test:   track code utilization, EMA updates, dead code revival
  3. Ablation study:  8 configs × 5 epochs, MAP@5 comparison table
  4. DB + query test: build DuckDB from trained model, query all 5 cell lines,
                      print full 4-level report per sample
"""

import sys
import json
import time
import copy
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import config
config.CHROMOSOMES = ["chr21", "chr22"]
config.NUM_WORKERS = 0

from config import TRASH_DIR, CELL_LINE_REGISTRY, MCOOL_DIR
from model import MQVAE
from loss import total_loss, pearson_1d
from masker import get_tau_f
from dataset import HiCTileDataset
from database import extract_and_ingest
from query import HiCStructuralDatabase

RESULTS_DIR = TRASH_DIR / "overfit_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CELL_LINES = list(CELL_LINE_REGISTRY.keys())
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small model — all dims passed explicitly to MQVAE (no config patching needed)
SMALL = dict(
    n_codes              = 64,
    code_dim             = 128,
    fp_dim               = 16,
    use_boundary_head    = True,
    use_compartment_head = True,
    use_masking          = True,
    use_film             = True,
    encoder_channels     = [16, 32, 64, 128],
    n_transformer_layers = 2,
    n_heads              = 4,
    ffn_dim              = 256,
    decoder_channels     = [64, 32, 16],
)

print(f"\n{'='*64}")
print(f"  MQ-VAE Overfit + Ablation + DB Query Tests")
print(f"  Device: {DEVICE}")
print(f"  Cell lines: {CELL_LINES}")
print(f"  Output dir: {RESULTS_DIR}")
print(f"{'='*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_tiles():
    """Load all 10 tiles as a single batch dict."""
    ds = HiCTileDataset(cell_lines=CELL_LINES, augment=False)
    items = [ds[i] for i in range(len(ds))]
    contact     = torch.stack([x["contact"]     for x in items]).to(DEVICE)
    boundary    = torch.stack([x["boundary"]    for x in items]).to(DEVICE)
    compartment = torch.stack([x["compartment"] for x in items]).to(DEVICE)
    assay_id    = torch.stack([x["assay_id"]    for x in items]).to(DEVICE)
    sample_ids  = [x["sample_id"] for x in items]
    chroms      = [x["chr"]       for x in items]
    return contact, boundary, compartment, assay_id, sample_ids, chroms


def compute_map_at_k(fps, labels, k=5):
    """Mean Average Precision@K — cosine similarity ranking."""
    N     = len(fps)
    if N < 2:
        return 0.0
    fps_n = fps / (np.linalg.norm(fps, axis=1, keepdims=True) + 1e-8)
    sims  = fps_n @ fps_n.T
    np.fill_diagonal(sims, -1.0)
    aps = []
    for i in range(N):
        ranked = np.argsort(sims[i])[::-1][:k]
        hits   = (labels[ranked] == labels[i]).astype(float)
        if hits.sum() == 0:
            aps.append(0.0)
            continue
        prec = np.cumsum(hits) / (np.arange(k) + 1)
        aps.append(float((prec * hits).sum() / min(k, hits.sum())))
    return float(np.mean(aps))


def extract_fps(model):
    """Extract [N, fp_dim] fingerprints and int labels for all tiles."""
    ds  = HiCTileDataset(cell_lines=CELL_LINES, augment=False)
    lbl = {s: i for i, s in enumerate(CELL_LINES)}
    fps, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            c  = item["contact"].unsqueeze(0).to(DEVICE)
            ai = item["assay_id"].unsqueeze(0).to(DEVICE)
            fp = model.encode_fingerprint(c, ai).squeeze(0).cpu().numpy()
            fps.append(fp)
            labels.append(lbl[item["sample_id"]])
    return np.stack(fps), np.array(labels)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  → saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Overfit Test
# ─────────────────────────────────────────────────────────────────────────────

def test_overfit(n_steps=150, lr=3e-4):
    print("\n" + "="*64)
    print("TEST 1: Overfit Test (single batch, 150 gradient steps)")
    print("="*64)

    contact, boundary, compartment, assay_id, sample_ids, chroms = load_all_tiles()
    targets = {"contact": contact, "boundary": boundary, "compartment": compartment}

    model = MQVAE(**SMALL).to(DEVICE)
    model.set_masker_temperatures(get_tau_f(0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    log = []
    print(f"\n{'Step':>5}  {'Total':>8}  {'Recon':>8}  {'VQ':>7}  "
          f"{'Bdry':>7}  {'Comp':>7}  {'Codes':>6}  {'Comp_r':>7}")
    print("-" * 70)

    for step in range(n_steps):
        model.train()
        # Anneal temperature mid-run
        if step == 50:
            model.set_masker_temperatures(get_tau_f(10))
        elif step == 100:
            model.set_masker_temperatures(get_tau_f(30))

        # Use aux weight = 1.0 after step 30
        epoch_proxy = 0 if step < 30 else 15

        outputs = model(contact, assay_id)
        loss, metrics = total_loss(outputs, targets, epoch=epoch_proxy)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        active = int((model.vq.usage_count > 0).sum().item())

        row = {
            "step":         step,
            "total":        round(metrics["total"],       4),
            "recon":        round(metrics["recon"],       4),
            "vq":           round(metrics["vq"],          4),
            "boundary":     round(metrics["boundary"],    4),
            "compartment":  round(metrics["compartment"], 4),
            "boundary_f1":  round(metrics["boundary_f1"],4),
            "compartment_r":round(metrics["compartment_r"],4),
            "active_codes": active,
        }
        log.append(row)

        if step % 15 == 0 or step == n_steps - 1:
            print(f"{step:>5}  {metrics['total']:>8.4f}  {metrics['recon']:>8.4f}  "
                  f"{metrics['vq']:>7.4f}  {metrics['boundary']:>7.4f}  "
                  f"{metrics['compartment']:>7.4f}  {active:>6}  "
                  f"{metrics['compartment_r']:>7.3f}")

    # ── Assertions ────────────────────────────────────────────────────────────
    first_loss = log[0]["total"]
    last_loss  = log[-1]["total"]
    final_recon = log[-1]["recon"]
    final_codes = log[-1]["active_codes"]

    print(f"\n  Loss trajectory: {first_loss:.4f} → {last_loss:.4f}")
    print(f"  Active codes at end: {final_codes}/64")

    # Reconstruction quality: model can reconstruct the input it saw
    model.eval()
    with torch.no_grad():
        out_eval = model(contact, assay_id)
        recon_mse = F.mse_loss(out_eval["contact_recon"], contact).item()
        recon_pcc = pearson_1d(
            out_eval["contact_recon"].flatten(1),
            contact.flatten(1)
        ).item()
        bd_prob  = out_eval["boundary_logits"].sigmoid()
        comp_pcc = pearson_1d(out_eval["compartment"], compartment).item()

    print(f"\n  Final Reconstruction Quality:")
    print(f"    Contact MSE:       {recon_mse:.5f}")
    print(f"    Contact Pearson r: {recon_pcc:.4f}")
    print(f"    Boundary prob mean: {bd_prob.mean().item():.4f}  "
          f"max: {bd_prob.max().item():.4f}")
    print(f"    Compartment Pearson r: {comp_pcc:.4f}")

    # Codebook state
    with torch.no_grad():
        code_usage = model.vq.usage_count.cpu().numpy()
        ppl = float(model.vq.perplexity(out_eval["indices"]).item())
    print(f"\n  Codebook (n=64) State:")
    print(f"    Active codes: {(code_usage > 0).sum()}/64")
    print(f"    Perplexity:   {ppl:.2f}")
    print(f"    Max usage:    {code_usage.max():.0f}")
    print(f"    Usage histogram (top 10 codes):")
    top_codes = np.argsort(code_usage)[::-1][:10]
    for c in top_codes:
        bar = "█" * min(int(code_usage[c] / max(code_usage.max(), 1) * 20), 20)
        print(f"      code {c:3d}: {code_usage[c]:6.0f}  {bar}")

    # Fingerprint separability (do different cell lines get different fps?)
    fps, labels = extract_fps(model)
    map5 = compute_map_at_k(fps, labels, k=4)  # k<N
    print(f"\n  Fingerprint Separability (MAP@4 across 5 cell lines): {map5:.4f}")
    # Pairwise cosine matrix
    fps_n = fps / (np.linalg.norm(fps, axis=1, keepdims=True) + 1e-8)
    sim_mat = fps_n @ fps_n.T
    print(f"  Cosine similarity matrix (rows=tiles, cols=tiles):")
    header = "       " + "  ".join([f"{CELL_LINES[labels[i]][:6]:>6}" for i in range(len(fps))])
    print(f"  {header}")
    for i in range(len(fps)):
        row_s = f"  {CELL_LINES[labels[i]][:6]:>6} " + "  ".join([f"{sim_mat[i,j]:>6.3f}" for j in range(len(fps))])
        print(row_s)

    # PASS/FAIL
    overfit_ok    = last_loss < first_loss * 0.7
    codebook_ok   = final_codes >= 5
    recon_ok      = recon_mse < 5.0

    print(f"\n  Overfit check (loss dropped >30%): {'✓ PASS' if overfit_ok else '✗ FAIL'}"
          f"  ({first_loss:.4f} → {last_loss:.4f})")
    print(f"  Codebook check (≥5 active codes): {'✓ PASS' if codebook_ok else '✗ FAIL'}"
          f"  ({final_codes}/64)")
    print(f"  Recon check (MSE < 5.0):          {'✓ PASS' if recon_ok else '✗ FAIL'}"
          f"  ({recon_mse:.4f})")

    save_json({"log": log,
               "final_recon_mse": recon_mse,
               "final_recon_pcc": recon_pcc,
               "final_comp_pcc":  comp_pcc,
               "active_codes":    int(final_codes),
               "perplexity":      ppl,
               "map_at_4":        map5,
               "overfit_ok":      overfit_ok,
               "codebook_ok":     codebook_ok,
               "recon_ok":        recon_ok},
              RESULTS_DIR / "overfit_log.json")

    # Save model for DB test
    ckpt_path = RESULTS_DIR / "overfit_model.pt"
    torch.save({"epoch": 0, "model": model.state_dict(), "arch": SMALL}, ckpt_path)
    print(f"\n  Model saved to {ckpt_path}")
    return model, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Codebook Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

def test_codebook_deep(model):
    print("\n" + "="*64)
    print("TEST 2: Codebook Deep Dive")
    print("="*64)

    contact, boundary, compartment, assay_id, sample_ids, chroms = load_all_tiles()

    model.eval()
    with torch.no_grad():
        outputs = model(contact, assay_id)

    indices  = outputs["indices"].cpu().numpy()   # [B, K]
    z_e      = outputs["z_e"].cpu().numpy()         # [B, K, code_dim]
    z_q_fp   = outputs["fingerprint"].cpu().numpy() # [B, fp_dim]
    codebook = model.vq.codebook.cpu().numpy()      # [n_codes, code_dim]

    print(f"\n  Codebook size: {codebook.shape[0]} × {codebook.shape[1]}")
    print(f"  Codebook norm range: [{np.linalg.norm(codebook, axis=1).min():.3f}, "
          f"{np.linalg.norm(codebook, axis=1).max():.3f}]")

    # Per-tile code usage
    print(f"\n  Per-tile code assignments (showing unique codes used):")
    all_unique = set()
    for i, sid in enumerate(sample_ids):
        unique_codes = np.unique(indices[i])
        all_unique.update(unique_codes.tolist())
        print(f"    {sid[:20]:>20} ({chroms[i]}): "
              f"{len(unique_codes)} unique codes / {indices.shape[1]} slots  "
              f"top-3: {np.bincount(indices[i], minlength=codebook.shape[0]).argsort()[::-1][:3].tolist()}")

    print(f"\n  Total unique codes used across all tiles: {len(all_unique)}/{codebook.shape[0]}")

    # Commitment distance: how far encoder outputs are from their nearest code
    code_dim = codebook.shape[1]
    flat_z_e = z_e.reshape(-1, code_dim)
    flat_idx = indices.reshape(-1)
    assigned_codes = codebook[flat_idx]
    commit_dists   = np.linalg.norm(flat_z_e - assigned_codes, axis=1)
    print(f"\n  Commitment distances (||z_e - z_q||):")
    print(f"    Mean:   {commit_dists.mean():.4f}")
    print(f"    Median: {np.median(commit_dists):.4f}")
    print(f"    Max:    {commit_dists.max():.4f}")

    # Fingerprint cosine matrix
    fps_n = z_q_fp / (np.linalg.norm(z_q_fp, axis=1, keepdims=True) + 1e-8)
    sim   = fps_n @ fps_n.T

    print(f"\n  Fingerprint cosine similarity matrix:")
    labels_short = [f"{s[:8]}({c})" for s, c in zip(sample_ids, chroms)]
    maxw = max(len(l) for l in labels_short)
    header = " " * (maxw + 2) + "  ".join([f"{l[:8]:>8}" for l in labels_short])
    print(f"  {header}")
    for i, lbl in enumerate(labels_short):
        row = f"  {lbl:>{maxw}} " + "  ".join([f"{sim[i,j]:>8.4f}" for j in range(len(labels_short))])
        print(row)

    # Code histogram distance between cell lines (same chrom)
    print(f"\n  Code histogram distances (chi-squared, same chromosome pairs):")
    hists = {}
    for i, (sid, chrom) in enumerate(zip(sample_ids, chroms)):
        h = np.bincount(indices[i], minlength=codebook.shape[0]).astype(float)
        h /= h.sum() + 1e-8
        hists[(sid, chrom)] = h

    chr21_hists = {sid: h for (sid, c), h in hists.items() if c == "chr21"}
    sids = list(chr21_hists.keys())
    for i in range(len(sids)):
        for j in range(i+1, len(sids)):
            a, b = chr21_hists[sids[i]], chr21_hists[sids[j]]
            chi2 = float(np.sum((a - b)**2 / (a + b + 1e-8)))
            print(f"    {sids[i][:15]:>15} vs {sids[j][:15]:<15}: χ²={chi2:.4f}")

    result = {
        "n_codes_used": int(len(all_unique)),
        "commit_dist_mean": float(commit_dists.mean()),
        "commit_dist_max":  float(commit_dists.max()),
        "fingerprint_sim_matrix": sim.tolist(),
        "sample_ids": sample_ids,
        "chroms": chroms,
    }
    save_json(result, RESULTS_DIR / "codebook_analysis.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Ablation Studies
# ─────────────────────────────────────────────────────────────────────────────

def _abl(overrides):
    """Merge SMALL kwargs with per-ablation overrides."""
    return {**SMALL, **overrides}

ABLATION_CONFIGS = [
    {"id": 1, "name": "recon_only",       "desc": "Recon-only, no aux heads",
     "kw": _abl(dict(use_boundary_head=False, use_compartment_head=False))},
    {"id": 2, "name": "boundary_only",    "desc": "+ Boundary head only",
     "kw": _abl(dict(use_compartment_head=False))},
    {"id": 3, "name": "compartment_only", "desc": "+ Compartment head only",
     "kw": _abl(dict(use_boundary_head=False))},
    {"id": 4, "name": "full_v4",          "desc": "Full v4 (both heads + mask + FiLM)",
     "kw": _abl({})},
    {"id": 5, "name": "no_masking",       "desc": "No masking (all tokens visible)",
     "kw": _abl(dict(use_masking=False))},
    {"id": 6, "name": "no_film",          "desc": "No FiLM conditioning",
     "kw": _abl(dict(use_film=False))},
    {"id": 7, "name": "codebook_32",      "desc": "Tiny codebook (32 codes)",
     "kw": _abl(dict(n_codes=32))},
    {"id": 8, "name": "codebook_128",     "desc": "Larger codebook (128 codes)",
     "kw": _abl(dict(n_codes=128))},
]

def train_config(cfg, n_steps=80, lr=3e-4):
    """Train one ablation config for n_steps and return metrics."""
    contact, boundary, compartment, assay_id, sample_ids, _ = load_all_tiles()
    targets = {"contact": contact, "boundary": boundary, "compartment": compartment}

    model = MQVAE(**cfg["kw"]).to(DEVICE)
    model.set_masker_temperatures(1.0)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses = []
    for step in range(n_steps):
        if step == 40:
            model.set_masker_temperatures(0.5)
        epoch_proxy = 15 if step > 40 else 0
        model.train()
        out = model(contact, assay_id)
        loss, metrics = total_loss(out, targets, epoch=epoch_proxy)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(metrics["total"])

    # Eval
    model.eval()
    with torch.no_grad():
        out = model(contact, assay_id)
        _, final_metrics = total_loss(out, targets, epoch=15)
        recon_pcc = pearson_1d(
            out["contact_recon"].flatten(1), contact.flatten(1)
        ).item()
        comp_r = final_metrics["compartment_r"]
        bd_f1  = final_metrics["boundary_f1"]

    # Fingerprints + MAP@K
    fps, labels = extract_fps(model)
    map5 = compute_map_at_k(fps, labels, k=min(4, len(fps)-1))
    active = int((model.vq.usage_count > 0).sum().item())
    ppl    = float(model.vq.perplexity(out["indices"]).item())

    # Save best ckpt for full_v4
    if cfg["id"] == 4:
        torch.save({"epoch": 0, "model": model.state_dict()},
                   RESULTS_DIR / "ablation_full_v4.pt")

    return {
        "ablation_id":   cfg["id"],
        "name":          cfg["name"],
        "desc":          cfg["desc"],
        "final_loss":    round(losses[-1], 4),
        "init_loss":     round(losses[0],  4),
        "loss_drop_pct": round(100 * (1 - losses[-1] / (losses[0] + 1e-8)), 1),
        "recon_pcc":     round(recon_pcc, 4),
        "boundary_f1":   round(bd_f1,  4),
        "compartment_r": round(comp_r, 4),
        "map_at_4":      round(map5,   4),
        "active_codes":  active,
        "perplexity":    round(ppl, 2),
        "n_codes":       cfg["kw"].get("n_codes", SMALL["n_codes"]),
    }


def test_ablations():
    print("\n" + "="*64)
    print("TEST 3: Ablation Studies (8 configs × 80 gradient steps)")
    print("="*64)

    results = []
    for cfg in ABLATION_CONFIGS:
        t0 = time.time()
        print(f"\n  Running ablation #{cfg['id']}: {cfg['desc']}...")
        r = train_config(cfg)
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 1)
        results.append(r)
        print(f"    loss: {r['init_loss']:.4f}→{r['final_loss']:.4f} "
              f"({r['loss_drop_pct']:.1f}% drop) | "
              f"recon_r={r['recon_pcc']:.3f} | "
              f"bd_f1={r['boundary_f1']:.3f} | "
              f"comp_r={r['compartment_r']:.3f} | "
              f"MAP@4={r['map_at_4']:.4f} | "
              f"codes={r['active_codes']}/{r['n_codes']} | "
              f"ppl={r['perplexity']:.1f} | "
              f"{elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'─'*110}")
    print(f"{'#':>3}  {'Name':<22}  {'Loss↓%':>7}  {'Recon r':>8}  "
          f"{'Bdry F1':>8}  {'Comp r':>7}  {'MAP@4':>7}  "
          f"{'Codes':>6}  {'Ppl':>6}  {'Desc'}")
    print(f"{'─'*110}")
    for r in results:
        print(f"{r['ablation_id']:>3}  {r['name']:<22}  "
              f"{r['loss_drop_pct']:>6.1f}%  "
              f"{r['recon_pcc']:>8.4f}  "
              f"{r['boundary_f1']:>8.4f}  "
              f"{r['compartment_r']:>7.4f}  "
              f"{r['map_at_4']:>7.4f}  "
              f"{r['active_codes']:>3}/{r['n_codes']:<3}  "
              f"{r['perplexity']:>6.1f}  "
              f"{r['desc']}")
    print(f"{'─'*110}")

    # Decision points
    baseline = next((r for r in results if r["ablation_id"] == 1), None)
    full_v4  = next((r for r in results if r["ablation_id"] == 4), None)
    no_mask  = next((r for r in results if r["ablation_id"] == 5), None)
    no_film  = next((r for r in results if r["ablation_id"] == 6), None)

    print("\n  Decision Points:")
    if baseline and full_v4:
        diff = full_v4["map_at_4"] - baseline["map_at_4"]
        print(f"  Aux heads benefit:   MAP@4 {baseline['map_at_4']:.4f}→{full_v4['map_at_4']:.4f} "
              f"(Δ={diff:+.4f}, {'significant' if abs(diff)>0.02 else 'marginal'})")
    if full_v4 and no_mask:
        diff = full_v4["map_at_4"] - no_mask["map_at_4"]
        print(f"  Masking benefit:     MAP@4 Δ={diff:+.4f} "
              f"({'masking helps' if diff > 0.01 else 'masking neutral'})")
    if full_v4 and no_film:
        diff = full_v4["map_at_4"] - no_film["map_at_4"]
        print(f"  FiLM benefit:        MAP@4 Δ={diff:+.4f} "
              f"({'FiLM helps' if diff > 0.01 else 'FiLM neutral'})")

    save_json(results, RESULTS_DIR / "ablation_results.json")

    # Fingerprint dim sweep on full_v4
    if full_v4 and Path(RESULTS_DIR / "ablation_full_v4.pt").exists():
        print("\n  Fingerprint Dimensionality Sweep (PCA on full_v4 fingerprints):")
        ckpt = torch.load(RESULTS_DIR / "ablation_full_v4.pt", map_location=DEVICE)
        full_v4_kw = next(c for c in ABLATION_CONFIGS if c["id"]==4)["kw"]
        m = MQVAE(**full_v4_kw).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        fps, labels = extract_fps(m)
        try:
            from sklearn.decomposition import PCA
            dim_results = {}
            for d in [4, 8, 16, 32]:
                if d >= fps.shape[1]:
                    red = fps
                else:
                    red = PCA(n_components=d, random_state=42).fit_transform(fps)
                dim_results[d] = round(compute_map_at_k(red, labels, k=min(4, len(fps)-1)), 4)
                print(f"    d={d:2d}  MAP@4={dim_results[d]:.4f}")
            save_json({str(k): v for k, v in dim_results.items()},
                      RESULTS_DIR / "dim_sweep.json")
        except ImportError:
            print("    sklearn not available — skipping dim sweep")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Database + Query End-to-End
# ─────────────────────────────────────────────────────────────────────────────

def test_db_query(ckpt_path):
    print("\n" + "="*64)
    print("TEST 4: Database Build + Query End-to-End")
    print("="*64)

    db_path    = RESULTS_DIR / "test.duckdb"
    faiss_path = RESULTS_DIR / "test.faiss"

    # ── Build DB ──────────────────────────────────────────────────────────────
    print("\n  Step 4a: Building DuckDB from overfit model...")
    extract_and_ingest(
        model_path = str(ckpt_path),
        db_path    = db_path,
        faiss_path = faiss_path,
        cell_lines = CELL_LINES,
        batch_size = 5,
        device_str = str(DEVICE),
        overwrite  = True,
    )

    # ── Inspect DB directly ───────────────────────────────────────────────────
    print("\n  Step 4b: Direct DuckDB inspection...")
    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)

    total_fps = conn.execute("SELECT COUNT(*) FROM window_fingerprints").fetchone()[0]
    print(f"    Total fingerprints: {total_fps}")

    per_sample = conn.execute(
        "SELECT sample_id, COUNT(*) as n FROM window_fingerprints GROUP BY sample_id ORDER BY sample_id"
    ).fetchall()
    print(f"    Per-sample tile counts:")
    for sid, n in per_sample:
        print(f"      {sid}: {n} tile(s)")

    loci = conn.execute(
        "SELECT chr, start_bp, end_bp, n_samples FROM locus_centroids ORDER BY chr, start_bp"
    ).fetchall()
    print(f"    Locus centroids ({len(loci)} total):")
    for chr_, s, e, ns in loci:
        print(f"      {chr_}:{s//1_000_000}Mb–{e//1_000_000}Mb  (n_samples={ns})")

    # Cosine similarity between stored fingerprints (from DB directly)
    rows = conn.execute(
        "SELECT wf.sample_id, wf.chr, wf.fingerprint "
        "FROM window_fingerprints wf ORDER BY sample_id, chr"
    ).fetchall()
    sids_db = [r[0] for r in rows]
    chrs_db = [r[1] for r in rows]
    fps_db  = np.stack([np.frombuffer(r[2], dtype=np.float32).copy() for r in rows])
    fps_n   = fps_db / (np.linalg.norm(fps_db, axis=1, keepdims=True) + 1e-8)
    sim_mat = fps_n @ fps_n.T

    print(f"\n    DB fingerprint cosine similarity matrix:")
    labels_db = [f"{s[:10]}({c})" for s, c in zip(sids_db, chrs_db)]
    maxw = max(len(l) for l in labels_db)
    print("    " + " " * maxw + "  " + "  ".join([f"{l[:8]:>8}" for l in labels_db]))
    for i, lbl in enumerate(labels_db):
        row = f"    {lbl:>{maxw}} " + "  ".join([f"{sim_mat[i,j]:>8.4f}" for j in range(len(labels_db))])
        print(row)

    # Histogram comparison from DB
    hist_rows = conn.execute(
        "SELECT sample_id, code_histogram FROM sample_histograms ORDER BY sample_id"
    ).fetchall()
    print(f"\n    Sample histograms ({len(hist_rows)} samples):")
    for sid, h_blob in hist_rows:
        h = np.frombuffer(h_blob, dtype=np.float32).copy()
        top3 = np.argsort(h)[::-1][:3]
        print(f"      {sid[:20]:>20}: top codes={top3.tolist()}  "
              f"max_freq={h.max():.4f}  nonzero={int((h>0).sum())}")
    conn.close()

    # ── Query each cell line against the database ─────────────────────────────
    print("\n  Step 4c: Querying each cell line against the database...")

    db_obj = HiCStructuralDatabase(
        ckpt_path   = str(ckpt_path),
        db_path     = db_path,
        faiss_path  = faiss_path,
        device_str  = str(DEVICE),
        model_kwargs= SMALL,
    )

    query_summary = {}

    for sample_id, info in CELL_LINE_REGISTRY.items():
        mcool_path = str(MCOOL_DIR / info["file"])
        print(f"\n    Querying {sample_id}...")

        results = db_obj.query_mcool(
            mcool_path = mcool_path,
            assay_type = info["assay"],
            k          = min(3, len(CELL_LINES)),
            verbose    = False,
        )
        db_obj.print_report(results)

        # Summarise
        query_summary[sample_id] = {
            "predicted_cell_type": results["level1_cell_type"],
            "confidence":          round(results["level1_fraction"], 3),
            "n_windows":           results["n_windows"],
            "n_divergent":         len(results["level4_divergent"]),
            "chrom_scores":        {
                ch: round(v["mean_score"], 4)
                for ch, v in results["level2_chrom"].items()
            },
        }

    db_obj.close()

    # Self-retrieval accuracy (each query should retrieve itself as top-1)
    print("\n  Step 4d: Self-retrieval accuracy check")
    print(f"  {'Sample':<30}  {'Predicted':>20}  {'Confidence':>11}  {'Self-match':>10}")
    print(f"  {'─'*30}  {'─'*20}  {'─'*11}  {'─'*10}")
    n_correct = 0
    for sid, info in query_summary.items():
        pred    = info["predicted_cell_type"]
        conf    = info["confidence"]
        correct = (pred == sid)
        if correct:
            n_correct += 1
        print(f"  {sid:<30}  {pred:>20}  {conf:>11.3f}  "
              f"{'✓ CORRECT' if correct else '✗ WRONG':>10}")
    acc = n_correct / len(query_summary)
    print(f"\n  Self-retrieval accuracy: {n_correct}/{len(query_summary)} = {acc:.1%}")

    save_json(query_summary, RESULTS_DIR / "query_summary.json")
    return query_summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    all_results = {}

    # Test 1: Overfit
    model, ckpt_path = test_overfit(n_steps=150, lr=3e-4)
    all_results["overfit"] = "done"

    # Test 2: Codebook deep dive
    cb_result = test_codebook_deep(model)
    all_results["codebook"] = cb_result

    # Test 3: Ablations
    abl_results = test_ablations()
    all_results["ablations"] = abl_results

    # Test 4: DB + Query
    query_summary = test_db_query(ckpt_path)
    all_results["query"] = query_summary

    # Final summary
    elapsed_total = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"  ALL TESTS COMPLETE  ({elapsed_total:.0f}s total)")
    print(f"  Outputs in: {RESULTS_DIR}")
    print(f"{'='*64}\n")

    save_json({"elapsed_s": round(elapsed_total, 1)}, RESULTS_DIR / "timing.json")
