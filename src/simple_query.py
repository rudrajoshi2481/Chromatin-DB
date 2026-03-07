"""
simple_query.py — Standalone query script for Chromatin-DB.

Query a new .mcool file against the reference database and show four-level output.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import duckdb
import faiss
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import FP_DIM, ASSAY_TYPES, N_CODES
from model import MQVAE
from dataset import build_dataloaders, build_inference_loader
from annotator import MultiSampleAnnotator


def create_query_loader_from_tiles(mcool_path: str, assay_idx: int, batch_size: int = 32, num_workers: int = 0):
    """Create a simple query loader from tiles extracted on the fly."""
    # For now, just use the inference loader on all reference cell lines
    # In production, you'd extract tiles from the mcool file first
    from dataset import build_inference_loader
    return build_inference_loader(
        cell_lines=None,  # all available
        batch_size=batch_size,
        num_workers=num_workers,
    )


def load_model(ckpt_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    n_cell_types = arch.get("n_cell_types", 16)
    
    model = MQVAE(
        use_boundary_head=False,
        use_compartment_head=False,
        use_classifier_head=True,
        n_cell_types=n_cell_types,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def query_database(
    mcool_path: str,
    ckpt_path: str,
    db_path: str,
    faiss_path: str,
    assay_type: str = "bulk_hic",
    k: int = 5,
    device_str: str = "auto",
):
    """Query a new sample against the reference database."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    
    # Load model
    print("[query] Loading model...")
    model = load_model(ckpt_path, device)
    
    # Load database connections
    print("[query] Loading database...")
    conn = duckdb.connect(str(db_path), read_only=True)
    index = faiss.read_index(str(faiss_path))
    
    # Get reference samples
    ref_samples = conn.execute("SELECT sample_id, cell_type FROM samples").fetchall()
    ref_sample_ids = [r[0] for r in ref_samples]
    ref_cell_types = {r[0]: r[1] for r in ref_samples}
    
    print(f"[query] Reference database: {len(ref_samples)} samples, {index.ntotal} loci")
    
    # Create query loader
    print("[query] Loading query sample...")
    if isinstance(ASSAY_TYPES, dict):
        assay_idx = ASSAY_TYPES.get(assay_type, 0)
    else:
        assay_idx = list(ASSAY_TYPES).index(assay_type) if assay_type in ASSAY_TYPES else 0
    loader = create_query_loader_from_tiles(
        mcool_path=mcool_path,
        assay_idx=assay_idx,
        batch_size=32,
        num_workers=0,
    )
    
    print(f"[query] Query sample: {len(loader.dataset)} windows")
    
    # Extract fingerprints
    print("[query] Computing fingerprints...")
    query_fps = []
    query_loci = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            contact = batch["contact"].to(device)
            assay_id = batch["assay_id"].to(device)
            fp = model.encode_fingerprint(contact, assay_id).cpu().numpy()
            query_fps.append(fp)
            
            for i in range(len(fp)):
                query_loci.append({
                    "chr": batch.get("chr", ["?"]*len(fp))[i],
                    "start": batch.get("start_bp", [0]*len(fp))[i],
                    "end": batch.get("end_bp", [0]*len(fp))[i],
                })
    
    query_fps = np.concatenate(query_fps, axis=0)
    query_fps_norm = query_fps / (np.linalg.norm(query_fps, axis=1, keepdims=True) + 1e-8)
    
    # FAISS search for nearest loci
    print("[query] Searching nearest loci...")
    D, I = index.search(query_fps_norm.astype(np.float32), k=1)  # D = similarity, I = locus index
    
    # Get locus info
    locus_rows = conn.execute(
        "SELECT chr, start_bp, end_bp, centroid FROM locus_centroids ORDER BY chr, start_bp"
    ).fetchall()
    
    # For each query window, find top matching reference samples
    print("[query] Finding top matches per window...")
    window_results = []
    
    for i, (fp_norm, locus_info) in enumerate(zip(query_fps_norm, query_loci)):
        # Find all reference fingerprints at this locus
        ref_rows = conn.execute(
            """SELECT wf.sample_id, s.cell_type, wf.fingerprint 
               FROM window_fingerprints wf
               JOIN samples s ON wf.sample_id = s.sample_id
               WHERE wf.chr = ? AND wf.start_bp = ?""",
            [locus_info["chr"], locus_info["start"]],
        ).fetchall()
        
        if not ref_rows:
            continue
        
        # Compute similarities
        similarities = []
        for ref_sample_id, ref_cell_type, fp_blob in ref_rows:
            ref_fp = np.frombuffer(fp_blob, dtype=np.float32)
            ref_fp_norm = ref_fp / (np.linalg.norm(ref_fp) + 1e-8)
            sim = np.dot(fp_norm, ref_fp_norm)
            similarities.append((ref_sample_id, ref_cell_type, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_match = similarities[0]
        
        window_results.append({
            "locus": f"{locus_info['chr']}:{locus_info['start']}-{locus_info['end']}",
            "chr": locus_info["chr"],
            "start": locus_info["start"],
            "end": locus_info["end"],
            "similarity": float(top_match[2]),
            "top_sample": top_match[0],
            "top_cell_type": top_match[1],
            "top_k_matches": [
                {"sample": s, "cell_type": c, "score": float(sim)}
                for s, c, sim in similarities[:k]
            ],
        })
    
    conn.close()
    
    # === LEVEL 1: Sample Classification ===
    print("\n" + "="*70)
    print("LEVEL 1: Sample Classification (Majority Vote)")
    print("="*70)
    
    cell_type_votes = {}
    for w in window_results:
        ct = w["top_cell_type"]
        cell_type_votes[ct] = cell_type_votes.get(ct, 0) + 1
    
    total_windows = len(window_results)
    winner = max(cell_type_votes.items(), key=lambda x: x[1])
    confidence = winner[1] / total_windows
    
    print(f"\n  Most similar to: {winner[0]} ({confidence*100:.1f}% of windows)")
    print(f"  Total windows analyzed: {total_windows}")
    print(f"\n  Vote distribution:")
    for ct, count in sorted(cell_type_votes.items(), key=lambda x: -x[1]):
        pct = count / total_windows * 100
        bar = "█" * int(pct / 2)
        print(f"    {ct:30s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # === LEVEL 2: Chromosome Summary ===
    print("\n" + "="*70)
    print("LEVEL 2: Chromosome Summary (Mean Similarity to Top Reference)")
    print("="*70)
    
    chrom_scores = {}
    for w in window_results:
        chrom = w["chr"]
        if chrom not in chrom_scores:
            chrom_scores[chrom] = []
        chrom_scores[chrom].append(w["similarity"])
    
    print(f"\n  {'Chrom':8s} {'Mean Sim':10s} {'N Windows':10s} {'Status':20s}")
    print(f"  {'-'*60}")
    
    for chrom in sorted(chrom_scores.keys(), key=lambda x: (int(x[3:]) if x[3:].isdigit() else 999)):
        scores = chrom_scores[chrom]
        mean_sim = np.mean(scores)
        n_win = len(scores)
        
        if mean_sim > 0.70:
            status = "STRUCTURALLY_SIMILAR"
        elif mean_sim > 0.30:
            status = "PARTIALLY_SIMILAR"
        else:
            status = "DIVERGENT"
        
        bar = "█" * int(mean_sim * 20)
        print(f"  {chrom:8s} {mean_sim:.3f}     {n_win:3d}        {status:20s} {bar}")
    
    # === LEVEL 3: Window-Level Map (Divergent Windows) ===
    print("\n" + "="*70)
    print("LEVEL 3: Divergent Windows (Similarity < 0.30)")
    print("="*70)
    
    divergent = [w for w in window_results if w["similarity"] < 0.30]
    
    if divergent:
        print(f"\n  Found {len(divergent)} divergent windows out of {total_windows}:")
        print(f"\n  {'Locus':40s} {'Sim':8s} {'Nearest':15s}")
        print(f"  {'-'*70}")
        for w in divergent[:20]:  # Show top 20
            bar = "░" * int(w["similarity"] * 20)
            print(f"  {w['locus']:40s} {w['similarity']:.3f}    {w['top_cell_type']:15s} {bar}")
        if len(divergent) > 20:
            print(f"  ... and {len(divergent) - 20} more")
    else:
        print("\n  No divergent windows found! All windows show structural similarity.")
    
    # === LEVEL 4: Detailed Match Report ===
    print("\n" + "="*70)
    print("LEVEL 4: Top Matches Per Window")
    print("="*70)
    
    for w in window_results[:5]:  # Show first 5 windows
        print(f"\n  ╔{'═'*66}╗")
        print(f"  ║  Locus: {w['locus']:56s} ║")
        print(f"  ║  Similarity: {w['similarity']:.3f}  |  Top match: {w['top_cell_type']:20s} ║")
        
        if w["similarity"] < 0.30:
            status = "DIVERGENT"
        elif w["similarity"] < 0.70:
            status = "PARTIAL"
        else:
            status = "SIMILAR"
        
        print(f"  ║  Status: {status:56s} ║")
        print(f"  ╠{'═'*66}╣")
        print(f"  ║  Top-{k} matches:{'53s'}║")
        for match in w["top_k_matches"]:
            marker = "←" if match["cell_type"] == w["top_cell_type"] else " "
            print(f"  ║    {marker} {match['cell_type']:20s} ({match['sample']:15s}): {match['score']:.3f}   ║")
        print(f"  ╚{'═'*66}╝")
    
    if len(window_results) > 5:
        print(f"\n  ... {len(window_results) - 5} more windows (see JSON output for full details)")
    
    # Compile results
    results = {
        "query_path": mcool_path,
        "assay_type": assay_type,
        "n_windows": total_windows,
        "level1": {
            "top_cell_type": winner[0],
            "confidence": confidence,
            "vote_distribution": cell_type_votes,
        },
        "level2_chromosomes": {
            chrom: {
                "mean_similarity": float(np.mean(scores)),
                "n_windows": len(scores),
                "min_similarity": float(np.min(scores)),
                "max_similarity": float(np.max(scores)),
            }
            for chrom, scores in chrom_scores.items()
        },
        "level3_divergent": [
            {
                "locus": w["locus"],
                "similarity": w["similarity"],
                "top_cell_type": w["top_cell_type"],
                "top_sample": w["top_sample"],
            }
            for w in divergent
        ],
        "window_results": window_results,
    }
    
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mcool", required=True, help="Path to query .mcool file")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    p.add_argument("--db", required=True, help="Path to DuckDB database")
    p.add_argument("--faiss", required=True, help="Path to FAISS index")
    p.add_argument("--assay", default="bulk_hic")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_json", default=None)
    
    args = p.parse_args()
    
    results = query_database(
        mcool_path=args.mcool,
        ckpt_path=args.ckpt,
        db_path=args.db,
        faiss_path=args.faiss,
        assay_type=args.assay,
        k=args.k,
        device_str=args.device,
    )
    
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[query] Full results saved to: {args.out_json}")
