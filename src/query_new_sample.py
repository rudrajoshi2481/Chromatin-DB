"""
query_new_sample.py — Query a raw .mcool file against the Chromatin-DB reference.

This is for held-out samples NOT in the training database.
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
import cooler

sys.path.insert(0, str(Path(__file__).parent))

from config import FP_DIM, ASSAY_TYPES, TILE_SIZE, RESOLUTION
from model import MQVAE
from preprocess import compute_oe_matrix, tile_chromosome


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


def extract_tiles_from_mcool(mcool_path: str, assay_idx: int = 0, chroms=None):
    """Extract 256×256 tiles from a raw mcool file on-the-fly."""
    print(f"[query] Opening {mcool_path}...")
    clr = cooler.Cooler(f"{mcool_path}::/resolutions/{RESOLUTION}")
    
    if chroms is None:
        # Use common chromosomes present in the file
        chroms = [c for c in [f"chr{i}" for i in range(1, 23)] + ["chrX"] 
                  if c in clr.chromnames][:6]  # Limit to first 6 for speed
    
    print(f"[query] Extracting tiles from {len(chroms)} chromosomes...")
    tiles = []
    
    for chrom in chroms:
        try:
            # Compute OE matrix
            oe_mat = compute_oe_matrix(clr, chrom)
            if oe_mat is None or oe_mat.size == 0:
                continue
                
            # Compute dummy boundaries/compartments (not used for fingerprint, but required)
            n_bins = oe_mat.shape[0]
            boundaries = np.zeros(n_bins, dtype=np.float32)
            compartments = np.zeros(n_bins, dtype=np.float32)
            
            # Tile the chromosome
            chrom_tiles = tile_chromosome(
                oe_matrix=oe_mat,
                boundaries=boundaries,
                compartments=compartments,
                chrom=chrom,
                step=TILE_SIZE // 2,  # 50% overlap
            )
            
            # Add assay index to each tile
            for tile in chrom_tiles:
                tile["assay_id"] = assay_idx
                
            tiles.extend(chrom_tiles)
            print(f"  {chrom}: {len(chrom_tiles)} tiles")
            
        except Exception as e:
            print(f"  {chrom}: skipped ({e})")
            continue
    
    print(f"[query] Total tiles extracted: {len(tiles)}")
    return tiles


def query_new_sample(
    mcool_path: str,
    ckpt_path: str,
    db_path: str,
    faiss_path: str,
    assay_type: str = "bulk_hic",
    k: int = 5,
    device_str: str = "auto",
    max_windows: int = 200,
):
    """Query a new mcool file against the reference database."""
    
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    
    # Load model
    print("[query] Loading model...")
    model = load_model(ckpt_path, device)
    
    # Load database
    print("[query] Loading database...")
    conn = duckdb.connect(str(db_path), read_only=True)
    index = faiss.read_index(str(faiss_path))
    
    # Get reference samples
    ref_samples = conn.execute("SELECT sample_id, cell_type FROM samples").fetchall()
    print(f"[query] Reference: {len(ref_samples)} samples, {index.ntotal} loci")
    
    # Extract tiles from query mcool
    assay_idx = ASSAY_TYPES.get(assay_type, 0) if isinstance(ASSAY_TYPES, dict) else 0
    tiles = extract_tiles_from_mcool(mcool_path, assay_idx)
    
    if len(tiles) == 0:
        print("[query] ERROR: No tiles extracted from mcool file")
        return None
    
    # Limit windows for speed
    tiles = tiles[:max_windows]
    print(f"[query] Using {len(tiles)} windows for query")
    
    # Compute fingerprints
    print("[query] Computing fingerprints...")
    query_fps = []
    query_loci = []
    
    with torch.no_grad():
        for tile in tqdm(tiles):
            contact = torch.from_numpy(tile["matrix"]).unsqueeze(0).unsqueeze(0).to(device)
            assay_id = torch.tensor([tile["assay_id"]], dtype=torch.long).to(device)
            fp = model.encode_fingerprint(contact, assay_id).cpu().numpy()[0]
            query_fps.append(fp)
            query_loci.append({
                "chr": tile["chr"],
                "start": tile["start_bp"],
                "end": tile["end_bp"],
            })
    
    query_fps = np.array(query_fps)
    query_fps_norm = query_fps / (np.linalg.norm(query_fps, axis=1, keepdims=True) + 1e-8)
    
    # For each query window, find matches at the SAME locus across reference samples
    print("[query] Finding top matches per window (locus-filtered)...")
    window_results = []
    
    for i, (fp_norm, locus) in enumerate(zip(query_fps_norm, query_loci)):
        # Query database for windows at EXACTLY this locus
        ref_rows = conn.execute(
            """SELECT wf.sample_id, s.cell_type, wf.fingerprint 
               FROM window_fingerprints wf
               JOIN samples s ON wf.sample_id = s.sample_id
               WHERE wf.chr = ? AND wf.start_bp = ?""",
            [locus["chr"], locus["start"]],
        ).fetchall()
        
        if not ref_rows:
            # No reference data at this exact locus - use FAISS nearest locus
            D, I = index.search(fp_norm.reshape(1, -1).astype(np.float32), k=1)
            window_results.append({
                "locus": f"{locus['chr']}:{locus['start']}-{locus['end']}",
                "chr": locus["chr"],
                "start": locus["start"],
                "end": locus["end"],
                "similarity": float(D[0][0]),
                "top_sample": "NO_REFERENCE_LOCUS",
                "top_cell_type": "UNKNOWN",
                "top_k_matches": [],
                "status": "NO_REFERENCE",
            })
            continue
        
        # Compute cosine similarity to each reference sample at this locus
        similarities = []
        for ref_sample_id, ref_cell_type, fp_blob in ref_rows:
            ref_fp = np.frombuffer(fp_blob, dtype=np.float32)
            ref_fp_norm = ref_fp / (np.linalg.norm(ref_fp) + 1e-8)
            sim = np.dot(fp_norm, ref_fp_norm)
            similarities.append({
                "sample": ref_sample_id,
                "cell_type": ref_cell_type,
                "score": float(sim),
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["score"], reverse=True)
        top = similarities[0]
        
        # Determine status
        if top["score"] < 0.30:
            status = "STRUCTURALLY_DIFFERENT"
        elif top["score"] < 0.70:
            status = "PARTIALLY_SIMILAR"
        else:
            status = "STRUCTURALLY_SIMILAR"
        
        window_results.append({
            "locus": f"{locus['chr']}:{locus['start']}-{locus['end']}",
            "chr": locus["chr"],
            "start": locus["start"],
            "end": locus["end"],
            "similarity": top["score"],
            "top_sample": top["sample"],
            "top_cell_type": top["cell_type"],
            "top_k_matches": similarities[:k],
            "status": status,
        })
    
    conn.close()
    
    # === LEVEL 1: Sample Classification ===
    print("\n" + "="*70)
    print("LEVEL 1: Sample Classification (Majority Vote Across All Windows)")
    print("="*70)
    
    cell_type_votes = {}
    sample_votes = {}
    total_valid = 0
    
    for w in window_results:
        if w["status"] == "NO_REFERENCE":
            continue
        total_valid += 1
        ct = w["top_cell_type"]
        sid = w["top_sample"]
        cell_type_votes[ct] = cell_type_votes.get(ct, 0) + 1
        sample_votes[sid] = sample_votes.get(sid, 0) + 1
    
    if total_valid == 0:
        print("\n  ERROR: No valid windows with reference matches")
        return None
    
    # Winner by cell type
    winner_ct = max(cell_type_votes.items(), key=lambda x: x[1])
    confidence_ct = winner_ct[1] / total_valid
    
    # Winner by sample
    winner_sid = max(sample_votes.items(), key=lambda x: x[1])
    confidence_sid = winner_sid[1] / total_valid
    
    print(f"\n  Query: {Path(mcool_path).name}")
    print(f"  Most similar CELL TYPE: {winner_ct[0]} ({confidence_ct*100:.1f}% of windows)")
    print(f"  Most similar SAMPLE:    {winner_sid[0]} ({confidence_sid*100:.1f}% of windows)")
    print(f"  Total windows analyzed: {total_valid}")
    
    print(f"\n  Top cell type votes:")
    for ct, count in sorted(cell_type_votes.items(), key=lambda x: -x[1])[:5]:
        pct = count / total_valid * 100
        bar = "█" * int(pct / 3)
        print(f"    {ct:30s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # === LEVEL 2: Chromosome Summary ===
    print("\n" + "="*70)
    print("LEVEL 2: Chromosome Summary (Mean Similarity to Top Reference)")
    print("="*70)
    
    chrom_stats = {}
    for w in window_results:
        if w["status"] == "NO_REFERENCE":
            continue
        chrom = w["chr"]
        if chrom not in chrom_stats:
            chrom_stats[chrom] = {"scores": [], "top_ct": []}
        chrom_stats[chrom]["scores"].append(w["similarity"])
        chrom_stats[chrom]["top_ct"].append(w["top_cell_type"])
    
    print(f"\n  {'Chrom':8s} {'Mean Sim':10s} {'N':5s} {'Top Cell Type':20s} {'Status':20s}")
    print(f"  {'-'*70}")
    
    for chrom in sorted(chrom_stats.keys(), key=lambda x: (int(x[3:]) if x[3:].isdigit() else 999)):
        stats = chrom_stats[chrom]
        mean_sim = np.mean(stats["scores"])
        n = len(stats["scores"])
        # Most common cell type on this chromosome
        from collections import Counter
        top_ct = Counter(stats["top_ct"]).most_common(1)[0][0]
        
        if mean_sim > 0.70:
            status = "STRUCTURALLY_SIMILAR"
        elif mean_sim > 0.30:
            status = "PARTIALLY_SIMILAR"
        else:
            status = "DIVERGENT"
        
        bar = "█" * int(mean_sim * 20)
        print(f"  {chrom:8s} {mean_sim:.3f}     {n:3d} {top_ct:20s} {status:20s} {bar}")
    
    # === LEVEL 3: Divergent Windows ===
    divergent = [w for w in window_results if w["status"] == "STRUCTURALLY_DIFFERENT"]
    partial = [w for w in window_results if w["status"] == "PARTIALLY_SIMILAR"]
    
    print("\n" + "="*70)
    print("LEVEL 3: Divergent Windows (Similarity < 0.30)")
    print("="*70)
    
    if divergent:
        print(f"\n  Found {len(divergent)} STRUCTURALLY DIFFERENT windows:")
        print(f"\n  {'Locus':40s} {'Sim':8s} {'Nearest Cell Type':20s}")
        print(f"  {'-'*70}")
        for w in divergent[:15]:  # Show top 15
            bar = "░" * int(w["similarity"] * 20) if w["similarity"] > 0 else "░░░░░░"
            print(f"  {w['locus']:40s} {w['similarity']:.3f}    {w['top_cell_type']:20s} {bar}")
        if len(divergent) > 15:
            print(f"  ... and {len(divergent) - 15} more divergent windows")
    else:
        print("\n  No structurally different windows found.")
    
    if partial:
        print(f"\n  {len(partial)} windows are PARTIALLY SIMILAR (0.30-0.70)")
    
    similar = [w for w in window_results if w["status"] == "STRUCTURALLY_SIMILAR"]
    print(f"\n  {len(similar)} windows are STRUCTURALLY SIMILAR (>0.70)")
    
    # === LEVEL 4: Detailed Matches ===
    print("\n" + "="*70)
    print("LEVEL 4: Top Matches Per Window (showing DIVERGENT windows first)")
    print("="*70)
    
    # Show divergent windows first, then a few similar ones
    windows_to_show = divergent[:5] + similar[:3]
    
    for w in windows_to_show:
        print(f"\n  ╔{'═'*66}╗")
        print(f"  ║  Locus: {w['locus']:56s} ║")
        print(f"  ║  Similarity to nearest: {w['similarity']:.3f}  |  Top match: {w['top_cell_type']:15s} ║")
        print(f"  ║  Status: {w['status']:56s} ║")
        print(f"  ╠{'═'*66}╣")
        print(f"  ║  Top-{k} matches at this locus:{'41s'}║")
        for match in w["top_k_matches"]:
            marker = "←" if match["cell_type"] == w["top_cell_type"] else " "
            print(f"  ║    {marker} {match['cell_type']:20s} ({match['sample']:15s}): {match['score']:.3f}   ║")
        print(f"  ╚{'═'*66}╝")
    
    # Compile results
    results = {
        "query_path": mcool_path,
        "query_sample": Path(mcool_path).stem,
        "assay_type": assay_type,
        "n_windows_total": len(tiles),
        "n_windows_valid": total_valid,
        "level1": {
            "top_cell_type": winner_ct[0],
            "cell_type_confidence": confidence_ct,
            "top_sample": winner_sid[0],
            "sample_confidence": confidence_sid,
            "vote_distribution": cell_type_votes,
        },
        "level2_chromosomes": {
            chrom: {
                "mean_similarity": float(np.mean(stats["scores"])),
                "n_windows": len(stats["scores"]),
                "top_cell_type": str(Counter(stats["top_ct"]).most_common(1)[0][0]),
            }
            for chrom, stats in chrom_stats.items()
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
        "summary": {
            "n_divergent": len(divergent),
            "n_partial": len(partial),
            "n_similar": len(similar),
            "mean_similarity": float(np.mean([w["similarity"] for w in window_results if w["status"] != "NO_REFERENCE"])),
        },
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
    p.add_argument("--max_windows", type=int, default=200, help="Limit windows for speed")
    p.add_argument("--out_json", default=None)
    
    args = p.parse_args()
    
    results = query_new_sample(
        mcool_path=args.mcool,
        ckpt_path=args.ckpt,
        db_path=args.db,
        faiss_path=args.faiss,
        assay_type=args.assay,
        k=args.k,
        device_str=args.device,
        max_windows=args.max_windows,
    )
    
    if results and args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[query] Full results saved to: {args.out_json}")
