"""
query.py — HiCStructuralDatabase: genome-wide structural similarity retrieval.

Four levels of output:
  Level 1: Sample classification  (majority vote over all windows)
  Level 2: Chromosome summary     (per-chrom mean similarity)
  Level 3: Window-level map       (per-window similarity score + ASCII bar)
  Level 4: Divergent loci report  (novel structural regions)

Usage:
    from query import HiCStructuralDatabase
    db = HiCStructuralDatabase(ckpt_path, db_path, faiss_path)
    report = db.query_file("path/to/query.mcool", assay_type="bulk_hic")
    db.print_report(report)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import duckdb
import faiss

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DB_PATH, FAISS_PATH, ASSAY_TYPES, FP_DIM,
    SIM_THRESHOLD_HIGH, SIM_THRESHOLD_MED, TOP_K,
    CHROMOSOMES, RESOLUTION, CCRE_REGISTRY,
    ANNOT_PERM_N, ANNOT_DIFF_THRESHOLD,
)
from model import MQVAE
from preprocess import preprocess_sample, get_cooler, compute_oe_matrix, compute_boundary_labels, compute_compartment_scores, tile_chromosome
from annotator import (
    MultiSampleAnnotator, CcreIndex, annotate_window,
    bytes_to_annot, differential_report, permutation_pvalue,
    ANNOT_NAMES, CCRE_CATEGORIES, N_CCRE_CATS,
)
from gene_annotator import GeneAnnotator, build_fallback_annotator
from specificity import full_specificity_report, clinical_priority_summary
from config import GENCODE_GTF_PATH


def _classify_score(score: float) -> str:
    if score > SIM_THRESHOLD_HIGH:
        return "STRUCTURALLY_SIMILAR"
    elif score > SIM_THRESHOLD_MED:
        return "PARTIALLY_SIMILAR"
    else:
        return "STRUCTURALLY_DIFFERENT"


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for similarity score."""
    filled = int(round(score * width))
    empty  = width - filled
    return "█" * filled + "░" * empty


class HiCStructuralDatabase:
    """
    Genome-wide structural similarity retrieval against the reference DuckDB database.
    """

    def __init__(
        self,
        ckpt_path:    str,
        db_path:      Path  = DB_PATH,
        faiss_path:   Path  = FAISS_PATH,
        device_str:   str   = "auto",
        model_kwargs: dict  = None,
        ccre_registry: dict = None,
    ):
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # Load model
        print(f"[query] Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        # model_kwargs: explicit override > saved arch > defaults
        arch = model_kwargs or ckpt.get("arch") or {}
        self.model = MQVAE(**arch).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # DuckDB connection (read-only view — no write lock)
        self.conn = duckdb.connect(str(db_path), read_only=True)

        # FAISS centroid index
        self.faiss_index = faiss.read_index(str(faiss_path))
        key_path = Path(faiss_path).with_suffix(".locus_keys.json")
        if key_path.exists():
            with open(key_path) as f:
                raw = json.load(f)
            self.locus_keys = [tuple(k) for k in raw]  # [(chr, start_bp, end_bp), ...]
        else:
            self.locus_keys = []

        print(f"[query] Database ready: {self.faiss_index.ntotal} locus centroids")

        # cCRE annotator for query-side annotation
        reg = ccre_registry if ccre_registry is not None else CCRE_REGISTRY
        if reg:
            self.annotator = MultiSampleAnnotator(reg)
        else:
            self.annotator = None

        # Gene annotator — use GENCODE if available, fallback to curated table
        gtf_path = str(GENCODE_GTF_PATH)
        if Path(gtf_path).exists():
            self.gene_ann = GeneAnnotator(gtf_path)
        else:
            self.gene_ann = build_fallback_annotator()

    # ── Per-locus numpy cosine search ─────────────────────────────────────────

    def _search_locus_numpy(
        self,
        query_fp: np.ndarray,
        chrom:    str,
        start_bp: int,
        k:        int = TOP_K,
    ) -> List[Dict]:
        """
        Fetch all stored fingerprints at this locus and rank by cosine similarity.
        Returns top-k matches as list of dicts.
        Sub-millisecond at ≤500 reference samples.
        """
        rows = self.conn.execute(
            "SELECT wf.sample_id, wf.fingerprint, wf.boundary, wf.compartment, "
            "s.cell_type, s.tissue, wf.ccre_annot "
            "FROM window_fingerprints wf "
            "JOIN samples s ON wf.sample_id = s.sample_id "
            "WHERE wf.chr = ? AND wf.start_bp = ?",
            [chrom, start_bp],
        ).fetchall()

        if not rows:
            return []

        sample_ids  = [r[0] for r in rows]
        fps         = np.stack([np.frombuffer(r[1], dtype=np.float32).copy() for r in rows])
        cell_types  = [r[4] for r in rows]
        tissues     = [r[5] for r in rows]

        # Cosine similarity
        q_norm   = query_fp / (np.linalg.norm(query_fp) + 1e-8)
        fps_norm = fps / (np.linalg.norm(fps, axis=1, keepdims=True) + 1e-8)
        sims     = fps_norm @ q_norm                                 # [n_samples]

        # Permutation p-value against entire reference set at this locus
        _, p_val = permutation_pvalue(query_fp, fps, n_perm=ANNOT_PERM_N)

        top_idx = np.argsort(sims)[::-1][:k]
        matches = []
        for i in top_idx:
            match = {
                "sample_id": sample_ids[i],
                "score":     float(sims[i]),
                "cell_type": cell_types[i],
                "tissue":    tissues[i],
                "status":    _classify_score(float(sims[i])),
                "p_value":   p_val,
            }
            # Boundary concordance
            if rows[i][2] is not None:
                match["boundary_stored"] = np.frombuffer(rows[i][2], dtype=np.float32).copy()
            # Compartment concordance
            if rows[i][3] is not None:
                match["compartment_stored"] = np.frombuffer(rows[i][3], dtype=np.float32).copy()
            # cCRE annotation for reference window
            if rows[i][6] is not None:
                match["ccre_annot_ref"] = bytes_to_annot(rows[i][6])
            matches.append(match)

        return matches

    def _concordance(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        """Pearson correlation between two 1D arrays, or None if unavailable."""
        if a is None or b is None or len(a) == 0:
            return None
        a_c = a - a.mean();  b_c = b - b.mean()
        num = (a_c * b_c).sum()
        den = np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-8
        return float(num / den)

    # ── Main query entry point ─────────────────────────────────────────────────

    def query_mcool(
        self,
        mcool_path:  str,
        sample_id:   str     = "query",
        assay_type:  str     = "bulk_hic",
        k:           int     = TOP_K,
        chroms:      Optional[List[str]] = None,
        verbose:     bool    = True,
    ) -> Dict:
        """
        Query a .mcool file against the reference database.
        Returns structured results dict (all 4 output levels).
        """
        if chroms is None:
            chroms = CHROMOSOMES

        assay_id_int = ASSAY_TYPES.get(assay_type, 0)
        window_results = {}

        # ── Tile the query mcool ───────────────────────────────────────────────
        import cooler as cooler_lib
        uri  = f"{mcool_path}::/resolutions/{RESOLUTION}"
        clr  = cooler_lib.Cooler(uri)
        available_chroms = set(clr.chromnames)

        for chrom in chroms:
            if chrom not in available_chroms:
                continue
            if verbose:
                print(f"  [query] Processing {chrom}...", end="\r")

            oe          = compute_oe_matrix(clr, chrom)
            boundaries  = compute_boundary_labels(clr, chrom)
            compartments = compute_compartment_scores(clr, chrom)
            tiles       = tile_chromosome(oe, boundaries, compartments, chrom)

            for tile in tiles:
                locus = f"{chrom}:{tile['start_bp']}-{tile['end_bp']}"

                # Encode fingerprint
                mat_t = torch.from_numpy(tile["matrix"]).unsqueeze(0).unsqueeze(0).to(self.device)
                aid_t = torch.tensor([assay_id_int], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    fp = self.model.encode_fingerprint(mat_t, aid_t).squeeze(0).cpu().float().numpy()

                # Per-locus numpy search
                matches = self._search_locus_numpy(fp, chrom, tile["start_bp"], k)

                # Concordance with top match
                bd_conc   = None
                comp_conc = None
                if matches:
                    top = matches[0]
                    bd_conc   = self._concordance(tile["boundary"],    top.get("boundary_stored"))
                    comp_conc = self._concordance(tile["compartment"], top.get("compartment_stored"))

                # ── cCRE annotation for query window ──────────────────────
                ccre_annot_query = None
                query_ccre_idx   = None
                ref_ccre_idx     = None
                if self.annotator is not None:
                    ccre_annot_query = self.annotator.annotate(
                        sample_id, chrom, tile["start_bp"], tile["end_bp"]
                    )
                    query_ccre_idx = self.annotator.indices.get(sample_id)
                    if matches:
                        ref_ccre_idx = self.annotator.indices.get(
                            matches[0]["sample_id"]
                        )

                # Simple diff lines (backwards compat + fast display)
                ccre_diff_lines = []
                if (matches and ccre_annot_query is not None
                        and "ccre_annot_ref" in matches[0]):
                    ref_annot = matches[0]["ccre_annot_ref"]
                    diff      = ccre_annot_query - ref_annot
                    for ci, cat in enumerate(CCRE_CATEGORIES):
                        delta = diff[ci]
                        if abs(delta) >= ANNOT_DIFF_THRESHOLD:
                            direction = "more in query" if delta > 0 else "fewer in query"
                            ccre_diff_lines.append(f"{cat}: {delta:+.0f} ({direction})")

                # ── Full specificity report ────────────────────────────────
                sim_score  = matches[0]["score"] if matches else 0.0
                p_val      = matches[0].get("p_value") if matches else None
                ref_sample = matches[0]["sample_id"] if matches else "none"

                # Build ref_fps dict from all matches
                ref_fps_dict = {
                    m["sample_id"]: np.frombuffer(b"", dtype=np.float32)
                    for m in matches
                }
                # Actually use the stored fps from DB for driver analysis
                ref_fps_dict = {}
                for m in matches:
                    rfp_rows = self.conn.execute(
                        "SELECT fingerprint FROM window_fingerprints "
                        "WHERE sample_id=? AND chr=? AND start_bp=?",
                        [m["sample_id"], chrom, tile["start_bp"]],
                    ).fetchone()
                    if rfp_rows:
                        ref_fps_dict[m["sample_id"]] = np.frombuffer(
                            rfp_rows[0], dtype=np.float32
                        ).copy()

                specificity = full_specificity_report(
                    chrom          = chrom,
                    win_start      = tile["start_bp"],
                    win_end        = tile["end_bp"],
                    query_sample_id= sample_id,
                    ref_sample_id  = ref_sample,
                    query_ccre_idx = query_ccre_idx,
                    ref_ccre_idx   = ref_ccre_idx,
                    gene_ann       = self.gene_ann,
                    query_boundary = tile["boundary"],
                    ref_boundary   = matches[0].get("boundary_stored") if matches else None,
                    query_comp     = tile["compartment"],
                    ref_comp       = matches[0].get("compartment_stored") if matches else None,
                    query_fp       = fp,
                    ref_fps        = ref_fps_dict,
                    similarity     = sim_score,
                    p_value        = p_val,
                    resolution     = RESOLUTION,
                )

                window_results[locus] = {
                    "fingerprint":              fp,
                    "chr":                      chrom,
                    "start_bp":                 tile["start_bp"],
                    "end_bp":                   tile["end_bp"],
                    "top_matches":              matches,
                    "similarity_score":         sim_score,
                    "top_cell_type":            matches[0]["cell_type"] if matches else "unknown",
                    "status":                   _classify_score(sim_score),
                    "p_value":                  p_val,
                    "boundary_concordance":     bd_conc,
                    "compartment_concordance":  comp_conc,
                    "boundary_query":           tile["boundary"],
                    "compartment_query":        tile["compartment"],
                    "ccre_annot_query":         ccre_annot_query,
                    "ccre_diff":                ccre_diff_lines,
                    "specificity":              specificity,
                }

        if verbose:
            print()

        # ── Level 1: Sample classification ────────────────────────────────────
        all_cell_types = [v["top_cell_type"] for v in window_results.values() if v["top_matches"]]
        ct_counter     = Counter(all_cell_types)
        total_w        = len(all_cell_types)
        top_cell_type  = ct_counter.most_common(1)[0][0] if ct_counter else "unknown"
        top_fraction   = ct_counter.most_common(1)[0][1] / max(total_w, 1) if ct_counter else 0.0

        # ── Level 2: Chromosome summary ───────────────────────────────────────
        chrom_scores: Dict[str, List[float]] = defaultdict(list)
        for v in window_results.values():
            chrom_scores[v["chr"]].append(v["similarity_score"])
        chrom_summary = {
            ch: {
                "mean_score": float(np.mean(scores)),
                "n_windows":  len(scores),
                "divergent":  float(np.mean(scores)) < SIM_THRESHOLD_MED,
            }
            for ch, scores in chrom_scores.items()
        }

        # ── Level 4: Divergent loci ───────────────────────────────────────────
        divergent_loci = [
            {
                "locus":         locus,
                "chr":           v["chr"],
                "start_bp":      v["start_bp"],
                "end_bp":        v["end_bp"],
                "similarity":    v["similarity_score"],
                "nearest_sample": v["top_matches"][0]["sample_id"] if v["top_matches"] else "none",
                "nearest_ct":    v["top_cell_type"],
                "status":        v["status"],
                "p_value":       v.get("p_value"),
                "boundary_concordance":    v["boundary_concordance"],
                "compartment_concordance": v["compartment_concordance"],
                "ccre_diff":     v.get("ccre_diff", []),
                "specificity":   v.get("specificity"),
                "ccre_annot_query": (
                    v["ccre_annot_query"].tolist()
                    if v.get("ccre_annot_query") is not None else None
                ),
            }
            for locus, v in window_results.items()
            if v["similarity_score"] < SIM_THRESHOLD_MED
        ]
        divergent_loci.sort(key=lambda x: x["similarity"])  # most divergent first

        return {
            "query_path":         mcool_path,
            "assay_type":         assay_type,
            "n_windows":          len(window_results),
            "window_results":     window_results,
            "level1_cell_type":   top_cell_type,
            "level1_fraction":    top_fraction,
            "level2_chrom":       chrom_summary,
            "level4_divergent":   divergent_loci,
        }

    # ── Report formatting ──────────────────────────────────────────────────────

    def print_report(self, results: Dict, max_divergent: int = 10) -> None:
        """Print all four levels of output to stdout."""
        print("\n" + "=" * 60)
        print("  MQ-VAE STRUCTURAL SIMILARITY REPORT")
        print("=" * 60)
        print(f"  Query:    {results['query_path']}")
        print(f"  Assay:    {results['assay_type']}")
        print(f"  Windows:  {results['n_windows']}")

        # ── Level 1 ───────────────────────────────────────────────────────────
        print("\n── LEVEL 1: Sample Classification ─────────────────────────")
        frac_pct = results["level1_fraction"] * 100
        print(f"  Most similar to: {results['level1_cell_type']}  "
              f"({frac_pct:.0f}% of windows)")

        # ── Level 2 ───────────────────────────────────────────────────────────
        print("\n── LEVEL 2: Chromosome Summary ─────────────────────────────")
        for chrom in CHROMOSOMES:
            if chrom not in results["level2_chrom"]:
                continue
            info  = results["level2_chrom"][chrom]
            score = info["mean_score"]
            bar   = _bar(score, 20)
            flag  = "  ← DIVERGENT" if info["divergent"] else ""
            print(f"  {chrom:<6}  {score:.2f}  {bar}{flag}")

        # ── Level 3 ───────────────────────────────────────────────────────────
        print("\n── LEVEL 3: Window-Level Map (divergent windows only) ──────")
        shown = 0
        for locus, v in sorted(results["window_results"].items(),
                                key=lambda x: x[1]["similarity_score"]):
            if v["similarity_score"] >= SIM_THRESHOLD_MED:
                continue
            score  = v["similarity_score"]
            bar    = _bar(score, 15)
            ct     = v["top_cell_type"]
            pv_str = f"  p={v['p_value']:.3f}" if v.get("p_value") is not None else ""
            print(f"  {locus:<35}  {score:.3f}  {bar}  nearest: {ct}{pv_str}")
            if v.get("ccre_diff"):
                for line in v["ccre_diff"][:3]:
                    print(f"    ↳ {line}")
            shown += 1
            if shown >= 20:
                print(f"  ... ({len(results['level4_divergent']) - shown} more divergent windows)")
                break

        # ── Level 4 ───────────────────────────────────────────────────────────
        print("\n── LEVEL 4: Divergent Loci Report ──────────────────────────")
        if not results["level4_divergent"]:
            print("  No divergent loci detected (all windows above threshold).")
        else:
            for i, loc in enumerate(results["level4_divergent"][:max_divergent]):
                w   = 66
                sp  = loc.get("specificity") or {}
                cl  = sp.get("clinical") or {}
                qc  = sp.get("qc") or {}
                bd  = sp.get("boundaries") or {}
                cpr = sp.get("compartments") or {}
                ccr = sp.get("ccre") or {}

                print("\n  ╔" + "═" * w + "╗")
                print(f"  ║  Locus:      {loc['locus']:<{w-14}}║")
                print(f"  ║  Similarity: {loc['similarity']:.3f}   nearest: {loc['nearest_ct']:<{w-30}}║")

                # Clinical priority
                prio  = cl.get('priority', 'INFORMATIONAL')
                score = cl.get('actionability_score', 0)
                print(f"  ║  Priority:   {prio} (score={score}/10)".ljust(w+4) + "║")
                if cl.get('summary'):
                    print(f"  ║  ↳ {cl['summary'][:w-2]:<{w-2}}║")

                # P-value
                if loc.get("p_value") is not None:
                    sig = " *" if loc["p_value"] < 0.05 else ""
                    print(f"  ║  P-value:    {loc['p_value']:.4f}{sig:<{w-18}}║")

                # QC flags
                if qc.get("flags"):
                    for flag in qc["flags"][:2]:
                        print(f"  ║  QC: {flag['type']} — {flag['description'][:w-10]:<{w-10}}║")

                # cCRE coordinate details
                cancer_ccre = [
                    (cat, info) for cat, info in ccr.items()
                    if info.get("cancer_genes_nearby") and abs(info.get("delta", 0)) >= 5
                ]
                if cancer_ccre:
                    print(f"  ║  Regulatory changes near cancer genes:".ljust(w+4) + "║")
                    for cat, info in cancer_ccre[:4]:
                        genes = ", ".join(info["cancer_genes_nearby"][:2])
                        delta = info["delta"]
                        note  = info.get("clinical_note", "")
                        print(f"  ║    {cat}: {delta:+d} near {genes}".ljust(w+4) + "║")
                        if note:
                            print(f"  ║    ↳ {note[:w-4]:<{w-4}}║")
                        # Show top 3 exact coordinates
                        gained = info.get("gained_coords", [])[:3]
                        for coord in gained:
                            gene  = coord.get('nearest_gene', '?')
                            dlbl  = coord.get('distance_label', '')
                            cr    = coord.get('cancer_role', '')
                            cr_str= f" [{cr}]" if cr else ""
                            line  = f"      chr{loc['chr'] if not loc['chr'].startswith('chr') else loc['chr'][3:]}:{coord['start']:,}-{coord['end']:,}  near {gene} ({dlbl}){cr_str}"
                            print(f"  ║  {line[:w]:<{w}}║")

                elif loc.get("ccre_diff"):
                    print(f"  ║  Regulatory differences (query vs {loc['nearest_ct'][:20]}):".ljust(w+4) + "║")
                    for line in loc["ccre_diff"][:4]:
                        print(f"  ║    {line:<{w-2}}║")

                # Lost TAD boundaries
                lost_bd = bd.get("lost_boundaries", [])
                if lost_bd:
                    print(f"  ║  Lost TAD boundaries:".ljust(w+4) + "║")
                    for b in lost_bd[:3]:
                        pos   = b.get("position_bp", 0)
                        lg    = ", ".join(b.get("left_genes", [])[:2])
                        rg    = ", ".join(b.get("right_genes", [])[:2])
                        note  = b.get("clinical_note", "")
                        print(f"  ║    {loc['chr']}:{pos:,}  flanking: {lg} | {rg}".ljust(w+4) + "║")
                        if note:
                            print(f"  ║    ↳ {note[:w-4]:<{w-4}}║")

                # Compartment switches
                switches = cpr.get("switches", [])
                if switches:
                    print(f"  ║  Compartment switches:".ljust(w+4) + "║")
                    for sw in switches[:2]:
                        direction   = sw.get("direction", "?")
                        consequence = sw.get("functional_consequence", "")
                        cgenes      = ", ".join(sw.get("cancer_genes", [])[:2])
                        note        = sw.get("clinical_note", "")
                        line = f"    {direction}  {consequence[:40]}"
                        if cgenes:
                            line += f"  genes: {cgenes}"
                        print(f"  ║  {line[:w]:<{w}}║")
                        if note:
                            print(f"  ║    ↳ {note[:w-4]:<{w-4}}║")

                # Concordance
                if loc["boundary_concordance"] is not None:
                    bc = loc["boundary_concordance"]
                    bc_label = "low" if bc < 0.3 else ("medium" if bc < 0.7 else "high")
                    print(f"  ║  Boundary concordance: {bc:.3f} ({bc_label})".ljust(w+4) + "║")
                if loc["compartment_concordance"] is not None:
                    cc = loc["compartment_concordance"]
                    cc_label = "very low" if cc < 0.2 else ("low" if cc < 0.4 else ("medium" if cc < 0.7 else "high"))
                    print(f"  ║  Compartment concordance: {cc:.3f} ({cc_label})".ljust(w+4) + "║")

                print("  ╚" + "═" * w + "╝")
        print("\n" + "=" * 60 + "\n")

    def close(self):
        self.conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as json_mod

    p = argparse.ArgumentParser(description="Query Hi-C database for structural similarity")
    p.add_argument("--mcool",       required=True, help="Path to query .mcool file")
    p.add_argument("--ckpt",        required=True, help="Model checkpoint (.pt)")
    p.add_argument("--db",          default=str(DB_PATH))
    p.add_argument("--faiss",       default=str(FAISS_PATH))
    p.add_argument("--assay",       default="bulk_hic")
    p.add_argument("--k",           type=int, default=TOP_K)
    p.add_argument("--out_json",    default=None, help="Save results to JSON")
    p.add_argument("--device",      default="auto")
    args = p.parse_args()

    db = HiCStructuralDatabase(
        ckpt_path  = args.ckpt,
        db_path    = Path(args.db),
        faiss_path = Path(args.faiss),
        device_str = args.device,
    )
    results = db.query_mcool(
        mcool_path = args.mcool,
        assay_type = args.assay,
        k          = args.k,
        verbose    = True,
    )
    db.print_report(results)

    if args.out_json:
        # Serialise (drop raw numpy arrays)
        safe = {
            k: v for k, v in results.items()
            if k not in ("window_results",)
        }
        window_safe = {}
        for locus, v in results["window_results"].items():
            window_safe[locus] = {
                kk: (float(vv) if isinstance(vv, (float, np.floating)) else
                     (int(vv) if isinstance(vv, (int, np.integer)) else
                      (vv.tolist() if isinstance(vv, np.ndarray) else vv)))
                for kk, vv in v.items()
                if kk not in ("fingerprint", "boundary_query", "compartment_query",
                               "boundary_stored", "compartment_stored")
            }
        safe["window_results"] = window_safe
        with open(args.out_json, "w") as f:
            json_mod.dump(safe, f, indent=2)
        print(f"[query] Results saved to {args.out_json}")

    db.close()
