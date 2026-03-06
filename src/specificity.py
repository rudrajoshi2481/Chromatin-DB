"""
specificity.py — Actionable genomic location reports.

Transforms vague counts into precise, biologically-grounded findings:

  1. cCRE coordinates  → exact positions + nearest gene + distance + cancer role
  2. Boundary losses   → which TAD boundaries disrupted + flanking genes
  3. Compartment switch→ A/B switch regions + affected genes + consequence
  4. Similarity drivers→ which sub-windows drive divergence
  5. QC flags          → artifact regions, low coverage, telomere/centromere
  6. Clinical summary  → prioritised findings with actionability score

All functions return structured dicts suitable for JSON output
and for printing in the LEVEL 4 report.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from annotator import (
    CcreIndex, CCRE_CATEGORIES, N_CCRE_CATS, ANNOT_NAMES,
    counts_to_annot_vector,
)
from gene_annotator import GeneAnnotator, ALL_CANCER_GENES, ONCOGENES, TUMOR_SUPPRESSORS

# ── Constants ─────────────────────────────────────────────────────────────────
NEAR_GENE_DIST   = 500_000    # 500kb = "near" a gene
MAX_GENE_DIST    = 2_000_000  # 2Mb   = within regulatory range
QC_TELOMERE_BPS  = 5_000_000  # first/last 5Mb of chrom = telomere risk
CENTROMERE_CHR21 = (10_000_000, 13_000_000)
CENTROMERE_CHR22 = (11_000_000, 15_000_000)


# ── 1. cCRE coordinate report ─────────────────────────────────────────────────

def ccre_coordinate_report(
    query_index:  CcreIndex,
    ref_index:    Optional[CcreIndex],
    gene_ann:     GeneAnnotator,
    chrom:        str,
    win_start:    int,
    win_end:      int,
    diff_threshold: int = 5,
    max_coords:   int   = 10,
) -> Dict:
    """
    Full coordinate report for one window comparing query vs reference.

    Returns:
      {
        cat_name: {
          "query_count": int,
          "ref_count": int,
          "delta": int,
          "direction": "gained" | "lost" | "unchanged",
          "query_coords": [{start, end, nearest_gene, distance_label, cancer_role}, ...],
          "gained_coords": [...],   # in query but not in ref
          "lost_coords":  [...],    # in ref but not in query
          "cancer_genes_nearby": [gene_name, ...],
          "clinical_note": str | None,
        }
      }
    """
    q_counts, q_coords = query_index.query_window_with_coords(chrom, win_start, win_end)

    r_counts = np.zeros(N_CCRE_CATS, dtype=np.int32)
    r_coords: Dict = {c: [] for c in CCRE_CATEGORIES}
    if ref_index is not None:
        r_counts, r_coords = ref_index.query_window_with_coords(chrom, win_start, win_end)

    win_size = win_end - win_start

    # Annotate query coords with gene context
    q_annotated = gene_ann.annotate_ccre_hits(chrom, q_coords, NEAR_GENE_DIST) if gene_ann.loaded else {c: [] for c in CCRE_CATEGORIES}
    r_annotated = gene_ann.annotate_ccre_hits(chrom, r_coords, NEAR_GENE_DIST) if gene_ann.loaded else {c: [] for c in CCRE_CATEGORIES}

    result = {}
    for ci, cat in enumerate(CCRE_CATEGORIES):
        qc  = int(q_counts[ci])
        rc  = int(r_counts[ci])
        delta = qc - rc

        if abs(delta) < diff_threshold and qc == 0:
            continue

        # Identify gained (in query, not in ref) and lost (in ref, not in query)
        q_set = set(q_coords[cat])
        r_set = set(r_coords[cat])
        gained = sorted(q_set - r_set, key=lambda x: x[0])
        lost   = sorted(r_set - q_set, key=lambda x: x[0])

        # Cancer genes near any query element
        cancer_nearby = list({
            e["nearest_gene"]
            for e in q_annotated.get(cat, [])
            if e.get("cancer_role") and e.get("nearest_gene") != "intergenic"
        })

        # Format coordinate entries
        def fmt_coords(coord_list, annotated_list, limit):
            ann_map = {(e["start"], e["end"]): e for e in annotated_list}
            out = []
            for (s, e) in coord_list[:limit]:
                ann = ann_map.get((s, e), {})
                out.append({
                    "start":          s,
                    "end":            e,
                    "size_bp":        e - s,
                    "nearest_gene":   ann.get("nearest_gene", "unknown"),
                    "distance_label": ann.get("distance_label", "unknown"),
                    "cancer_role":    ann.get("cancer_role"),
                })
            return out

        entry = {
            "query_count":        qc,
            "ref_count":          rc,
            "delta":              delta,
            "direction":          "gained" if delta > 0 else ("lost" if delta < 0 else "unchanged"),
            "query_coords":       fmt_coords(q_coords[cat], q_annotated.get(cat, []), max_coords),
            "gained_coords":      fmt_coords(gained, q_annotated.get(cat, []), max_coords),
            "lost_coords":        fmt_coords(lost,   r_annotated.get(cat, []), max_coords),
            "cancer_genes_nearby": cancer_nearby,
            "clinical_note":      _ccre_clinical_note(cat, delta, cancer_nearby),
        }

        if abs(delta) >= diff_threshold or qc > 0:
            result[cat] = entry

    return result


# ── 2. Boundary disruption report ─────────────────────────────────────────────

def boundary_disruption_report(
    query_boundary:  np.ndarray,   # [n_bins] float array, 1=boundary
    ref_boundary:    Optional[np.ndarray],
    chrom:           str,
    win_start:       int,
    resolution:      int,
    gene_ann:        GeneAnnotator,
    threshold:       float = 0.5,
    concordance_threshold: float = 0.4,
) -> Dict:
    """
    Identify specific TAD boundary positions that were lost or gained
    compared to the reference.

    Returns:
      {
        "concordance": float,
        "n_query_boundaries": int,
        "n_ref_boundaries":   int,
        "lost_boundaries":    [{position_bp, left_genes, right_genes, clinical_note}, ...],
        "gained_boundaries":  [{position_bp, left_genes, right_genes, clinical_note}, ...],
        "all_query_boundaries": [position_bp, ...],
      }
    """
    if query_boundary is None or len(query_boundary) == 0:
        return {}

    q_bins = np.where(query_boundary >= threshold)[0]
    q_bps  = [win_start + int(b) * resolution for b in q_bins]

    if ref_boundary is None or len(ref_boundary) == 0:
        return {
            "concordance": None,
            "n_query_boundaries": len(q_bps),
            "n_ref_boundaries": 0,
            "all_query_boundaries": q_bps,
            "lost_boundaries": [],
            "gained_boundaries": _annotate_boundary_positions(q_bps, chrom, gene_ann, "gained"),
        }

    r_bins = np.where(ref_boundary >= threshold)[0]
    r_bps  = [win_start + int(b) * resolution for b in r_bins]

    # Concordance
    a_c = query_boundary - query_boundary.mean()
    b_c = ref_boundary   - ref_boundary.mean()
    num = (a_c * b_c).sum()
    den = np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-8
    concordance = float(num / den)

    # Find lost (in ref but not in query) and gained (in query but not in ref)
    # Use 1-bin tolerance for matching
    def match(pos, candidates, tol_bp):
        return any(abs(pos - c) <= tol_bp for c in candidates)

    tol = resolution * 2
    lost   = [p for p in r_bps if not match(p, q_bps, tol)]
    gained = [p for p in q_bps if not match(p, r_bps, tol)]

    return {
        "concordance":          concordance,
        "n_query_boundaries":   len(q_bps),
        "n_ref_boundaries":     len(r_bps),
        "all_query_boundaries": q_bps,
        "all_ref_boundaries":   r_bps,
        "lost_boundaries":      _annotate_boundary_positions(lost,   chrom, gene_ann, "lost"),
        "gained_boundaries":    _annotate_boundary_positions(gained, chrom, gene_ann, "gained"),
        "is_disrupted":         concordance < concordance_threshold,
    }


def _annotate_boundary_positions(
    positions: List[int],
    chrom: str,
    gene_ann: GeneAnnotator,
    direction: str,
) -> List[Dict]:
    if not gene_ann.loaded:
        return [{"position_bp": p, "direction": direction} for p in positions]
    annotated = gene_ann.annotate_boundary_losses(chrom, positions)
    for entry in annotated:
        entry["direction"] = direction
    return annotated


# ── 3. Compartment switch report ──────────────────────────────────────────────

def compartment_switch_report(
    query_comp:  np.ndarray,   # [n_bins] E1 scores
    ref_comp:    Optional[np.ndarray],
    chrom:       str,
    win_start:   int,
    resolution:  int,
    gene_ann:    GeneAnnotator,
    switch_threshold: float = 0.3,
) -> Dict:
    """
    Detect A/B compartment switches between query and reference.
    A compartment = positive E1, B compartment = negative E1.

    Returns:
      {
        "concordance": float,
        "n_switches": int,
        "switches": [{start, end, direction, genes, consequence, clinical_note}, ...]
      }
    """
    if query_comp is None or len(query_comp) == 0:
        return {}

    if ref_comp is None or len(ref_comp) == 0:
        return {"concordance": None, "n_switches": 0, "switches": []}

    # Align lengths
    min_len = min(len(query_comp), len(ref_comp))
    qc = query_comp[:min_len]
    rc = ref_comp[:min_len]

    # Concordance
    a_c = qc - qc.mean()
    b_c = rc - rc.mean()
    num = (a_c * b_c).sum()
    den = np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-8
    concordance = float(num / den)

    # Find switched bins (sign change + magnitude threshold)
    q_a = qc > switch_threshold    # query A compartment
    r_a = rc > switch_threshold    # ref A compartment
    q_b = qc < -switch_threshold
    r_b = rc < -switch_threshold

    b_to_a = q_a & r_b  # gained A
    a_to_b = q_b & r_a  # lost A (gained B)

    def find_runs(mask, direction):
        runs = []
        in_run = False
        for i, v in enumerate(mask):
            if v and not in_run:
                run_start = i
                in_run = True
            elif not v and in_run:
                runs.append((run_start, i, direction))
                in_run = False
        if in_run:
            runs.append((run_start, len(mask), direction))
        return runs

    switches_raw = find_runs(b_to_a, "B->A") + find_runs(a_to_b, "A->B")
    switches_raw.sort(key=lambda x: x[0])

    # Convert bin indices to bp and annotate
    switch_dicts = []
    for (bin_start, bin_end, direction) in switches_raw:
        bp_start = win_start + bin_start * resolution
        bp_end   = win_start + bin_end   * resolution
        switch_dicts.append({
            "start":     bp_start,
            "end":       bp_end,
            "direction": direction,
        })

    annotated_switches = []
    if gene_ann.loaded:
        annotated_switches = gene_ann.annotate_compartment_switches(chrom, switch_dicts)
    else:
        annotated_switches = switch_dicts

    return {
        "concordance": concordance,
        "n_switches":  len(annotated_switches),
        "switches":    annotated_switches,
        "is_disrupted": concordance < 0.4,
    }


# ── 4. Similarity driver analysis ─────────────────────────────────────────────

def similarity_driver_report(
    query_fp:   np.ndarray,
    ref_fps:    Dict[str, np.ndarray],  # {sample_id: fingerprint}
    chrom:      str,
    win_start:  int,
    win_end:    int,
    gene_ann:   GeneAnnotator,
) -> Dict:
    """
    Identify which dimensions of the fingerprint drive divergence.
    Report with gene context for the window.
    """
    if not ref_fps:
        return {}

    # Cosine similarity to each reference
    q_norm = query_fp / (np.linalg.norm(query_fp) + 1e-8)
    sim_scores = {}
    for sid, rfp in ref_fps.items():
        r_norm = rfp / (np.linalg.norm(rfp) + 1e-8)
        sim_scores[sid] = float(np.dot(q_norm, r_norm))

    sorted_refs = sorted(sim_scores.items(), key=lambda x: x[1])
    most_similar   = sorted_refs[-1]
    most_divergent = sorted_refs[0]

    # Fingerprint dimension difference vs most similar
    sid_top = most_similar[0]
    rfp_top = ref_fps[sid_top]
    dim_diffs = np.abs(query_fp - rfp_top)
    top_dims  = np.argsort(dim_diffs)[::-1][:5]

    # Gene context for this window
    nearby_genes = gene_ann.genes_near(chrom, win_start, win_end, MAX_GENE_DIST)[:5] if gene_ann.loaded else []
    cancer_genes = [g for g in nearby_genes if g.get("cancer_role")]

    return {
        "window":              f"{chrom}:{win_start:,}-{win_end:,}",
        "most_similar":        {"sample": most_similar[0],   "score": most_similar[1]},
        "most_divergent":      {"sample": most_divergent[0], "score": most_divergent[1]},
        "all_similarities":    dict(sorted_refs),
        "top_driver_dims":     top_dims.tolist(),
        "dim_diffs":           dim_diffs[top_dims].tolist(),
        "nearby_cancer_genes": cancer_genes,
        "window_classification": _classify_window(nearby_genes),
    }


# ── 5. QC flags ───────────────────────────────────────────────────────────────

def qc_flag_report(
    chrom:     str,
    win_start: int,
    win_end:   int,
    similarity: float,
    query_comp: Optional[np.ndarray] = None,
) -> Dict:
    """
    Flag potential quality issues that could explain low similarity
    independent of biology.
    """
    flags = []
    chrom_size_approx = {
        "chr21": 46_709_983,
        "chr22": 50_818_468,
        "chr1":  248_956_422,
    }

    # Telomere risk
    chrom_size = chrom_size_approx.get(chrom, 250_000_000)
    if win_start < QC_TELOMERE_BPS:
        flags.append({
            "type":        "telomere_proximal",
            "severity":    "medium",
            "description": f"Window near telomere ({win_start//1_000_000}Mb from start) — low coverage risk",
            "recommendation": "Exclude or increase sequencing depth",
        })
    if win_end > chrom_size - QC_TELOMERE_BPS:
        flags.append({
            "type":        "telomere_proximal",
            "severity":    "medium",
            "description": f"Window near telomere end — low coverage risk",
            "recommendation": "Exclude or increase sequencing depth",
        })

    # Centromere risk
    for cname, (cs, ce) in [("chr21", CENTROMERE_CHR21), ("chr22", CENTROMERE_CHR22)]:
        if chrom == cname:
            overlap = min(win_end, ce) - max(win_start, cs)
            if overlap > 0:
                flags.append({
                    "type":        "centromere_overlap",
                    "severity":    "high",
                    "description": f"Window overlaps centromere ({overlap//1000}kb overlap) — unmappable regions",
                    "recommendation": "Exclude from analysis",
                })

    # Very low compartment signal — may indicate flat/uninformative region
    if query_comp is not None and len(query_comp) > 0:
        comp_std = float(np.std(query_comp))
        if comp_std < 0.05:
            flags.append({
                "type":        "flat_compartment",
                "severity":    "low",
                "description": f"Flat compartment signal (std={comp_std:.3f}) — may be unmappable",
                "recommendation": "Check coverage in this region",
            })

    return {
        "window":   f"{chrom}:{win_start:,}-{win_end:,}",
        "n_flags":  len(flags),
        "flags":    flags,
        "is_artifact": any(f["severity"] == "high" for f in flags),
    }


# ── 6. Clinical priority summary ─────────────────────────────────────────────

def clinical_priority_summary(
    ccre_report:       Dict,
    boundary_report:   Dict,
    compartment_report:Dict,
    qc_report:         Dict,
    similarity:        float,
    p_value:           Optional[float],
    chrom:             str,
    win_start:         int,
    win_end:           int,
) -> Dict:
    """
    Aggregate all findings into a prioritised clinical summary.

    Actionability score 0-10:
      - 10: Cancer gene directly affected, high confidence
      -  7: Cancer gene nearby, statistically significant
      -  5: Regulatory change, moderate confidence
      -  2: Interesting but not actionable
      -  0: Likely artifact
    """
    if qc_report.get("is_artifact"):
        return {
            "actionability_score": 0,
            "priority": "ARTIFACT",
            "summary": "Likely technical artifact — exclude from analysis",
            "findings": [],
        }

    findings = []
    score = 0

    # cCRE changes near cancer genes
    for cat, info in ccre_report.items():
        if info.get("cancer_genes_nearby") and abs(info.get("delta", 0)) >= 5:
            genes = info["cancer_genes_nearby"]
            direction = info["direction"]
            note = info.get("clinical_note", "")
            findings.append({
                "type":          "regulatory_change",
                "category":      cat,
                "genes":         genes,
                "delta":         info["delta"],
                "direction":     direction,
                "clinical_note": note,
                "priority":      "HIGH" if any(g.upper() in ONCOGENES for g in genes) else "MEDIUM",
            })
            score += 3 if any(g.upper() in ONCOGENES for g in genes) else 1

    # Boundary disruptions near cancer genes
    for b in boundary_report.get("lost_boundaries", []):
        if b.get("cancer_genes"):
            findings.append({
                "type":          "boundary_loss",
                "position_bp":   b["position_bp"],
                "genes":         b["cancer_genes"],
                "clinical_note": b.get("clinical_note"),
                "priority":      "HIGH",
            })
            score += 3

    # Compartment switches affecting cancer genes
    for sw in compartment_report.get("switches", []):
        if sw.get("cancer_genes"):
            findings.append({
                "type":          "compartment_switch",
                "direction":     sw["direction"],
                "genes":         sw["cancer_genes"],
                "consequence":   sw.get("functional_consequence"),
                "clinical_note": sw.get("clinical_note"),
                "priority":      "HIGH",
            })
            score += 4 if sw["direction"] == "B->A" else 2

    # Statistical significance boost
    if p_value is not None and p_value < 0.05:
        score = int(score * 1.5)

    # Similarity penalty
    if similarity > 0.7:
        score = max(0, score - 3)

    score = min(10, score)

    # Priority label
    if score >= 7:
        priority = "CRITICAL"
    elif score >= 5:
        priority = "HIGH"
    elif score >= 3:
        priority = "MEDIUM"
    elif score >= 1:
        priority = "LOW"
    else:
        priority = "INFORMATIONAL"

    # Build summary text
    if findings:
        top = sorted(findings, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("priority", "LOW"), 3))
        top_finding = top[0]
        genes_str = ", ".join(top_finding.get("genes", [])[:3])
        summary = (f"{top_finding['type'].replace('_', ' ').title()} "
                   f"affecting {genes_str}" if genes_str else
                   f"{top_finding['type'].replace('_', ' ').title()} detected")
        if top_finding.get("clinical_note"):
            summary += f" — {top_finding['clinical_note']}"
    else:
        summary = f"No high-priority findings (similarity={similarity:.3f})"

    return {
        "actionability_score": score,
        "priority":   priority,
        "summary":    summary,
        "n_findings": len(findings),
        "findings":   findings,
        "locus":      f"{chrom}:{win_start:,}-{win_end:,}",
        "similarity": similarity,
        "p_value":    p_value,
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _ccre_clinical_note(cat: str, delta: int, cancer_genes: List[str]) -> Optional[str]:
    if not cancer_genes:
        if cat == "PLS" and delta > 10:
            return "Promoter gain — potential gene activation"
        if cat in ("pELS", "dELS") and delta > 20:
            return "Enhancer gain — potential long-range activation"
        if cat == "CA-CTCF" and delta < -10:
            return "CTCF loss — TAD boundary disruption risk"
        return None

    gene = cancer_genes[0].upper()
    direction = "gain" if delta > 0 else "loss"
    if cat == "PLS":
        if gene in ONCOGENES and delta > 0:
            return f"Promoter gain near oncogene {gene} — activation risk"
        if gene in TUMOR_SUPPRESSORS and delta < 0:
            return f"Promoter loss near tumor suppressor {gene} — silencing risk"
        return f"Promoter {direction} near {gene}"
    if cat in ("pELS", "dELS"):
        if gene in ONCOGENES and delta > 0:
            return f"Enhancer gain near {gene} — oncogene activation risk"
        if gene in TUMOR_SUPPRESSORS and delta < 0:
            return f"Enhancer loss near {gene} — tumor suppressor dysregulation"
        return f"Enhancer {direction} near cancer gene {gene}"
    if cat == "CA-CTCF":
        if delta < 0:
            return f"CTCF loss near {gene} — domain boundary disruption"
        return f"CTCF gain near {gene} — new insulation boundary"
    if cat == "High-H3K27ac":
        if delta > 0:
            return f"Active enhancer gain near {gene}"
    return f"{cat} {direction} near {gene}"


def _classify_window(nearby_genes: List[Dict]) -> str:
    if not nearby_genes:
        return "intergenic"
    cancer = [g for g in nearby_genes if g.get("cancer_role")]
    if cancer:
        roles = {g["cancer_role"] for g in cancer}
        if "oncogene" in roles:
            return "oncogene_proximal"
        if "tumor_suppressor" in roles:
            return "tumor_suppressor_proximal"
        return "cancer_gene_proximal"
    types = {g.get("gene_type") for g in nearby_genes[:3]}
    if "protein_coding" in types:
        return "protein_coding"
    return "non_coding"


# ── Full window specificity report (combines all 6 analyses) ─────────────────

def full_specificity_report(
    chrom:           str,
    win_start:       int,
    win_end:         int,
    query_sample_id: str,
    ref_sample_id:   str,
    query_ccre_idx:  Optional[CcreIndex],
    ref_ccre_idx:    Optional[CcreIndex],
    gene_ann:        GeneAnnotator,
    query_boundary:  Optional[np.ndarray],
    ref_boundary:    Optional[np.ndarray],
    query_comp:      Optional[np.ndarray],
    ref_comp:        Optional[np.ndarray],
    query_fp:        Optional[np.ndarray],
    ref_fps:         Dict[str, np.ndarray],
    similarity:      float,
    p_value:         Optional[float],
    resolution:      int = 100_000,
) -> Dict:
    """
    Run all specificity analyses for one window and return combined report.
    """
    # 1. cCRE coordinates
    ccre_rep = {}
    if query_ccre_idx is not None:
        ccre_rep = ccre_coordinate_report(
            query_ccre_idx, ref_ccre_idx, gene_ann,
            chrom, win_start, win_end,
        )

    # 2. Boundary disruption
    bd_rep = {}
    if query_boundary is not None:
        bd_rep = boundary_disruption_report(
            query_boundary, ref_boundary,
            chrom, win_start, resolution, gene_ann,
        )

    # 3. Compartment switches
    cp_rep = {}
    if query_comp is not None:
        cp_rep = compartment_switch_report(
            query_comp, ref_comp,
            chrom, win_start, resolution, gene_ann,
        )

    # 4. Similarity drivers
    sd_rep = {}
    if query_fp is not None and ref_fps:
        sd_rep = similarity_driver_report(
            query_fp, ref_fps, chrom, win_start, win_end, gene_ann,
        )

    # 5. QC flags
    qc_rep = qc_flag_report(chrom, win_start, win_end, similarity, query_comp)

    # 6. Clinical summary
    cl_rep = clinical_priority_summary(
        ccre_rep, bd_rep, cp_rep, qc_rep,
        similarity, p_value, chrom, win_start, win_end,
    )

    return {
        "locus":           f"{chrom}:{win_start:,}-{win_end:,}",
        "query_sample":    query_sample_id,
        "ref_sample":      ref_sample_id,
        "similarity":      similarity,
        "p_value":         p_value,
        "ccre":            ccre_rep,
        "boundaries":      bd_rep,
        "compartments":    cp_rep,
        "drivers":         sd_rep,
        "qc":              qc_rep,
        "clinical":        cl_rep,
    }
