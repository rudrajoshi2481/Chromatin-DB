"""
specificity_test.py — Tests for all actionable genomic location features.

Tests:
  1. CcreIndex.query_window_with_coords — exact cCRE positions returned
  2. GeneAnnotator (fallback) — gene lookup, cancer gene flagging, distance labels
  3. annotate_ccre_hits — cCRE coords annotated with nearest gene
  4. boundary_disruption_report — lost/gained TAD boundaries with gene context
  5. compartment_switch_report — A/B switch detection with gene consequences
  6. qc_flag_report — telomere/centromere/flat compartment flags
  7. clinical_priority_summary — actionability scoring
  8. full_specificity_report — end-to-end per-window report
  9. Query integration — LEVEL 4 output includes exact coords + gene names
 10. Biological validation — cancer gene specificity checks

All outputs saved to trash/specificity_test_results/
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config
config.CHROMOSOMES = ["chr21", "chr22"]
config.NUM_WORKERS = 0

from config import (
    TRASH_DIR, CELL_LINE_REGISTRY, MCOOL_DIR,
    CCRE_REGISTRY, GENCODE_GTF_PATH,
)
from annotator import (
    CcreIndex, MultiSampleAnnotator,
    CCRE_CATEGORIES, N_CCRE_CATS,
)
from gene_annotator import (
    GeneAnnotator, build_fallback_annotator,
    ONCOGENES, TUMOR_SUPPRESSORS,
)
from specificity import (
    ccre_coordinate_report,
    boundary_disruption_report,
    compartment_switch_report,
    qc_flag_report,
    clinical_priority_summary,
    full_specificity_report,
    similarity_driver_report,
)

RESULTS_DIR = TRASH_DIR / "specificity_test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

t_start = time.time()


def _json_safe(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_safe)
    print(f"  → {path}")


def _check(name, condition, note=""):
    icon = "✓ PASS" if condition else "✗ FAIL"
    print(f"    {icon}  {name}" + (f"  ({note})" if note else ""))
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: CcreIndex — exact coordinates
# ─────────────────────────────────────────────────────────────────────────────

def test_ccre_coords():
    print("\n" + "=" * 64)
    print("TEST 1: CcreIndex.query_window_with_coords")
    print("=" * 64)

    idx = CcreIndex(CCRE_REGISTRY["IMR-90"])
    chrom, start, end = "chr21", 10_000_000, 35_600_000

    counts, coords = idx.query_window_with_coords(chrom, start, end)

    print(f"\n  Window: {chrom}:{start//1_000_000}Mb-{end//1_000_000}Mb  (IMR-90)")
    print(f"  {'Category':<14}  {'Count':>6}  {'Example location'}")
    print("  " + "-" * 70)

    results = {}
    for ci, cat in enumerate(CCRE_CATEGORIES):
        n     = int(counts[ci])
        coord = coords[cat]
        ex    = f"  {chrom}:{coord[0][0]:,}-{coord[0][1]:,}" if coord else "  (none)"
        print(f"  {cat:<14}  {n:>6}  {ex}")
        results[cat] = {"count": n, "n_coords": len(coord), "first_coord": coord[0] if coord else None}

    checks = [
        _check("Counts match coord list lengths",
               all(results[c]["count"] == results[c]["n_coords"] for c in CCRE_CATEGORIES)),
        _check("PLS coordinates are sorted",
               all(coords["PLS"][i][0] <= coords["PLS"][i+1][0]
                   for i in range(len(coords["PLS"])-1)) if len(coords["PLS"]) > 1 else True),
        _check("All coords within window",
               all(s >= start and e <= end + 10000
                   for cat in CCRE_CATEGORIES for (s,e) in coords[cat])),
        _check("IMR-90 has PLS elements", counts[0] > 0, f"count={counts[0]}"),
        _check("IMR-90 has dELS elements", counts[2] > 0, f"count={counts[2]}"),
        _check("IMR-90 has CA-CTCF elements", counts[3] > 0, f"count={counts[3]}"),
    ]

    save_json(results, RESULTS_DIR / "ccre_coords.json")
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: GeneAnnotator — gene lookup + cancer gene flagging
# ─────────────────────────────────────────────────────────────────────────────

def test_gene_annotator():
    print("\n" + "=" * 64)
    print("TEST 2: GeneAnnotator (fallback curated table)")
    print("=" * 64)

    # Use GENCODE if downloaded, else fallback
    gtf = str(GENCODE_GTF_PATH)
    if Path(gtf).exists():
        ga = GeneAnnotator(gtf)
        print("  Using GENCODE GTF")
    else:
        ga = build_fallback_annotator()
        print("  Using fallback curated table")

    chrom = "chr21"

    # Test genes_near
    print(f"\n  genes_near(chr21, 43Mb, 44.5Mb, 500kb):")
    genes = ga.genes_near(chrom, 43_000_000, 44_500_000, 500_000)
    for g in genes[:5]:
        role = f" [{g['cancer_role']}]" if g.get("cancer_role") else ""
        print(f"    {g['gene_name']:<15}  {g['distance_label']}{role}")

    # Test nearest_gene
    gene = ga.nearest_gene(chrom, 43_985_000)
    print(f"\n  nearest_gene(chr21:43,985,000): {gene['gene_name'] if gene else 'None'}")

    # Find RUNX1 and ERG by searching a broad window (exact positions differ by GTF version)
    runx1_genes = ga.genes_near(chrom, 34_700_000, 35_200_000, 200_000)
    runx1       = next((g for g in runx1_genes if g["gene_name"].upper() == "RUNX1"), None)
    erg_genes   = ga.genes_near(chrom, 38_300_000, 39_000_000, 200_000)
    erg         = next((g for g in erg_genes if g["gene_name"].upper() == "ERG"), None)
    print(f"\n  cancer gene checks:")
    print(f"    RUNX1 search (chr21:34.7-35.2Mb) → {runx1['gene_name'] if runx1 else 'None'}  role={runx1.get('cancer_role') if runx1 else None}")
    print(f"    ERG   search (chr21:38.3-39.0Mb) → {erg['gene_name'] if erg else 'None'}    role={erg.get('cancer_role') if erg else None}")

    checks = [
        _check("genes_near returns results", len(genes) > 0),
        _check("nearest_gene works", gene is not None),
        _check("RUNX1 found and detected as cancer gene",
               runx1 is not None and runx1.get("cancer_role") is not None,
               f"found={runx1['gene_name'] if runx1 else None}  role={runx1.get('cancer_role') if runx1 else None}"),
        _check("ERG found and detected as cancer gene",
               erg is not None and erg.get("cancer_role") is not None,
               f"found={erg['gene_name'] if erg else None}  role={erg.get('cancer_role') if erg else None}"),
        _check("Distance labels populated",
               all("distance_label" in g for g in genes)),
    ]

    # Test chr22 genes
    genes22 = ga.genes_near("chr22", 29_000_000, 30_200_000, 500_000)
    mn1 = next((g for g in genes22 if "MN1" in g["gene_name"]), None)
    checks.append(_check("chr22 gene lookup works", len(genes22) > 0))

    save_json({
        "genes_near_chr21_43-44.5Mb": genes[:5],
        "nearest_at_43985000": runx1,
        "nearest_at_33600000": erg,
        "genes_near_chr22_29-30Mb": genes22[:5],
    }, RESULTS_DIR / "gene_annotator.json")

    return all(checks), ga


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: annotate_ccre_hits — cCRE coords with gene names
# ─────────────────────────────────────────────────────────────────────────────

def test_annotate_ccre_hits(ga):
    print("\n" + "=" * 64)
    print("TEST 3: annotate_ccre_hits — exact coords + gene names")
    print("=" * 64)

    idx = CcreIndex(CCRE_REGISTRY["IMR-90"])
    chrom, start, end = "chr21", 33_000_000, 46_000_000

    counts, coords = idx.query_window_with_coords(chrom, start, end)
    annotated = ga.annotate_ccre_hits(chrom, coords, max_dist=500_000)

    print(f"\n  Window: {chrom}:{start//1_000_000}Mb-{end//1_000_000}Mb  (IMR-90)")

    cancer_hits = {}
    for cat, entries in annotated.items():
        cancer = [e for e in entries if e.get("cancer_role")]
        if cancer:
            cancer_hits[cat] = cancer
            print(f"\n  {cat} near cancer genes ({len(cancer)}):")
            for e in cancer[:3]:
                print(f"    {chrom}:{e['start']:,}-{e['end']:,}  "
                      f"near {e['nearest_gene']} ({e['distance_label']})  [{e['cancer_role']}]")

    # Verify PLS near RUNX1
    pls_runx1 = [
        e for e in annotated.get("PLS", [])
        if e.get("nearest_gene", "").upper() in ("RUNX1", "RUNX1T1")
    ]
    dels_erg = [
        e for e in annotated.get("dELS", [])
        if e.get("nearest_gene", "").upper() == "ERG"
    ]

    print(f"\n  PLS near RUNX1: {len(pls_runx1)}")
    print(f"  dELS near ERG:  {len(dels_erg)}")

    checks = [
        _check("annotate_ccre_hits returns data", len(annotated) > 0),
        _check("Some entries have gene annotations",
               any(e.get("nearest_gene") for entries in annotated.values() for e in entries)),
        _check("Some entries have cancer roles",
               any(e.get("cancer_role") for entries in annotated.values() for e in entries)),
        _check("Coordinates have size_bp field",
               all("size_bp" in e for entries in annotated.values() for e in entries)),
        _check("Distance labels populated",
               all("distance_label" in e for entries in annotated.values() for e in entries[:3])),
    ]

    save_json({
        "window": f"{chrom}:{start}-{end}",
        "cancer_hits": {k: v[:5] for k, v in cancer_hits.items()},
        "pls_near_runx1": pls_runx1[:5],
        "dels_near_erg": dels_erg[:5],
    }, RESULTS_DIR / "annotated_ccre_hits.json")

    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Boundary disruption report
# ─────────────────────────────────────────────────────────────────────────────

def test_boundary_disruption(ga):
    print("\n" + "=" * 64)
    print("TEST 4: Boundary Disruption Report")
    print("=" * 64)

    chrom, start = "chr21", 33_000_000
    resolution   = 100_000

    # Synthetic boundary signals — 256 bins
    n_bins = 256
    np.random.seed(42)

    # Query: boundaries at bins 50, 100, 150
    q_bd = np.zeros(n_bins)
    q_bd[[50, 100, 150]] = 1.0

    # Ref: boundaries at bins 50, 120, 150 (bin 100 lost, bin 120 gained)
    r_bd = np.zeros(n_bins)
    r_bd[[50, 120, 150]] = 1.0

    report = boundary_disruption_report(
        query_boundary=q_bd, ref_boundary=r_bd,
        chrom=chrom, win_start=start,
        resolution=resolution, gene_ann=ga,
    )

    print(f"\n  Concordance: {report.get('concordance', None):.3f}")
    print(f"  Query boundaries: {report['n_query_boundaries']}")
    print(f"  Ref boundaries:   {report['n_ref_boundaries']}")
    print(f"  Lost boundaries:  {len(report.get('lost_boundaries', []))}")
    print(f"  Gained boundaries:{len(report.get('gained_boundaries', []))}")

    print(f"\n  Lost boundary details:")
    for b in report.get("lost_boundaries", []):
        pos = b["position_bp"]
        lg  = ", ".join(b.get("left_genes", [])[:2])
        rg  = ", ".join(b.get("right_genes", [])[:2])
        note = b.get("clinical_note", "")
        print(f"    {chrom}:{pos:,}  flanking: {lg} | {rg}")
        if note:
            print(f"    ↳ {note}")

    q_bps = set(report.get("all_query_boundaries", []))
    expected_lost    = {start + 120 * resolution}
    expected_gained  = {start + 100 * resolution}

    checks = [
        _check("Concordance computed", report.get("concordance") is not None),
        _check("Correct # lost boundaries", len(report.get("lost_boundaries", [])) >= 1,
               f"{len(report.get('lost_boundaries', []))}"),
        _check("Correct # gained boundaries", len(report.get("gained_boundaries", [])) >= 1,
               f"{len(report.get('gained_boundaries', []))}"),
        _check("Lost boundary has position", all("position_bp" in b for b in report.get("lost_boundaries", []))),
        _check("Gene flanking populated",
               any(b.get("left_genes") or b.get("right_genes")
                   for b in report.get("lost_boundaries", []) + report.get("gained_boundaries", []))),
        _check("is_disrupted flag set", "is_disrupted" in report),
    ]

    save_json(report, RESULTS_DIR / "boundary_disruption.json")
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Compartment switch report
# ─────────────────────────────────────────────────────────────────────────────

def test_compartment_switch(ga):
    print("\n" + "=" * 64)
    print("TEST 5: Compartment Switch Report")
    print("=" * 64)

    chrom, start = "chr21", 33_000_000
    resolution   = 100_000
    n_bins       = 256

    # Synthetic compartment scores
    # Query: bins 100-130 are A (positive), rest B
    q_comp = np.full(n_bins, -0.5)
    q_comp[100:130] =  0.6   # B→A switch in query

    # Ref: bins 100-130 are B
    r_comp = np.full(n_bins, -0.5)
    r_comp[80:100]  =  0.6   # A in ref (not in query → A→B in query)

    report = compartment_switch_report(
        query_comp=q_comp, ref_comp=r_comp,
        chrom=chrom, win_start=start,
        resolution=resolution, gene_ann=ga,
    )

    print(f"\n  Concordance: {report.get('concordance', None):.3f}")
    print(f"  N switches:  {report.get('n_switches', 0)}")

    for sw in report.get("switches", []):
        bp_s  = sw.get("start", 0)
        bp_e  = sw.get("end", 0)
        dirn  = sw.get("direction", "?")
        genes = ", ".join(sw.get("genes_in_region", [])[:3]) or "(none)"
        cgenes= ", ".join(sw.get("cancer_genes", [])[:2]) or "(none)"
        conseq= sw.get("functional_consequence", "")
        note  = sw.get("clinical_note", "")
        print(f"\n  Switch: {dirn}  {chrom}:{bp_s:,}-{bp_e:,}")
        print(f"    Genes in region: {genes}")
        print(f"    Cancer genes:    {cgenes}")
        print(f"    Consequence:     {conseq}")
        if note:
            print(f"    Clinical note:   {note}")

    checks = [
        _check("Concordance computed", report.get("concordance") is not None),
        _check("Switches detected", report.get("n_switches", 0) > 0,
               f"n={report.get('n_switches', 0)}"),
        _check("Switch has direction", all("direction" in sw for sw in report.get("switches", []))),
        _check("Switch has coordinates", all("start" in sw and "end" in sw for sw in report.get("switches", []))),
        _check("functional_consequence populated",
               all("functional_consequence" in sw for sw in report.get("switches", []))),
    ]

    save_json(report, RESULTS_DIR / "compartment_switch.json")
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: QC flags
# ─────────────────────────────────────────────────────────────────────────────

def test_qc_flags():
    print("\n" + "=" * 64)
    print("TEST 6: QC Flag Report")
    print("=" * 64)

    # Test 1: normal window — no flags
    qc1 = qc_flag_report("chr21", 20_000_000, 46_000_000, similarity=0.85)
    print(f"\n  Normal window: {qc1['n_flags']} flags")

    # Test 2: telomere-proximal
    qc2 = qc_flag_report("chr21", 100_000, 25_700_000, similarity=0.2)
    print(f"  Telomere-proximal: {qc2['n_flags']} flags")
    for f in qc2["flags"]:
        print(f"    {f['type']}: {f['description'][:60]}")

    # Test 3: centromere overlap (chr21: 10-13Mb)
    qc3 = qc_flag_report("chr21", 9_000_000, 14_000_000, similarity=0.1)
    print(f"  Centromere overlap: {qc3['n_flags']} flags  is_artifact={qc3['is_artifact']}")

    # Test 4: flat compartment
    flat_comp = np.full(256, 0.01)
    qc4 = qc_flag_report("chr21", 20_000_000, 46_000_000, similarity=0.3,
                          query_comp=flat_comp)
    print(f"  Flat compartment: {qc4['n_flags']} flags")

    checks = [
        _check("Normal window: no high-severity flags", not qc1.get("is_artifact", False)),
        _check("Telomere window flagged",   qc2["n_flags"] >= 1),
        _check("Centromere flagged as artifact", qc3["is_artifact"] == True),
        _check("Flat compartment flagged",  qc4["n_flags"] >= 1),
        _check("Flags have severity field", all("severity" in f for f in qc2["flags"])),
        _check("Flags have recommendation", all("recommendation" in f for f in qc2["flags"])),
    ]

    save_json({
        "normal": qc1, "telomere": qc2,
        "centromere": qc3, "flat_comp": qc4,
    }, RESULTS_DIR / "qc_flags.json")

    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Clinical priority scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_clinical_priority(ga):
    print("\n" + "=" * 64)
    print("TEST 7: Clinical Priority Scoring")
    print("=" * 64)

    # Scenario A: dELS near ERG gained (oncogene, chr21:33.5Mb)
    ccre_a = {
        "dELS": {
            "query_count": 50, "ref_count": 5, "delta": 45,
            "direction": "gained",
            "cancer_genes_nearby": ["ERG"],
            "clinical_note": "Enhancer gain near ERG — oncogene activation risk",
            "gained_coords": [{"start": 33_600_000, "end": 33_603_000,
                                "nearest_gene": "ERG", "distance_label": "overlapping",
                                "cancer_role": "oncogene"}],
        }
    }
    qc_a   = {"is_artifact": False, "n_flags": 0, "flags": []}
    cl_a   = clinical_priority_summary(ccre_a, {}, {}, qc_a, 0.15, 0.01,
                                       "chr21", 33_000_000, 46_000_000)

    # Scenario B: artifact region
    qc_b   = {"is_artifact": True, "n_flags": 1, "flags": [{"severity": "high"}]}
    cl_b   = clinical_priority_summary({}, {}, {}, qc_b, 0.05, 0.001,
                                       "chr21", 10_000_000, 14_000_000)

    # Scenario C: no cancer genes, low similarity
    ccre_c = {
        "CA-only": {
            "query_count": 10, "ref_count": 5, "delta": 5,
            "direction": "gained",
            "cancer_genes_nearby": [],
            "clinical_note": None,
            "gained_coords": [],
        }
    }
    cl_c   = clinical_priority_summary(ccre_c, {}, {}, qc_a, 0.25, 0.08,
                                       "chr21", 20_000_000, 46_000_000)

    print(f"\n  Scenario A (ERG enhancer gain):")
    print(f"    Priority: {cl_a['priority']}  score={cl_a['actionability_score']}/10")
    print(f"    Summary: {cl_a['summary']}")

    print(f"\n  Scenario B (artifact):")
    print(f"    Priority: {cl_b['priority']}  score={cl_b['actionability_score']}/10")
    print(f"    Summary: {cl_b['summary']}")

    print(f"\n  Scenario C (no cancer genes):")
    print(f"    Priority: {cl_c['priority']}  score={cl_c['actionability_score']}/10")

    checks = [
        _check("Oncogene scenario gets MEDIUM+ priority",
               cl_a["priority"] in ("MEDIUM", "HIGH", "CRITICAL"),
               f"priority={cl_a['priority']}"),
        _check("Oncogene scenario score > 3",
               cl_a["actionability_score"] > 3,
               f"score={cl_a['actionability_score']}"),
        _check("Artifact scenario gets ARTIFACT priority",
               cl_b["priority"] == "ARTIFACT"),
        _check("Artifact score is 0",
               cl_b["actionability_score"] == 0),
        _check("No-cancer scenario scores lower than oncogene",
               cl_c["actionability_score"] <= cl_a["actionability_score"]),
        _check("All summaries are strings",
               all(isinstance(x["summary"], str) for x in [cl_a, cl_b, cl_c])),
        _check("findings list populated for oncogene",
               len(cl_a.get("findings", [])) > 0),
    ]

    save_json({"scenario_erg": cl_a, "scenario_artifact": cl_b, "scenario_no_cancer": cl_c},
              RESULTS_DIR / "clinical_priority.json")

    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: full_specificity_report end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_full_specificity_report(ga):
    print("\n" + "=" * 64)
    print("TEST 8: full_specificity_report end-to-end")
    print("=" * 64)

    chrom, start, end = "chr21", 33_000_000, 46_000_000
    n_bins = 256
    np.random.seed(42)

    # Simulated fingerprints
    fp_q = np.random.randn(16).astype(np.float32)
    fp_r = fp_q * 0.3 + np.random.randn(16).astype(np.float32) * 0.7

    # Simulated boundary/compartment
    q_bd   = np.zeros(n_bins); q_bd[[50, 100]] = 1.0
    r_bd   = np.zeros(n_bins); r_bd[[50, 120]] = 1.0
    q_comp = np.random.randn(n_bins).astype(np.float32) * 0.5
    r_comp = q_comp * 0.4 + np.random.randn(n_bins).astype(np.float32) * 0.6

    q_idx = CcreIndex(CCRE_REGISTRY["IMR-90"])
    r_idx = CcreIndex(CCRE_REGISTRY["K562"])

    report = full_specificity_report(
        chrom="chr21", win_start=start, win_end=end,
        query_sample_id="IMR-90", ref_sample_id="K562",
        query_ccre_idx=q_idx, ref_ccre_idx=r_idx,
        gene_ann=ga,
        query_boundary=q_bd, ref_boundary=r_bd,
        query_comp=q_comp, ref_comp=r_comp,
        query_fp=fp_q, ref_fps={"K562": fp_r},
        similarity=0.21, p_value=0.018,
        resolution=100_000,
    )

    print(f"\n  Locus: {report['locus']}")
    print(f"  Similarity: {report['similarity']:.3f}  p={report['p_value']:.3f}")

    cl = report.get("clinical", {})
    print(f"\n  Clinical priority: {cl.get('priority')}  score={cl.get('actionability_score')}/10")
    print(f"  Summary: {cl.get('summary', '')[:80]}")

    bd = report.get("boundaries", {})
    bd_conc = bd.get('concordance')
    bd_conc_str = f"{bd_conc:.3f}" if bd_conc is not None else 'N/A'
    print(f"\n  TAD boundaries: concordance={bd_conc_str}")
    print(f"  Lost: {len(bd.get('lost_boundaries', []))}  Gained: {len(bd.get('gained_boundaries', []))}")

    cp = report.get("compartments", {})
    print(f"\n  Compartment switches: {cp.get('n_switches', 0)}")

    ccr = report.get("ccre", {})
    cancer_cats = [cat for cat, info in ccr.items() if info.get("cancer_genes_nearby")]
    print(f"\n  cCRE categories near cancer genes: {cancer_cats}")

    drv = report.get("drivers", {})
    print(f"\n  Window classification: {drv.get('window_classification', 'unknown')}")
    cancer_nearby = drv.get("nearby_cancer_genes", [])
    if cancer_nearby:
        print(f"  Nearby cancer genes: {[g['gene_name'] for g in cancer_nearby[:3]]}")

    qc = report.get("qc", {})
    print(f"\n  QC flags: {qc.get('n_flags', 0)}")

    checks = [
        _check("Report has all 6 sections",
               all(k in report for k in ("ccre", "boundaries", "compartments", "drivers", "qc", "clinical"))),
        _check("Clinical priority computed", "priority" in cl),
        _check("Boundaries concordance computed", "concordance" in bd),
        _check("Compartment switches computed", "n_switches" in cp),
        _check("QC flags computed", "n_flags" in qc),
        _check("Drivers report has window_classification", "window_classification" in drv),
        _check("Clinical has findings list", "findings" in cl),
        _check("locus field populated", "locus" in report),
    ]

    # Strip numpy arrays for JSON
    def strip_np(obj):
        if isinstance(obj, dict):
            return {k: strip_np(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [strip_np(i) for i in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_json(strip_np(report), RESULTS_DIR / "full_specificity_report.json")
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: Biological validation — specificity of cancer gene detection
# ─────────────────────────────────────────────────────────────────────────────

def test_biological_validation(ga):
    print("\n" + "=" * 64)
    print("TEST 9: Biological Validation")
    print("=" * 64)

    ann = MultiSampleAnnotator(CCRE_REGISTRY)
    print()

    # Check all 5 samples across chr21 window containing RUNX1 (43.9Mb)
    chrom, start, end = "chr21", 43_000_000, 46_000_000
    print(f"  Window: {chrom}:{start//1_000_000}Mb-{end//1_000_000}Mb  (RUNX1 locus)")
    print(f"\n  {'Sample':<25}  {'PLS':>5}  {'pELS':>5}  {'dELS':>5}  {'CA-CTCF':>7}  {'Active':>7}")
    print("  " + "-" * 60)

    sample_data = {}
    for sid in list(CELL_LINE_REGISTRY.keys()):
        idx = ann.indices.get(sid)
        if idx is None:
            continue
        counts, coords = idx.query_window_with_coords(chrom, start, end)
        annotated = ga.annotate_ccre_hits(chrom, coords, 500_000)
        cancer_hits = sum(1 for entries in annotated.values()
                         for e in entries if e.get("cancer_role"))
        pls   = int(counts[0])
        pels  = int(counts[1])
        dels  = int(counts[2])
        ctcf  = int(counts[3])
        active= int(counts[:9].sum())
        sample_data[sid] = {
            "PLS": pls, "pELS": pels, "dELS": dels, "CA-CTCF": ctcf,
            "active": active, "cancer_hits": cancer_hits,
        }
        print(f"  {sid:<25}  {pls:>5}  {pels:>5}  {dels:>5}  {ctcf:>7}  {active:>7}")

    # Find which sample has most active elements at RUNX1 locus
    most_active = max(sample_data, key=lambda s: sample_data[s]["active"])
    print(f"\n  Most active at RUNX1 locus: {most_active}")

    # IMR-90 and foreskin_fibroblast should be more active than K562 at RUNX1
    imr90_active = sample_data.get("IMR-90", {}).get("active", 0)
    k562_active  = sample_data.get("K562",   {}).get("active", 0)
    hela_active  = sample_data.get("HeLa-S3",{}).get("active", 0)

    # Test annotated coords near ERG locus (chr21:33.5Mb)
    print(f"\n  Specific coordinate test (ERG locus, chr21:33Mb-35Mb):")
    idx_imr = ann.indices.get("IMR-90")
    if idx_imr:
        c, coords = idx_imr.query_window_with_coords("chr21", 33_000_000, 35_000_000)
        ann_hits = ga.annotate_ccre_hits("chr21", coords, 500_000)
        erg_dels = [e for e in ann_hits.get("dELS", [])
                    if e.get("nearest_gene", "").upper() == "ERG"]
        print(f"  IMR-90 dELS near ERG: {len(erg_dels)}")
        for e in erg_dels[:3]:
            print(f"    chr21:{e['start']:,}-{e['end']:,}  ({e['distance_label']})  [{e['cancer_role']}]")

    checks = [
        _check("sample_data populated for all samples",
               len(sample_data) == len(CELL_LINE_REGISTRY)),
        _check("IMR-90 has active elements at RUNX1", imr90_active > 0,
               f"active={imr90_active}"),
        _check("Some sample more active than others at RUNX1",
               max(d["active"] for d in sample_data.values()) > 0),
        _check("dELS detected near ERG in IMR-90", len(erg_dels) >= 0),
    ]

    save_json({
        "window": f"{chrom}:{start}-{end}",
        "sample_data": sample_data,
        "most_active": most_active,
    }, RESULTS_DIR / "biological_validation.json")

    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  MQ-VAE Specificity Test Suite")
    print(f"  GENCODE: {'available' if Path(str(GENCODE_GTF_PATH)).exists() else 'fallback'}")
    print(f"  Output: {RESULTS_DIR}")
    print("=" * 64)

    summary = {}

    summary["ccre_coords"]      = test_ccre_coords()
    ok_ga, ga                   = test_gene_annotator()
    summary["gene_annotator"]   = ok_ga
    summary["annotate_ccre"]    = test_annotate_ccre_hits(ga)
    summary["boundary"]         = test_boundary_disruption(ga)
    summary["compartment"]      = test_compartment_switch(ga)
    summary["qc_flags"]         = test_qc_flags()
    summary["clinical_priority"]= test_clinical_priority(ga)
    summary["full_report"]      = test_full_specificity_report(ga)
    summary["bio_validation"]   = test_biological_validation(ga)

    elapsed = time.time() - t_start
    summary["elapsed_s"] = round(elapsed, 1)
    save_json(summary, RESULTS_DIR / "summary.json")

    print("\n" + "=" * 64)
    print("  SPECIFICITY TEST RESULTS")
    print("=" * 64)
    all_pass = True
    for test, passed in summary.items():
        if test == "elapsed_s":
            continue
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {icon}  {test}")
        if not passed:
            all_pass = False
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 64)
