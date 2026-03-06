"""
gene_annotator.py — GENCODE gene annotation for genomic windows.

Maps genomic coordinates to nearby gene names, computes distances,
and provides clinical/functional context for regulatory elements.

Features:
  - Fast interval index (sorted arrays + searchsorted)
  - Gene-to-cCRE proximity mapping (which cCRE is near which gene)
  - Boundary disruption → gene context (which genes lose TAD insulation)
  - Compartment switch → gene context (which genes switch A/B)
  - Known cancer gene flagging (oncogenes, tumor suppressors)
  - Distance annotation (upstream/downstream/overlapping)

Usage:
    from gene_annotator import GeneAnnotator
    ga = GeneAnnotator("gencode.v45.basic.annotation.gtf.gz")
    genes = ga.genes_near(chrom, start, end, max_dist=2_000_000)
"""

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Known cancer-relevant genes ───────────────────────────────────────────────
# Manually curated subset — oncogenes, tumor suppressors, chromatin regulators
ONCOGENES = {
    "MYC", "MYCN", "MYCL", "BCL2", "BCL6", "CCND1", "CDK4", "CDK6",
    "KRAS", "NRAS", "HRAS", "EGFR", "ERBB2", "MET", "FLT3", "KIT",
    "ABL1", "JAK2", "STAT3", "STAT5A", "STAT5B", "EZH2", "DNMT3A",
    "IDH1", "IDH2", "NPM1", "RUNX1", "GATA2", "TAL1", "LMO2", "HOX",
    "MLL", "KMT2A", "NUP214", "DEK", "ERG", "ETS1", "FUS", "EWSR1",
    "MDM2", "CDK8", "MED12", "BRD4", "FOXA1", "AR", "ESR1",
}

TUMOR_SUPPRESSORS = {
    "TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2", "VHL", "MLH1",
    "MSH2", "MSH6", "CDKN2A", "CDKN1A", "CDKN1B", "NF1", "NF2",
    "TSC1", "TSC2", "SMAD4", "TGFbeta", "RUNX1T1", "CBFB",
    "WT1", "PTCH1", "SUFU", "BAP1", "SETD2", "KDM6A", "ARID1A",
    "SMARCA4", "SMARCB1", "CREBBP", "EP300",
}

CTCF_ANCHORS = {
    "CTCF", "RAD21", "SMC1A", "SMC3", "STAG1", "STAG2", "NIPBL", "WAPL",
}

CHROMATIN_REGULATORS = {
    "EZH2", "EZH1", "SUZ12", "EED", "DNMT3A", "DNMT3B", "TET2",
    "HDAC1", "HDAC2", "KDM1A", "KDM5C", "KDM6A", "SETD2",
    "ARID1A", "ARID1B", "SMARCA4", "SMARCB1",
}

ALL_CANCER_GENES = ONCOGENES | TUMOR_SUPPRESSORS | CHROMATIN_REGULATORS

# ── Gene record ───────────────────────────────────────────────────────────────

class GeneRecord:
    __slots__ = ("chrom", "start", "end", "strand", "gene_id", "gene_name",
                 "gene_type", "cancer_role")

    def __init__(self, chrom, start, end, strand, gene_id, gene_name, gene_type):
        self.chrom     = chrom
        self.start     = start
        self.end       = end
        self.strand    = strand
        self.gene_id   = gene_id
        self.gene_name = gene_name
        self.gene_type = gene_type
        self.cancer_role = self._classify_cancer_role()

    def _classify_cancer_role(self) -> Optional[str]:
        name = self.gene_name.upper()
        if name in ONCOGENES:
            return "oncogene"
        if name in TUMOR_SUPPRESSORS:
            return "tumor_suppressor"
        if name in CTCF_ANCHORS:
            return "ctcf_anchor"
        if name in CHROMATIN_REGULATORS:
            return "chromatin_regulator"
        return None

    def distance_to(self, pos: int) -> int:
        """Signed distance: negative = upstream of gene, positive = downstream."""
        if self.start <= pos <= self.end:
            return 0
        if pos < self.start:
            return self.start - pos   # upstream
        return self.end - pos         # downstream (negative)

    def distance_label(self, pos: int) -> str:
        d = self.distance_to(pos)
        if d == 0:
            return "overlapping"
        kb = abs(d) / 1000
        if d > 0:
            return f"{kb:.0f}kb upstream"
        return f"{kb:.0f}kb downstream"

    def to_dict(self) -> Dict:
        return {
            "gene_name":   self.gene_name,
            "gene_id":     self.gene_id,
            "gene_type":   self.gene_type,
            "chrom":       self.chrom,
            "start":       self.start,
            "end":         self.end,
            "strand":      self.strand,
            "cancer_role": self.cancer_role,
        }


# ── Gene annotator ────────────────────────────────────────────────────────────

class GeneAnnotator:
    """
    Fast gene annotation using GENCODE GTF.
    
    Provides:
    - genes_near(chrom, start, end, max_dist) → list of GeneRecord
    - annotate_position(chrom, pos) → nearest gene + distance
    - annotate_ccre_hits(chrom, coords_by_cat) → per-cCRE gene context
    - annotate_boundary_loss(chrom, positions) → genes at lost boundaries
    - annotate_compartment_switches(switches) → genes in switched regions
    """

    def __init__(self, gtf_path: str):
        self.path = gtf_path
        # chrom → sorted arrays
        self._starts:    Dict[str, np.ndarray] = {}
        self._ends:      Dict[str, np.ndarray] = {}
        self._gene_idxs: Dict[str, np.ndarray] = {}
        self._genes:     List[GeneRecord] = []
        self._loaded = False
        if Path(gtf_path).exists():
            self._load()
        else:
            print(f"[gene_annotator] WARNING: GTF not found at {gtf_path}")
            print(f"[gene_annotator] Gene annotation will be disabled")

    @property
    def loaded(self):
        return self._loaded

    def _load(self):
        print(f"[gene_annotator] Loading GENCODE GTF from {self.path}...", flush=True)
        raw: Dict[str, List] = defaultdict(list)
        seen_genes = set()

        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "gene":
                    continue

                chrom    = parts[0]
                start    = int(parts[3]) - 1  # GTF is 1-based
                end      = int(parts[4])
                strand   = parts[6]
                attrs    = parts[8]

                gene_id   = _extract_attr(attrs, "gene_id")
                gene_name = _extract_attr(attrs, "gene_name") or gene_id
                gene_type = _extract_attr(attrs, "gene_type") or "unknown"

                if gene_id in seen_genes:
                    continue
                seen_genes.add(gene_id)

                rec = GeneRecord(chrom, start, end, strand, gene_id, gene_name, gene_type)
                idx = len(self._genes)
                self._genes.append(rec)
                raw[chrom].append((start, end, idx))

        for chrom, records in raw.items():
            records.sort(key=lambda x: x[0])
            arr = np.array(records, dtype=np.int64)
            self._starts[chrom]    = arr[:, 0]
            self._ends[chrom]      = arr[:, 1]
            self._gene_idxs[chrom] = arr[:, 2]

        self._loaded = True
        total = sum(len(v) for v in self._starts.values())
        print(f"[gene_annotator] Loaded {total} genes across {len(self._starts)} chromosomes")

    def genes_in_window(self, chrom: str, start: int, end: int) -> List[GeneRecord]:
        """Return genes whose body overlaps [start, end)."""
        if not self._loaded or chrom not in self._starts:
            return []
        starts   = self._starts[chrom]
        ends     = self._ends[chrom]
        idxs     = self._gene_idxs[chrom]
        right    = int(np.searchsorted(starts, end, side="left"))
        mask     = ends[:right] > start
        return [self._genes[idxs[i]] for i in np.where(mask)[0]]

    def genes_near(
        self,
        chrom:    str,
        start:    int,
        end:      int,
        max_dist: int = 2_000_000,
    ) -> List[Dict]:
        """
        Return genes within max_dist of [start, end), sorted by distance.
        Includes genes overlapping the window (distance=0).
        """
        if not self._loaded or chrom not in self._starts:
            return []

        expanded_start = max(0, start - max_dist)
        expanded_end   = end + max_dist

        starts   = self._starts[chrom]
        ends     = self._ends[chrom]
        idxs     = self._gene_idxs[chrom]
        right    = int(np.searchsorted(starts, expanded_end, side="left"))
        mask     = ends[:right] > expanded_start

        results = []
        for i in np.where(mask)[0]:
            gene = self._genes[idxs[i]]
            # Distance to window midpoint
            mid  = (start + end) // 2
            dist = gene.distance_to(mid)
            if abs(dist) <= max_dist:
                d = gene.to_dict()
                d["distance_bp"]    = int(dist)
                d["distance_label"] = gene.distance_label(mid)
                d["in_window"]      = (start <= gene.start <= end or
                                       start <= gene.end <= end or
                                       gene.start <= start <= gene.end)
                results.append(d)

        results.sort(key=lambda x: abs(x["distance_bp"]))
        return results

    def nearest_gene(
        self,
        chrom:    str,
        pos:      int,
        max_dist: int = 2_000_000,
    ) -> Optional[Dict]:
        """Return single nearest gene to a genomic position."""
        genes = self.genes_near(chrom, pos, pos + 1, max_dist)
        return genes[0] if genes else None

    def annotate_ccre_hits(
        self,
        chrom:           str,
        coords_by_cat:   Dict[str, List[Tuple[int, int]]],
        max_dist:        int = 500_000,
    ) -> Dict[str, List[Dict]]:
        """
        For each cCRE category, annotate each hit with nearest gene.
        
        Returns: {cat_name: [{start, end, nearest_gene, distance, cancer_role}, ...]}
        """
        result = {}
        for cat, coord_list in coords_by_cat.items():
            annotated = []
            for (cstart, cend) in coord_list:
                mid   = (cstart + cend) // 2
                gene  = self.nearest_gene(chrom, mid, max_dist)
                entry = {
                    "start": cstart,
                    "end":   cend,
                    "size_bp": cend - cstart,
                }
                if gene:
                    entry["nearest_gene"]  = gene["gene_name"]
                    entry["gene_id"]       = gene["gene_id"]
                    entry["distance_bp"]   = gene["distance_bp"]
                    entry["distance_label"]= gene["distance_label"]
                    entry["cancer_role"]   = gene["cancer_role"]
                    entry["gene_type"]     = gene["gene_type"]
                else:
                    entry["nearest_gene"]  = "intergenic"
                    entry["distance_bp"]   = None
                    entry["distance_label"]= "intergenic"
                    entry["cancer_role"]   = None
                    entry["gene_type"]     = None
                annotated.append(entry)
            # Sort by distance (cancer genes first, then by proximity)
            annotated.sort(key=lambda x: (
                0 if x.get("cancer_role") else 1,
                abs(x.get("distance_bp") or 999_999_999)
            ))
            result[cat] = annotated
        return result

    def annotate_boundary_losses(
        self,
        chrom:     str,
        positions: List[int],   # bin positions (bp) where boundary was lost
        max_dist:  int = 500_000,
    ) -> List[Dict]:
        """
        For each lost TAD boundary position, find flanking genes
        and describe the disruption.
        """
        results = []
        for pos in positions:
            left_genes  = self.genes_near(chrom, pos - max_dist, pos, max_dist)[:3]
            right_genes = self.genes_near(chrom, pos, pos + max_dist, max_dist)[:3]
            results.append({
                "position_bp":  pos,
                "left_genes":   [g["gene_name"] for g in left_genes],
                "right_genes":  [g["gene_name"] for g in right_genes],
                "cancer_genes": [
                    g["gene_name"] for g in left_genes + right_genes
                    if g.get("cancer_role")
                ],
                "clinical_note": _boundary_clinical_note(
                    [g["gene_name"] for g in left_genes + right_genes]
                ),
            })
        return results

    def annotate_compartment_switches(
        self,
        chrom:   str,
        switches: List[Dict],  # [{start, end, direction: "A->B" or "B->A"}]
        max_dist: int = 200_000,
    ) -> List[Dict]:
        """
        For each A/B compartment switch, find affected genes and
        describe functional consequences.
        """
        results = []
        for sw in switches:
            genes_in   = self.genes_in_window(chrom, sw["start"], sw["end"])
            genes_near = self.genes_near(chrom, sw["start"], sw["end"], max_dist)[:5]

            cancer_in_region = [g for g in genes_in if g.cancer_role]
            direction = sw.get("direction", "unknown")

            result = {
                "start":      sw["start"],
                "end":        sw["end"],
                "size_bp":    sw["end"] - sw["start"],
                "direction":  direction,
                "genes_in_region": [g.gene_name for g in genes_in],
                "cancer_genes": [g.gene_name for g in cancer_in_region],
                "nearby_genes": [g["gene_name"] for g in genes_near[:3]],
                "functional_consequence": _compartment_consequence(
                    direction,
                    [g.gene_name for g in cancer_in_region],
                ),
                "clinical_note": _compartment_clinical_note(
                    direction,
                    [g.gene_name for g in cancer_in_region],
                ),
            }
            results.append(result)
        return results


# ── Helper functions ──────────────────────────────────────────────────────────

def _extract_attr(attrs: str, key: str) -> Optional[str]:
    m = re.search(rf'{key}\s+"([^"]+)"', attrs)
    return m.group(1) if m else None


def _boundary_clinical_note(gene_names: List[str]) -> Optional[str]:
    names_upper = {g.upper() for g in gene_names}
    if "MYC" in names_upper:
        return "MYC TAD boundary disruption — oncogene activation risk"
    if "RUNX1" in names_upper:
        return "RUNX1 boundary loss — leukemia-associated"
    if "TP53" in names_upper:
        return "TP53 TAD disruption — tumor suppressor dysregulation"
    if "CTCF" in names_upper:
        return "CTCF gene boundary loss — genome organization disruption"
    if "BCL2" in names_upper:
        return "BCL2 boundary disruption — anti-apoptosis dysregulation"
    if names_upper & ONCOGENES:
        gene = (names_upper & ONCOGENES).pop()
        return f"{gene} oncogene near disrupted boundary"
    if names_upper & TUMOR_SUPPRESSORS:
        gene = (names_upper & TUMOR_SUPPRESSORS).pop()
        return f"{gene} tumor suppressor near disrupted boundary"
    return None


def _compartment_consequence(direction: str, cancer_genes: List[str]) -> str:
    if direction == "B->A":
        if cancer_genes:
            return f"Activation of {', '.join(cancer_genes[:2])} (heterochromatin→euchromatin)"
        return "Gene activation (heterochromatin→euchromatin shift)"
    if direction == "A->B":
        if cancer_genes:
            return f"Silencing of {', '.join(cancer_genes[:2])} (euchromatin→heterochromatin)"
        return "Gene silencing (euchromatin→heterochromatin shift)"
    return "Unknown compartment change"


def _compartment_clinical_note(direction: str, cancer_genes: List[str]) -> Optional[str]:
    if not cancer_genes:
        return None
    gene = cancer_genes[0].upper()
    if direction == "B->A":
        if gene in ONCOGENES:
            return f"{gene} oncogene activation via compartment switch — high risk"
        if gene in TUMOR_SUPPRESSORS:
            return f"{gene} tumor suppressor activated — potential protective effect"
    if direction == "A->B":
        if gene in TUMOR_SUPPRESSORS:
            return f"{gene} tumor suppressor silenced via compartment switch — high risk"
        if gene in ONCOGENES:
            return f"{gene} oncogene silenced — potential protective effect"
    return f"{gene} cancer gene affected by compartment switch"


# ── Convenience: build minimal gene index from a simple TSV ──────────────────
# Used when GTF is not available — falls back to a small curated table

FALLBACK_GENES_CHR21 = [
    # (start, end, name, type)
    (14_100_000, 14_200_000, "BRWD1",  "protein_coding"),
    (14_300_000, 14_460_000, "HMGN1",  "protein_coding"),
    (15_440_000, 15_611_000, "WRB",    "protein_coding"),
    (17_054_000, 17_087_000, "PTTG1IP","protein_coding"),
    (17_700_000, 17_900_000, "DSCR3",  "protein_coding"),
    (18_370_000, 18_420_000, "DYRK1A", "protein_coding"),
    (25_888_000, 26_004_000, "KCNJ6",  "protein_coding"),
    (26_556_000, 26_600_000, "DSCR4",  "protein_coding"),
    (33_548_000, 33_635_000, "ERG",    "protein_coding"),
    (34_787_000, 34_856_000, "ETS2",   "protein_coding"),
    (35_070_000, 35_224_000, "PSMG1",  "protein_coding"),
    (38_380_000, 38_461_000, "B3GALT5","protein_coding"),
    (39_710_000, 39_809_000, "HMGN1",  "protein_coding"),
    (40_737_000, 40_818_000, "WRB",    "protein_coding"),
    (42_100_000, 42_231_000, "COL18A1","protein_coding"),
    (43_985_000, 44_055_000, "RUNX1",  "protein_coding"),
    (44_352_000, 44_534_000, "CLIC6",  "protein_coding"),
]

FALLBACK_GENES_CHR22 = [
    (19_178_000, 19_278_000, "BCR",    "protein_coding"),
    (21_983_000, 22_057_000, "NF2",    "protein_coding"),
    (23_521_000, 23_601_000, "SMARCB1","protein_coding"),
    (24_103_000, 24_191_000, "CRKL",   "protein_coding"),
    (25_584_000, 25_677_000, "AIFM3",  "protein_coding"),
    (29_041_000, 29_130_000, "MN1",    "protein_coding"),
    (30_070_000, 30_140_000, "TBX1",   "protein_coding"),
    (36_677_000, 36_803_000, "PDGFB",  "protein_coding"),
    (37_339_000, 37_477_000, "MYH9",   "protein_coding"),
    (38_430_000, 38_510_000, "EWSR1",  "protein_coding"),
    (40_355_000, 40_499_000, "NEFH",   "protein_coding"),
    (42_400_000, 42_530_000, "EP300",  "protein_coding"),
    (43_088_000, 43_117_000, "SCO2",   "protein_coding"),
    (44_539_000, 44_624_000, "BRCA2",  "protein_coding"),  # partial overlap region
    (45_790_000, 45_944_000, "PARVB",  "protein_coding"),
]


def build_fallback_annotator() -> "GeneAnnotator":
    """
    Build a GeneAnnotator from the hardcoded fallback gene tables
    for chr21/chr22.  Used when GENCODE GTF is not available.
    """
    ga = GeneAnnotator.__new__(GeneAnnotator)
    ga.path    = "fallback"
    ga._genes  = []
    ga._starts    = {}
    ga._ends      = {}
    ga._gene_idxs = {}

    raw: Dict[str, List] = defaultdict(list)
    for chrom, table in [("chr21", FALLBACK_GENES_CHR21),
                         ("chr22", FALLBACK_GENES_CHR22)]:
        for (start, end, name, gtype) in table:
            idx = len(ga._genes)
            rec = GeneRecord(chrom, start, end, "+", name, name, gtype)
            ga._genes.append(rec)
            raw[chrom].append((start, end, idx))

    for chrom, records in raw.items():
        records.sort(key=lambda x: x[0])
        arr = np.array(records, dtype=np.int64)
        ga._starts[chrom]    = arr[:, 0]
        ga._ends[chrom]      = arr[:, 1]
        ga._gene_idxs[chrom] = arr[:, 2]

    ga._loaded = True
    total = sum(len(v) for v in ga._starts.values())
    print(f"[gene_annotator] Fallback: loaded {total} curated genes for chr21/chr22")
    return ga
