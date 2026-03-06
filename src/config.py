"""
config.py — Central configuration for MQ-VAE Hi-C Structural Fingerprinting Database.
All hyperparameters, paths, and constants live here.

Path resolution priority:
  1. Environment variables (HIC_PROJECT_ROOT, HIC_MCOOL_DIR, HIC_CCRE_DIR)
  2. Auto-discovery from standard folder structure
  3. Defaults relative to this file's location

mcool / cCRE discovery:
  Any .mcool file found in MCOOL_DIR is auto-registered.
  Any .bed.gz file found in CCRE_DIR is matched to mcool by sample name prefix.
  Replicates (K562_rep1, K562_rep2) are auto-grouped to K562.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
# Override with HIC_PROJECT_ROOT env var if needed
_src_dir     = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("HIC_PROJECT_ROOT", str(_src_dir.parent)))
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED_DIR   = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TRASH_DIR    = PROJECT_ROOT / "trash"
DOCS_DIR     = PROJECT_ROOT / "docs"
PLOTS_DIR    = TRASH_DIR / "plots"

# ─── Data directories ─────────────────────────────────────────────────────────
# Override with HIC_MCOOL_DIR / HIC_CCRE_DIR env vars
_default_mcool = PROJECT_ROOT.parent / "ink-react" / "chromatin-manager" / "data" / "downloads" / "mcool"
_default_ccre  = PROJECT_ROOT.parent / "ink-react" / "chromatin-manager" / "data" / "downloads" / "ccre"

MCOOL_DIR = Path(os.environ.get("HIC_MCOOL_DIR", str(_default_mcool)))
CCRE_DIR  = Path(os.environ.get("HIC_CCRE_DIR",  str(_default_ccre)))

GENCODE_GTF_PATH = Path(os.environ.get(
    "HIC_GENCODE_GTF",
    str(DATA_DIR / "gencode.v45.basic.annotation.gtf.gz")
))

# DuckDB and FAISS paths
DB_PATH    = DATA_DIR / "hic_fingerprints.duckdb"
FAISS_PATH = DATA_DIR / "locus_centroids.faiss"

# ─── Auto-discover mcool files ────────────────────────────────────────────────
# Scans MCOOL_DIR for all .mcool files and registers them automatically.
# Replicate groups are detected by replicates.py at runtime.
import re as _re

# ENCODE accession patterns to strip from filenames
# e.g. K562_4DNFI18UHVRO → K562
#      IMR-90_4DNFIJTOIGOI → IMR-90
#      HeLa-S3_ENCFF960YUI → HeLa-S3
_ACCESSION_RE = _re.compile(
    r"[_\.]?"
    r"(?:4DNF[A-Z0-9]+|ENCFF[A-Z0-9]+|ENCSR[A-Z0-9]+|GSM\d+|SRR\d+|ERR\d+)"
    r"$",
    _re.IGNORECASE,
)

def _clean_sample_id(stem: str) -> str:
    """Strip ENCODE/4DN accession suffix to get clean sample name."""
    return _ACCESSION_RE.sub("", stem).strip("_.-")


def _discover_mcool(mcool_dir: Path) -> dict:
    if not mcool_dir.exists():
        return {}
    registry = {}
    for p in sorted(mcool_dir.glob("*.mcool")):
        sample_id = _clean_sample_id(p.stem)
        if not sample_id:
            sample_id = p.stem      # fallback: keep full stem
        registry[sample_id] = {
            "file":   p.name,
            "assay":  "bulk_hic",
            "tissue": "unknown",
        }
    return registry

# ─── Auto-discover cCRE files ─────────────────────────────────────────────────
def _discover_ccre(ccre_dir: Path, mcool_registry: dict) -> dict:
    if not ccre_dir.exists():
        return {}
    ccre_files = sorted(ccre_dir.glob("*.bed.gz"))
    registry   = {}
    for sample_id in mcool_registry:
        # Exact match (sample_id.bed.gz)
        exact = ccre_dir / f"{sample_id}.bed.gz"
        if exact.exists():
            registry[sample_id] = str(exact)
            continue
        # Prefix match — bed.gz name starts with sample_id
        matches = [f for f in ccre_files if f.name.startswith(sample_id)]
        if matches:
            registry[sample_id] = str(matches[0])
            continue
        # Clean-name prefix match — strip accession from cCRE filename too
        matches = [
            f for f in ccre_files
            if _clean_sample_id(f.name.replace(".bed.gz", "")) == sample_id
        ]
        if matches:
            registry[sample_id] = str(matches[0])
    return registry

# Build registries
CELL_LINE_REGISTRY = _discover_mcool(MCOOL_DIR)
CCRE_REGISTRY      = _discover_ccre(CCRE_DIR, CELL_LINE_REGISTRY)

# If discovery finds nothing (e.g. different folder structure),
# fall back to manually maintained entries below.
# These are used ONLY when auto-discovery produces an empty result.
_MANUAL_REGISTRY = {
    "HeLa-S3":             {"file": "HeLa-S3_4DNFIBM9QCFG.mcool",             "assay": "bulk_hic", "tissue": "cervical_cancer"},
    "IMR-90":              {"file": "IMR-90_4DNFIJTOIGOI.mcool",               "assay": "bulk_hic", "tissue": "lung_fibroblast"},
    "K562":                {"file": "K562_4DNFI18UHVRO.mcool",                 "assay": "bulk_hic", "tissue": "leukemia"},
    "KBM-7":               {"file": "KBM-7_4DNFI5IHU27G.mcool",               "assay": "bulk_hic", "tissue": "leukemia"},
    "foreskin_fibroblast": {"file": "foreskin_fibroblast_4DNFIQJQY7PW.mcool",  "assay": "bulk_hic", "tissue": "fibroblast"},
}
_MANUAL_CCRE = {
    "HeLa-S3":             str(CCRE_DIR / "HeLa-S3_ENCFF960YUI.bed.gz"),
    "IMR-90":              str(CCRE_DIR / "IMR-90_ENCFF685BXB.bed.gz"),
    "K562":                str(CCRE_DIR / "K562_ENCFF455VKH.bed.gz"),
    "KBM-7":               str(CCRE_DIR / "KBM-7_ENCFF141FBS.bed.gz"),
    "foreskin_fibroblast": str(CCRE_DIR / "foreskin_fibroblast_ENCFF422UQT.bed.gz"),
}

if not CELL_LINE_REGISTRY:
    CELL_LINE_REGISTRY = _MANUAL_REGISTRY
if not CCRE_REGISTRY:
    CCRE_REGISTRY = _MANUAL_CCRE

ASSAY_TYPES = {
    "bulk_hic": 0,
    "micro_c":  1,
    "sc_hic":   2,
    "chia_pet": 3,
}

# ─── Annotation ───────────────────────────────────────────────────────────────
ANNOT_PERM_N          = 500     # permutation test replicates
ANNOT_DIFF_THRESHOLD  = 10      # min cCRE count diff to flag in report
SIM_THRESHOLD_HIGH    = 0.7     # used in query classification too (keep here)
SIM_THRESHOLD_MED     = 0.3

# ─── Preprocessing ────────────────────────────────────────────────────────────
RESOLUTION         = 100_000        # 100 kb bins
TILE_SIZE          = 256            # 256×256 windows
TILE_OVERLAP       = 0.5            # 50% overlap → step = 128 bins
TILE_STEP          = int(TILE_SIZE * (1 - TILE_OVERLAP))   # 128
INSULATION_WINDOW  = 500_000        # 500 kb diamond for insulation score
MIN_VALID_FRAC     = 0.5            # skip tile if >50% NaN

# Chromosomes to process (autosomes + X)
CHROMOSOMES = [
    "chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
    "chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19",
    "chr20","chr21","chr22","chrX",
]

# ─── Model Architecture ───────────────────────────────────────────────────────
# Encoder
ENCODER_CHANNELS   = [32, 64, 128, 256]   # stage output channels
ASSAY_EMBED_DIM    = 8

# Codebook
N_CODES            = 512
CODE_DIM           = 256
EMA_GAMMA          = 0.99
DEAD_THRESHOLD     = 2
REVIVAL_INTERVAL   = 100

# Masker
KEEP_RATIO         = 0.5            # keep 50% of 1024 tokens → 512 visible
TAU_F_WARMUP       = 1.0
TAU_F_FINAL        = 0.3
TAU_F_WARMUP_EPOCHS  = 10
TAU_F_ANNEAL_EPOCHS  = 20          # anneal over epochs 10–30

# Transformer demasker
N_TRANSFORMER_LAYERS = 4
N_HEADS              = 8
FFN_DIM              = 1024
SPATIAL_TOKENS       = 1024        # 32×32 = 1024

# Decoder
DECODER_CHANNELS   = [128, 64, 32]  # after the 256-channel demasker output

# Fingerprint projection
FP_DIM             = 32            # 256 → 32

# ─── Loss ─────────────────────────────────────────────────────────────────────
POS_WEIGHT_BOUNDARY = 9.0          # boundary bins ≈ 5–10%
LOSS_W_RECON        = 1.0
LOSS_W_VQ           = 0.25
LOSS_W_BOUNDARY     = 0.5
LOSS_W_COMPARTMENT  = 0.75
AUX_WARMUP_EPOCHS   = 5
AUX_RAMP_EPOCHS     = 10

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE         = 8
NUM_EPOCHS         = 50
LR                 = 1e-4
BETAS              = (0.9, 0.999)
WEIGHT_DECAY       = 0.01
GRAD_CLIP          = 1.0
WARMUP_STEPS       = 500
NUM_WORKERS        = 4
SEED               = 42

# ─── Ablation ─────────────────────────────────────────────────────────────────
ABLATION_EPOCHS    = 20
ABLATION_SUBSET    = 1.0           # fraction of data to use in ablation (1.0 = all 5 cell lines)

# ─── Query / Retrieval ────────────────────────────────────────────────────────
TOP_K              = 5

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_EVERY_N_STEPS  = 10
SAVE_EVERY_N_EPOCHS = 5
