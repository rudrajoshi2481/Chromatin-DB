"""
database.py — DuckDB schema creation and fingerprint ingestion.

Three-tier storage:
  Tier 1: FAISS IndexFlatIP over locus centroids (~78 KB)
  Tier 2: DuckDB table 'window_fingerprints'  — per-sample 32-dim fps + labels
  Tier 3: DuckDB tables 'samples', 'sample_histograms'  — metadata + code histograms

Usage:
    python src/database.py --ckpt checkpoints/full/mqvae_epochXXX_best.pt
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
import duckdb
import faiss

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DB_PATH, FAISS_PATH, PROCESSED_DIR, CHECKPOINTS_DIR,
    CELL_LINE_REGISTRY, ASSAY_TYPES, DATA_DIR,
    FP_DIM, N_CODES, BATCH_SIZE, NUM_WORKERS,
    CCRE_REGISTRY,
)
from model import MQVAE
from dataset import build_inference_loader
from annotator import MultiSampleAnnotator, ANNOT_DIM, annot_to_bytes


# ── Schema creation ────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id   VARCHAR PRIMARY KEY,
    cell_type   VARCHAR NOT NULL,
    tissue      VARCHAR,
    assay_type  VARCHAR NOT NULL,
    source      VARCHAR DEFAULT '4DN',
    n_tiles     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS window_fingerprints (
    id          INTEGER PRIMARY KEY,
    sample_id   VARCHAR NOT NULL REFERENCES samples(sample_id),
    chr         VARCHAR NOT NULL,
    start_bp    BIGINT  NOT NULL,
    end_bp      BIGINT  NOT NULL,
    fingerprint BLOB    NOT NULL,    -- float32[fp_dim]
    boundary    BLOB,                -- float32[256]
    compartment BLOB,                -- float32[256]
    ccre_annot  BLOB                 -- float32[ANNOT_DIM] cCRE annotation vector
);

CREATE INDEX IF NOT EXISTS idx_wf_locus   ON window_fingerprints(chr, start_bp);
CREATE INDEX IF NOT EXISTS idx_wf_sample  ON window_fingerprints(sample_id);

CREATE TABLE IF NOT EXISTS sample_histograms (
    sample_id       VARCHAR PRIMARY KEY REFERENCES samples(sample_id),
    code_histogram  BLOB NOT NULL    -- float32[n_codes] L1-normalised
);

CREATE TABLE IF NOT EXISTS locus_centroids (
    chr         VARCHAR NOT NULL,
    start_bp    BIGINT  NOT NULL,
    end_bp      BIGINT  NOT NULL,
    centroid    BLOB    NOT NULL,    -- float32[fp_dim]
    n_samples   INTEGER DEFAULT 0,
    PRIMARY KEY (chr, start_bp)
);
"""


def init_db(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create DuckDB database and tables if not already present."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_SQL)
    conn.commit()
    print(f"[database] DuckDB initialised at {db_path}")
    return conn


def register_samples(
    conn: duckdb.DuckDBPyConnection,
    cell_lines: Optional[List[str]] = None,
) -> None:
    """Insert sample metadata rows (skip if already present)."""
    if cell_lines is None:
        cell_lines = list(CELL_LINE_REGISTRY.keys())

    for sample_id in cell_lines:
        info = CELL_LINE_REGISTRY[sample_id]
        existing = conn.execute(
            "SELECT sample_id FROM samples WHERE sample_id = ?", [sample_id]
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO samples (sample_id, cell_type, tissue, assay_type) VALUES (?, ?, ?, ?)",
                [sample_id, sample_id, info.get("tissue", "unknown"), info["assay"]],
            )
    conn.commit()
    print(f"[database] Registered {len(cell_lines)} samples")


# ── Fingerprint extraction ─────────────────────────────────────────────────────

def extract_and_ingest(
    model_path:  str,
    db_path:     Path = DB_PATH,
    faiss_path:  Path = FAISS_PATH,
    cell_lines:  Optional[List[str]] = None,
    batch_size:  int  = 16,
    device_str:  str  = "auto",
    overwrite:   bool = False,
) -> None:
    """
    Full pipeline:
      1. Load trained model checkpoint
      2. Extract 32-dim fingerprints + code histograms for all tiles
      3. Insert into DuckDB (window_fingerprints + sample_histograms)
      4. Compute per-locus centroids → DuckDB locus_centroids
      5. Build FAISS IndexFlatIP over centroids → save to disk
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[database] Loading model from {model_path}")
    ckpt  = torch.load(model_path, map_location=device)
    arch  = ckpt.get("arch") or {}
    model = MQVAE(**arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[database] Model loaded (epoch {ckpt.get('epoch','?')})")

    # ── Connect DB ────────────────────────────────────────────────────────────
    conn = init_db(db_path)
    if cell_lines is None:
        cell_lines = list(CELL_LINE_REGISTRY.keys())
    register_samples(conn, cell_lines)

    # ── Purge existing fingerprints if overwrite ───────────────────────────────
    if overwrite:
        for sid in cell_lines:
            conn.execute("DELETE FROM window_fingerprints WHERE sample_id = ?", [sid])
            conn.execute("DELETE FROM sample_histograms   WHERE sample_id = ?", [sid])
        conn.execute("DELETE FROM locus_centroids")
        conn.commit()
        print("[database] Cleared existing fingerprints")

    # ── Load cCRE annotator ───────────────────────────────────────────────────
    ccre_reg = {s: CCRE_REGISTRY[s] for s in cell_lines if s in CCRE_REGISTRY}
    annotator = MultiSampleAnnotator(ccre_reg) if ccre_reg else None

    # ── Extract per-sample ────────────────────────────────────────────────────
    loader = build_inference_loader(cell_lines=cell_lines, batch_size=batch_size, num_workers=NUM_WORKERS)

    fp_id       = 1
    # Track existing id high-water mark
    row = conn.execute("SELECT MAX(id) FROM window_fingerprints").fetchone()
    if row and row[0] is not None:
        fp_id = int(row[0]) + 1

    # Accumulate histograms per sample: sample_id → list of [N] index arrays
    hist_accum: Dict[str, List[np.ndarray]] = {s: [] for s in cell_lines}

    insert_rows = []

    with torch.no_grad():
        for batch in loader:
            contact     = batch["contact"].to(device)
            assay_id    = batch["assay_id"].to(device)
            sample_ids  = batch["sample_ids"]
            chroms      = batch["chroms"]
            start_bps   = batch["start_bps"].cpu().numpy()
            end_bps     = batch["end_bps"].cpu().numpy()
            bd_np       = batch["boundary"].cpu().numpy()
            comp_np     = batch["compartment"].cpu().numpy()

            # Forward pass
            outputs = model(contact, assay_id)
            fps     = outputs["fingerprint"].cpu().float().numpy()   # [B, fp_dim]
            indices = outputs["indices"].cpu()                        # [B, K]

            for i in range(len(sample_ids)):
                sid    = sample_ids[i]
                chrom  = chroms[i]
                s_bp   = int(start_bps[i])
                e_bp   = int(end_bps[i])
                fp_b   = fps[i].astype(np.float32).tobytes()
                bd_b   = bd_np[i].astype(np.float32).tobytes()
                cp_b   = comp_np[i].astype(np.float32).tobytes()

                # cCRE annotation vector
                if annotator is not None:
                    annot_v = annotator.annotate(sid, chrom, s_bp, e_bp)
                    ann_b   = annot_to_bytes(annot_v) if annot_v is not None else None
                else:
                    ann_b = None

                insert_rows.append((fp_id, sid, chrom, s_bp, e_bp, fp_b, bd_b, cp_b, ann_b))
                fp_id += 1

                # Accumulate code usage for histogram
                hist_accum[sid].append(indices[i].numpy().astype(np.int64))

            # Batch-insert rows every 1000 tiles
            if len(insert_rows) >= 1000:
                conn.executemany(
                    "INSERT OR IGNORE INTO window_fingerprints "
                    "(id,sample_id,chr,start_bp,end_bp,fingerprint,boundary,compartment,ccre_annot) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    insert_rows,
                )
                conn.commit()
                print(f"  Inserted {fp_id-1} fingerprints so far...", end="\r")
                insert_rows = []

    # Insert remaining
    if insert_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO window_fingerprints "
            "(id,sample_id,chr,start_bp,end_bp,fingerprint,boundary,compartment,ccre_annot) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            insert_rows,
        )
        conn.commit()

    total_fps = fp_id - 1
    print(f"\n[database] Inserted {total_fps} window fingerprints")

    # ── Update n_tiles count per sample ───────────────────────────────────────
    conn.execute("""
        UPDATE samples
        SET n_tiles = (
            SELECT COUNT(*) FROM window_fingerprints wf
            WHERE wf.sample_id = samples.sample_id
        )
    """)
    conn.commit()

    # ── Sample histograms ─────────────────────────────────────────────────────
    for sid, idx_list in hist_accum.items():
        if not idx_list:
            continue
        all_idx = np.concatenate(idx_list)
        counts  = np.bincount(all_idx, minlength=N_CODES).astype(np.float32)
        counts  /= counts.sum() + 1e-8
        hist_b  = counts.tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO sample_histograms (sample_id, code_histogram) VALUES (?,?)",
            [sid, hist_b],
        )
    conn.commit()
    print(f"[database] Inserted {len(hist_accum)} sample histograms")

    # ── Compute locus centroids ────────────────────────────────────────────────
    print("[database] Computing locus centroids...")
    rows = conn.execute(
        "SELECT chr, start_bp, end_bp, fingerprint FROM window_fingerprints"
    ).fetchall()

    locus_fps: Dict[tuple, List[np.ndarray]] = {}
    for chr_, start_bp, end_bp, fp_b in rows:
        key = (chr_, start_bp, end_bp)
        fp  = np.frombuffer(fp_b, dtype=np.float32).copy()
        locus_fps.setdefault(key, []).append(fp)

    centroid_rows = []
    centroids_list = []
    locus_keys     = []

    for (chr_, start_bp, end_bp), fps_list in locus_fps.items():
        centroid = np.mean(fps_list, axis=0).astype(np.float32)
        centroid_rows.append((chr_, start_bp, end_bp, centroid.tobytes(), len(fps_list)))
        centroids_list.append(centroid)
        locus_keys.append((chr_, start_bp, end_bp))

    conn.executemany(
        "INSERT OR REPLACE INTO locus_centroids (chr,start_bp,end_bp,centroid,n_samples) VALUES (?,?,?,?,?)",
        centroid_rows,
    )
    conn.commit()
    print(f"[database] Stored {len(centroid_rows)} locus centroids")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    if centroids_list:
        centroids_arr = np.stack(centroids_list).astype(np.float32)          # [N_loci, fp_dim]
        faiss.normalize_L2(centroids_arr)                                     # cosine via inner product
        actual_fp_dim = centroids_arr.shape[1]
        index = faiss.IndexFlatIP(actual_fp_dim)
        index.add(centroids_arr)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(faiss_path))

        # Save locus key mapping alongside FAISS index
        key_path = faiss_path.with_suffix(".locus_keys.json")
        with open(key_path, "w") as f:
            json.dump([list(k) for k in locus_keys], f)

        print(f"[database] FAISS index saved: {faiss_path} ({len(centroids_list)} centroids)")

    conn.close()
    print("[database] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build DuckDB fingerprint database")
    p.add_argument("--ckpt",        required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--db",          default=str(DB_PATH))
    p.add_argument("--faiss",       default=str(FAISS_PATH))
    p.add_argument("--cell_lines",  nargs="+", default=None)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--overwrite",   action="store_true")
    args = p.parse_args()

    extract_and_ingest(
        model_path  = args.ckpt,
        db_path     = Path(args.db),
        faiss_path  = Path(args.faiss),
        cell_lines  = args.cell_lines,
        batch_size  = args.batch_size,
        device_str  = args.device,
        overwrite   = args.overwrite,
    )
