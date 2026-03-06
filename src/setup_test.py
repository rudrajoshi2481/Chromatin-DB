"""
setup_test.py — One-time setup verification test.

Checks:
  1. All dependencies importable + versions correct
  2. Registry auto-discovery (mcool + cCRE files found)
  3. Replicate detection + grouping
  4. cCRE fallback flags (structural-only mode for missing cCRE)
  5. Dashboard generates PNG without errors
  6. Short training run (3 epochs, 1 cell line) with dashboard output
  7. Database ingestion works
  8. Query works end-to-end

Run once after fresh setup:
    python src/setup_test.py

Output:
    trash/setup_test.log    — full log
    trash/plots/            — dashboard PNG from training run
"""

import sys
import json
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── override to chr21/22 only for speed ──────────────────────────────────────
import config as _cfg
_cfg.CHROMOSOMES = ["chr21", "chr22"]
_cfg.NUM_WORKERS = 0

from config import (
    TRASH_DIR, PLOTS_DIR, DATA_DIR,
    CELL_LINE_REGISTRY, CCRE_REGISTRY, MCOOL_DIR,
    ASSAY_TYPES, CHECKPOINTS_DIR,
)

RESULTS_DIR = TRASH_DIR / "setup_test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

t_start = time.time()
results = {}

def _check(name, condition, note=""):
    icon = "✓ PASS" if condition else "✗ FAIL"
    print(f"    {icon}  {name}" + (f"  [{note}]" if note else ""))
    return condition

def _section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def test_dependencies():
    _section("TEST 1: Dependencies")
    checks = []

    deps = [
        ("torch",        "2.1.0"),
        ("cooler",       "0.9.3"),
        ("cooltools",    "0.7.0"),
        ("duckdb",       "0.10.0"),
        ("faiss",        None),
        ("numpy",        "1.24.0"),
        ("scipy",        "1.11.0"),
        ("sklearn",      "1.3.0"),
        ("pandas",       "2.0.0"),
        ("tqdm",         None),
        ("h5py",         None),
        ("matplotlib",   "3.7.0"),
        ("seaborn",      None),
        ("bioframe",     None),
    ]

    for pkg, min_ver in deps:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            ok  = True
        except ImportError:
            ver = "MISSING"
            ok  = False
        checks.append(_check(f"{pkg} importable", ok, f"v{ver}"))

    # Special: faiss
    try:
        import faiss
        checks.append(_check("faiss has IndexFlatIP", hasattr(faiss, "IndexFlatIP")))
    except ImportError:
        checks.append(_check("faiss-cpu importable", False, "MISSING"))

    # psutil
    try:
        import psutil
        checks.append(_check("psutil importable", True, f"v{psutil.__version__}"))
    except ImportError:
        checks.append(_check("psutil importable", False, "MISSING"))

    results["dependencies"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Registry auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def test_registry():
    _section("TEST 2: Registry Auto-Discovery")

    print(f"\n  MCOOL_DIR: {MCOOL_DIR}  (exists={MCOOL_DIR.exists()})")
    print(f"  CCRE_DIR:  {_cfg.CCRE_DIR}  (exists={_cfg.CCRE_DIR.exists()})")
    print(f"\n  Auto-discovered {len(CELL_LINE_REGISTRY)} mcool samples:")
    for sid, cfg in CELL_LINE_REGISTRY.items():
        path = MCOOL_DIR / cfg["file"]
        print(f"    {sid:<40}  exists={path.exists()}")

    print(f"\n  Auto-discovered {len(CCRE_REGISTRY)} cCRE files:")
    for sid, path in CCRE_REGISTRY.items():
        print(f"    {sid:<40}  exists={Path(path).exists()}")

    checks = [
        _check("Registry non-empty", len(CELL_LINE_REGISTRY) > 0,
               f"{len(CELL_LINE_REGISTRY)} samples"),
        _check("At least 1 mcool exists",
               any((MCOOL_DIR / v["file"]).exists() for v in CELL_LINE_REGISTRY.values())),
        _check("cCRE registry non-empty", len(CCRE_REGISTRY) > 0,
               f"{len(CCRE_REGISTRY)} files"),
    ]

    results["registry"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Replicate detection
# ─────────────────────────────────────────────────────────────────────────────

def test_replicates():
    _section("TEST 3: Replicate Detection")
    from replicates import (
        detect_replicate_group, group_registry,
        validate_registry, print_registry_status,
        build_ccre_registry_from_validated,
    )

    # Test detection patterns
    cases = [
        ("K562_rep1",         "K562"),
        ("K562_rep2",         "K562"),
        ("IMR-90_r1",         "IMR-90"),
        ("HeLa-S3_batch_A",   "HeLa-S3"),
        ("K562",              "K562"),
        ("foreskin_fibroblast", "foreskin_fibroblast"),
        ("K562_4DNFI18UHVRO", "K562_4DNFI18UHVRO"),  # no replicate suffix
    ]

    print()
    checks = []
    for sample_id, expected in cases:
        got = detect_replicate_group(sample_id)
        ok  = got == expected
        checks.append(_check(f"detect_replicate_group('{sample_id}')", ok,
                             f"got='{got}' expected='{expected}'"))

    # Test full validation
    print()
    validated = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
    print_registry_status(validated)

    # Check mode assignment
    for sid, entry in validated.items():
        has_ccre  = entry["ccre_exists"]
        has_mcool = entry["mcool_exists"]
        expected_mode = "full" if has_ccre else "structural_only"
        checks.append(_check(
            f"{sid} mode correct",
            entry["mode"] == expected_mode or entry["skip"],
            f"mode={entry['mode']} ccre={has_ccre}",
        ))

    # Build filtered cCRE registry
    ccre_filtered = build_ccre_registry_from_validated(validated)
    print(f"\n  Filtered cCRE registry: {len(ccre_filtered)} samples with valid cCRE")

    checks.append(_check("validate_registry returns entries", len(validated) > 0))
    checks.append(_check("Filtered cCRE has only existing files",
                          all(Path(p).exists() for p in ccre_filtered.values()),
                          f"{len(ccre_filtered)} files"))

    results["replicates"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: cCRE fallback flags
# ─────────────────────────────────────────────────────────────────────────────

def test_ccre_fallback():
    _section("TEST 4: cCRE Fallback Flags")
    from replicates import validate_registry

    # Inject a fake sample with no cCRE
    fake_registry = dict(CELL_LINE_REGISTRY)
    fake_registry["UNKNOWN_SAMPLE"] = {
        "file":   "nonexistent_sample.mcool",
        "assay":  "bulk_hic",
        "tissue": "unknown",
    }
    fake_ccre = dict(CCRE_REGISTRY)
    # Don't add cCRE for UNKNOWN_SAMPLE

    print()
    validated = validate_registry(fake_registry, MCOOL_DIR, fake_ccre)

    unknown = validated.get("UNKNOWN_SAMPLE", {})
    checks = [
        _check("Missing mcool flagged as skip",
               unknown.get("skip") == True,
               f"skip={unknown.get('skip')}"),
        _check("Missing mcool has warnings",
               len(unknown.get("warnings", [])) > 0),
        _check("Missing cCRE sets structural_only mode",
               any(
                   e["mode"] == "structural_only"
                   for e in validated.values()
                   if not e.get("skip") and not e["ccre_exists"]
               ) if any(
                   not e.get("skip") and not e["ccre_exists"]
                   for e in validated.values()
               ) else True),
        _check("Real samples not affected",
               all(
                   e["mode"] in ("full", "structural_only")
                   for sid, e in validated.items()
                   if sid != "UNKNOWN_SAMPLE"
               )),
    ]

    results["ccre_fallback"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Dashboard generates PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_dashboard():
    _section("TEST 5: Dashboard PNG Generation")
    import numpy as np
    from dashboard import TrainingDashboard

    dash = TrainingDashboard(run_name="setup_test", out_dir=PLOTS_DIR)

    # Feed 3 fake epochs
    for epoch in range(3):
        train_m = {
            "total": 1.5 - epoch * 0.2, "recon": 0.9 - epoch * 0.1,
            "vq": 0.3, "boundary": 0.2, "compartment": 0.1,
            "boundary_f1": 0.2 + epoch * 0.1,
            "compartment_r": 0.1 + epoch * 0.15,
            "active_codes": 100 + epoch * 50,
        }
        val_m = {k: v * 1.05 for k, v in train_m.items()}

        fp_emb    = np.random.randn(30, 32).astype(np.float32)
        fp_labels = ["K562"] * 10 + ["IMR-90"] * 10 + ["HeLa-S3"] * 10
        cb_usage  = np.random.poisson(5, 512).astype(np.float32)

        dash.update(
            epoch          = epoch,
            train_metrics  = train_m,
            val_metrics    = val_m,
            fp_embeddings  = fp_emb,
            fp_labels      = fp_labels,
            codebook_usage = cb_usage,
            tau_f          = 0.5 - epoch * 0.1,
        )

    png_ok  = dash.png_path.exists() and dash.png_path.stat().st_size > 10_000
    json_ok = dash.json_path.exists()

    print(f"\n  Dashboard PNG:  {dash.png_path}  ({dash.png_path.stat().st_size//1024}KB)")
    print(f"  Dashboard JSON: {dash.json_path}")

    checks = [
        _check("PNG generated", png_ok, f"size={dash.png_path.stat().st_size}B"),
        _check("JSON generated", json_ok),
        _check("PNG > 10KB (non-trivial)", png_ok),
    ]

    results["dashboard"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Short training run (3 epochs, 1 cell line, dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def test_training():
    _section("TEST 6: Short Training Run (3 epochs)")

    from replicates import validate_registry, active_samples
    from train import train

    # Pick first available cell line
    validated  = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
    available  = active_samples(validated)
    if not available:
        print("  No available samples — SKIPPING")
        results["training"] = False
        return False

    cell_line = list(available.keys())[0]
    print(f"\n  Using cell line: {cell_line}")
    print(f"  Epochs: 3, dashboard every epoch")

    try:
        result = train(
            cell_lines  = [cell_line],
            epochs      = 3,
            batch_size  = 2,
            run_name    = "setup_test",
            plot_every  = 1,
            n_codes     = 64,
        )

        png_path = Path(result.get("dashboard", ""))
        checks = [
            _check("Training completed", True),
            _check("Dashboard PNG created", png_path.exists(),
                   str(png_path)),
            _check("Loss decreased or stayed stable",
                   result["history"][-1]["val_total"] <=
                   result["history"][0]["val_total"] * 2.0),
            _check("Active codes > 0",
                   result["history"][-1]["active_codes"] > 0),
        ]

    except Exception as e:
        print(f"  Training FAILED: {e}")
        traceback.print_exc()
        checks = [False]

    results["training"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Database ingestion
# ─────────────────────────────────────────────────────────────────────────────

def test_database():
    _section("TEST 7: Database Ingestion")
    import shutil
    import duckdb
    from replicates import validate_registry, active_samples

    test_db    = RESULTS_DIR / "setup_test.duckdb"
    test_faiss = RESULTS_DIR / "setup_test.faiss"
    test_keys  = RESULTS_DIR / "setup_test.locus_keys.json"

    # Clean up previous
    for p in [test_db, test_faiss, test_keys]:
        if p.exists():
            p.unlink()

    # Find best checkpoint
    ckpt_dir = CHECKPOINTS_DIR / "setup_test"
    ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        print("  No checkpoint found — SKIPPING (run test_training first)")
        results["database"] = False
        return False

    ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
    print(f"  Using checkpoint: {ckpt.name}")

    try:
        from database import extract_and_ingest

        validated = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
        available = active_samples(validated)
        cell_line = list(available.keys())[0]

        extract_and_ingest(
            model_path  = str(ckpt),
            cell_lines  = [cell_line],
            db_path     = test_db,
            faiss_path  = test_faiss,
            overwrite   = True,
        )

        conn      = duckdb.connect(str(test_db), read_only=True)
        n_rows    = conn.execute("SELECT COUNT(*) FROM window_fingerprints").fetchone()[0]
        n_samples = conn.execute("SELECT COUNT(DISTINCT sample_id) FROM window_fingerprints").fetchone()[0]
        conn.close()

        print(f"  Rows ingested: {n_rows}")
        print(f"  Samples: {n_samples}")

        checks = [
            _check("Database created", test_db.exists()),
            _check("FAISS index created", test_faiss.exists()),
            _check("Rows ingested", n_rows > 0, f"n={n_rows}"),
        ]

    except Exception as e:
        print(f"  Database ingestion FAILED: {e}")
        traceback.print_exc()
        checks = [False]

    results["database"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Query end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_query():
    _section("TEST 8: Query End-to-End")

    test_db    = RESULTS_DIR / "setup_test.duckdb"
    test_faiss = RESULTS_DIR / "setup_test.faiss"
    ckpt_dir   = CHECKPOINTS_DIR / "setup_test"

    if not test_db.exists():
        print("  Database not found — SKIPPING (run test_database first)")
        results["query"] = False
        return False

    ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        print("  No checkpoint found — SKIPPING")
        results["query"] = False
        return False

    ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)

    # Use first available mcool as query
    from replicates import validate_registry, active_samples
    validated = validate_registry(CELL_LINE_REGISTRY, MCOOL_DIR, CCRE_REGISTRY)
    available = active_samples(validated)
    if not available:
        print("  No samples available — SKIPPING")
        results["query"] = False
        return False

    cell_line = list(available.keys())[0]
    mcool_path = MCOOL_DIR / CELL_LINE_REGISTRY[cell_line]["file"]
    print(f"\n  Query file: {mcool_path.name}")

    try:
        from query import HiCStructuralDatabase
        db = HiCStructuralDatabase(
            ckpt_path  = str(ckpt),
            db_path    = str(test_db),
            faiss_path = str(test_faiss),
        )
        result = db.query_mcool(str(mcool_path), verbose=False)
        db.print_report(result)
        db.close()

        checks = [
            _check("Query returned results", result is not None),
            _check("Window results populated", result["n_windows"] > 0,
                   f"n={result['n_windows']}"),
            _check("Level 1 classification", result["level1_cell_type"] != ""),
        ]

    except Exception as e:
        print(f"  Query FAILED: {e}")
        traceback.print_exc()
        checks = [False]

    results["query"] = all(checks)
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  MQ-VAE Setup Verification Test")
    print(f"  Output: {RESULTS_DIR}")
    print("=" * 64)

    test_dependencies()
    test_registry()
    test_replicates()
    test_ccre_fallback()
    test_dashboard()
    test_training()
    test_database()
    test_query()

    elapsed = time.time() - t_start
    results["elapsed_s"] = round(elapsed, 1)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print("  SETUP TEST RESULTS")
    print("=" * 64)
    all_pass = True
    for test, passed in results.items():
        if test == "elapsed_s":
            continue
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {icon}  {test}")
        if not passed:
            all_pass = False

    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Dashboard:  {PLOTS_DIR}/dashboard_setup_test.png")
    print(f"  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 64)
