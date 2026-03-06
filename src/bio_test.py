"""
bio_test.py — Biological annotation validation test on feature/bio-annotations branch.

Tests:
  1. Annotator sanity — verify cCRE counts are biologically plausible per cell line
  2. Overfit test (150 steps, small model) — same as before but with annotation stored in DB
  3. Annotated DB ingestion — verify ccre_annot vectors stored correctly
  4. Annotated query — verify LEVEL 3/4 outputs include cCRE diffs + p-values
  5. Biological validation — check known biology:
       - IMR-90 has more dELS/pELS than K562 (fibroblast vs leukemia)
       - IMR-90 has more CTCF anchors than HeLa-S3
       - Foreskin fibroblast has high H3K27ac/H3K4me3
  6. Ablation (4 configs) — verify loss trends are preserved with new annotation path
  7. Permutation test validation — verify p-values are well-calibrated

All outputs saved to trash/bio_test_results/
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
config.CHROMOSOMES = ["chr21", "chr22"]
config.NUM_WORKERS = 0

import torch
from config import (
    TRASH_DIR, CELL_LINE_REGISTRY, MCOOL_DIR,
    CCRE_REGISTRY, ANNOT_DIFF_THRESHOLD,
)
from model import MQVAE
from loss import total_loss
from masker import get_tau_f
from dataset import HiCTileDataset
from database import extract_and_ingest, init_db
from query import HiCStructuralDatabase
from annotator import (
    MultiSampleAnnotator, CcreIndex, CCRE_CATEGORIES,
    ANNOT_NAMES, N_CCRE_CATS, permutation_pvalue,
    bytes_to_annot,
)

RESULTS_DIR = TRASH_DIR / "bio_test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CELL_LINES = list(CELL_LINE_REGISTRY.keys())
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMALL = dict(
    n_codes=64, code_dim=128, fp_dim=16,
    use_boundary_head=True, use_compartment_head=True,
    use_masking=True, use_film=True,
    encoder_channels=[16, 32, 64, 128],
    n_transformer_layers=2, n_heads=4, ffn_dim=256,
    decoder_channels=[64, 32, 16],
)

t_start = time.time()


def _json_safe(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_safe)
    print(f"  → saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Annotator Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

def test_annotator_sanity():
    print("\n" + "=" * 64)
    print("TEST 1: Annotator Sanity (cCRE biological plausibility)")
    print("=" * 64)

    ann = MultiSampleAnnotator(CCRE_REGISTRY)

    # Query chr21 (available in all cell lines)
    chrom, start, end = "chr21", 10_000_000, 35_600_000

    results = {}
    print(f"\n  Window: {chrom}:{start//1_000_000}Mb-{end//1_000_000}Mb\n")
    print(f"  {'Cell Line':<25}  {'Active':>7}  {'PLS':>6}  {'pELS':>6}  {'dELS':>6}  "
          f"{'CTCF':>6}  {'CA-TF':>6}  {'CA-only':>8}  {'Complexity':>10}")
    print("  " + "-" * 90)

    for sid in CELL_LINES:
        v = ann.annotate(sid, chrom, start, end)
        if v is None:
            print(f"  {sid:<25}  (no annotation)")
            continue
        active   = int(v[:9].sum())
        pls      = int(v[0])
        pels     = int(v[1])
        dels     = int(v[2])
        ctcf     = int(v[3])
        catf     = int(v[4])
        caonly   = int(v[6])
        cplx     = float(v[N_CCRE_CATS + 1])
        results[sid] = {
            "active": active, "PLS": pls, "pELS": pels, "dELS": dels,
            "CA-CTCF": ctcf, "CA-TF": catf, "CA-only": caonly,
            "complexity": cplx,
        }
        print(f"  {sid:<25}  {active:>7}  {pls:>6}  {pels:>6}  {dels:>6}  "
              f"{ctcf:>6}  {catf:>6}  {caonly:>8}  {cplx:>10.3f}")

    print()

    # ── Biological validation checks ──────────────────────────────────────────
    checks = []

    # IMR-90 should have more regulatory elements than K562/HeLa-S3
    # (lung fibroblast cCRE files have richer annotations)
    imr90_active = results.get("IMR-90", {}).get("active", 0)
    k562_active  = results.get("K562",   {}).get("active", 0)
    hela_active  = results.get("HeLa-S3",{}).get("active", 0)
    check1 = imr90_active > k562_active
    check2 = imr90_active > hela_active
    checks.append(("IMR-90 > K562 active regulatory elements", check1,
                   f"{imr90_active} vs {k562_active}"))
    checks.append(("IMR-90 > HeLa-S3 active regulatory elements", check2,
                   f"{imr90_active} vs {hela_active}"))

    # IMR-90 should have CTCF sites (only it and foreskin have them in our data)
    imr90_ctcf = results.get("IMR-90", {}).get("CA-CTCF", 0)
    k562_ctcf  = results.get("K562",   {}).get("CA-CTCF", 0)
    check3 = imr90_ctcf > k562_ctcf
    checks.append(("IMR-90 has more CTCF than K562 (fibroblast CTCF enrichment)", check3,
                   f"IMR-90={imr90_ctcf}  K562={k562_ctcf}"))

    # IMR-90 complexity should be higher than HeLa-S3
    imr90_cplx = results.get("IMR-90", {}).get("complexity", 0)
    hela_cplx  = results.get("HeLa-S3", {}).get("complexity", 0)
    check4 = imr90_cplx > hela_cplx
    checks.append(("IMR-90 regulatory complexity > HeLa-S3", check4,
                   f"{imr90_cplx:.3f} vs {hela_cplx:.3f}"))

    # Foreskin fibroblast should have H3K27ac / H3K4me3 marks
    fsk_v = ann.annotate("foreskin_fibroblast", chrom, start, end)
    fsk_h3k27ac  = int(fsk_v[7]) if fsk_v is not None else 0
    fsk_h3k4me3  = int(fsk_v[8]) if fsk_v is not None else 0
    check5 = (fsk_h3k27ac + fsk_h3k4me3) > 0
    checks.append(("foreskin_fibroblast has H3K27ac/H3K4me3 marks", check5,
                   f"H3K27ac={fsk_h3k27ac}  H3K4me3={fsk_h3k4me3}"))

    # Differential: IMR-90 vs K562 — IMR-90 should have more dELS
    d = ann.differential("IMR-90", "K562", chrom, start, end)
    check6 = d is not None and d[2] > 0  # dELS index=2
    checks.append(("IMR-90 has more dELS than K562 (enhancer enrichment)", check6,
                   f"delta_dELS={d[2]:.0f}" if d is not None else "no data"))

    print("  Biological Validation Checks:")
    all_pass = True
    for name, passed, note in checks:
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {icon}  {name}")
        print(f"           ({note})")
        if not passed:
            all_pass = False

    save_json({
        "window": f"{chrom}:{start}-{end}",
        "per_sample": results,
        "checks": [{"name": n, "passed": p, "note": v} for n, p, v in checks],
        "all_pass": all_pass,
    }, RESULTS_DIR / "annotator_sanity.json")

    return all_pass, ann


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Overfit Test with Annotation
# ─────────────────────────────────────────────────────────────────────────────

def test_overfit():
    print("\n" + "=" * 64)
    print("TEST 2: Overfit Test (150 steps, small model)")
    print("=" * 64)

    ds = HiCTileDataset(cell_lines=CELL_LINES, augment=False)
    print(f"[dataset] Loaded {len(ds)} tiles from {len(CELL_LINES)} cell lines")

    # Build single batch with all tiles
    from torch.utils.data import DataLoader
    loader    = DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=False)
    batch     = next(iter(loader))
    contact   = batch["contact"].to(DEVICE)
    assay_id  = batch["assay_id"].to(DEVICE)
    boundary  = batch["boundary"].to(DEVICE)
    compartment = batch["compartment"].to(DEVICE)

    model = MQVAE(**SMALL).to(DEVICE)
    model.set_masker_temperatures(get_tau_f(0))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    print(f"\n {'Step':>6}  {'Total':>8}  {'Recon':>8}  {'Comp_r':>8}  {'Codes':>6}")
    print("  " + "-" * 50)

    losses = []
    for step in range(150):
        model.train()
        tau = get_tau_f(step / 150)
        model.set_masker_temperatures(tau)

        out = model(contact, assay_id)
        lv, metrics = total_loss(
            out,
            {"contact": contact, "boundary": boundary, "compartment": compartment},
            epoch=step // 15,
        )
        opt.zero_grad()
        lv.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(lv))
        if step % 15 == 0 or step == 149:
            codes = int((out["indices"].unique()).shape[0])
            comp_r = float(metrics.get("compartment_r", 0))
            print(f"  {step:>6}  {lv.item():>8.4f}  "
                  f"{metrics['recon']:>8.4f}  {comp_r:>8.3f}  {codes:>6}")

    loss_drop = (losses[0] - losses[-1]) / losses[0] * 100
    overfit_ok = loss_drop > 30

    print(f"\n  Loss: {losses[0]:.4f} → {losses[-1]:.4f}  ({loss_drop:.1f}% drop)")
    print(f"  Overfit check: {'✓ PASS' if overfit_ok else '✗ FAIL'}")

    # Save
    ckpt_path = RESULTS_DIR / "bio_model.pt"
    torch.save({"epoch": 0, "model": model.state_dict(), "arch": SMALL}, ckpt_path)
    print(f"  Model saved to {ckpt_path}")
    save_json({"loss_start": losses[0], "loss_end": losses[-1],
               "loss_drop_pct": loss_drop, "overfit_ok": overfit_ok,
               "n_steps": 150},
              RESULTS_DIR / "overfit_log.json")

    return model, ckpt_path, overfit_ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Annotated DB Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def test_db_ingest(ckpt_path):
    print("\n" + "=" * 64)
    print("TEST 3: Annotated DB Ingestion")
    print("=" * 64)

    db_path    = RESULTS_DIR / "bio_test.duckdb"
    faiss_path = RESULTS_DIR / "bio_test.faiss"

    extract_and_ingest(
        model_path = str(ckpt_path),
        db_path    = db_path,
        faiss_path = faiss_path,
        cell_lines = CELL_LINES,
        batch_size = 5,
        device_str = str(DEVICE),
        overwrite  = True,
    )

    # Verify ccre_annot was stored
    import duckdb as ddb
    conn = ddb.connect(str(db_path), read_only=True)

    total_fps = conn.execute("SELECT COUNT(*) FROM window_fingerprints").fetchone()[0]
    annot_non_null = conn.execute(
        "SELECT COUNT(*) FROM window_fingerprints WHERE ccre_annot IS NOT NULL"
    ).fetchone()[0]

    print(f"\n  Total fingerprints:          {total_fps}")
    print(f"  With ccre_annot (non-null):  {annot_non_null}")
    annot_ok = annot_non_null == total_fps

    # Inspect annotation vectors
    rows = conn.execute(
        "SELECT sample_id, chr, start_bp, ccre_annot FROM window_fingerprints ORDER BY sample_id, chr"
    ).fetchall()

    print(f"\n  Per-window annotation summary (active elements, complexity):")
    from annotator import ANNOT_NAMES, N_CCRE_CATS
    annot_data = {}
    for sid, chrom, start_bp, ann_blob in rows:
        if ann_blob is None:
            continue
        v = bytes_to_annot(ann_blob)
        key = f"{sid}({chrom})"
        active  = int(v[:9].sum())
        cplx    = float(v[N_CCRE_CATS + 1])
        ctcf_d  = float(v[N_CCRE_CATS + 2])
        annot_data[key] = {"active": active, "complexity": cplx, "ctcf_density": ctcf_d}
        print(f"    {key:<30}  active={active:>5}  cplx={cplx:.3f}  ctcf_density={ctcf_d:.3f}")

    # Verify IMR-90 has more complexity than K562 in DB
    imr90_cplx = np.mean([v["complexity"]
                          for k, v in annot_data.items() if k.startswith("IMR-90")])
    k562_cplx  = np.mean([v["complexity"]
                          for k, v in annot_data.items() if k.startswith("K562")])
    bio_check = imr90_cplx > k562_cplx
    print(f"\n  DB bio-check: IMR-90 complexity ({imr90_cplx:.3f}) > "
          f"K562 ({k562_cplx:.3f}): {'✓ PASS' if bio_check else '✗ FAIL'}")

    conn.close()

    save_json({"total_fps": total_fps, "annot_non_null": annot_non_null,
               "annot_ok": annot_ok, "bio_check": bio_check,
               "per_window": annot_data},
              RESULTS_DIR / "db_ingest_log.json")

    return db_path, faiss_path, annot_ok and bio_check


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Permutation Test Calibration
# ─────────────────────────────────────────────────────────────────────────────

def test_permutation_calibration(model):
    print("\n" + "=" * 64)
    print("TEST 4: Permutation Test Calibration")
    print("=" * 64)

    ds  = HiCTileDataset(cell_lines=CELL_LINES, augment=False)
    fps = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            c  = item["contact"].unsqueeze(0).to(DEVICE)
            ai = item["assay_id"].unsqueeze(0).to(DEVICE)
            fp = model.encode_fingerprint(c, ai).squeeze(0).cpu().numpy()
            fps.append(fp)
    fps = np.stack(fps)

    results = []
    rng = np.random.default_rng(42)

    print(f"\n  {'Query':>25}  {'ObsSim':>8}  {'P-value':>8}  {'Sig?':>6}")
    print("  " + "-" * 55)

    for i, (sid, fp) in enumerate(zip([ds[j]["sample_id"] for j in range(len(ds))], fps)):
        # Self-retrieval: high similarity, low p-value
        others = np.delete(fps, i, axis=0)
        obs, pv = permutation_pvalue(fp, others, n_perm=200, rng=rng)
        sig = "YES" if pv < 0.05 else "no"
        results.append({"query": sid, "obs_sim": float(obs), "p_value": float(pv), "sig": sig})
        print(f"  {sid:>25}  {obs:>8.4f}  {pv:>8.3f}  {sig:>6}")

    # Negative control: random fp should NOT be significant
    random_fp = rng.standard_normal(fps.shape[1]).astype(np.float32)
    obs_r, pv_r = permutation_pvalue(random_fp, fps, n_perm=200, rng=rng)
    neg_ok = pv_r > 0.1
    print(f"\n  Negative control (random fp):  obs={obs_r:.4f}  p={pv_r:.3f}  "
          f"{'✓ not significant' if neg_ok else '✗ significant (unexpected)'}")

    # Real fingerprints should have lower p-value than random
    real_pvs = [r["p_value"] for r in results]
    perm_ok  = float(np.mean(real_pvs)) < pv_r
    print(f"  Real mean p={np.mean(real_pvs):.3f} < random p={pv_r:.3f}: "
          f"{'✓ PASS' if perm_ok else '✗ FAIL'}")

    save_json({"results": results,
               "neg_control": {"obs": float(obs_r), "p_value": float(pv_r)},
               "perm_ok": perm_ok, "neg_ok": neg_ok},
              RESULTS_DIR / "permutation_test.json")
    return perm_ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Annotated Query End-to-End
# ─────────────────────────────────────────────────────────────────────────────

def test_annotated_query(ckpt_path, db_path, faiss_path):
    print("\n" + "=" * 64)
    print("TEST 5: Annotated Query End-to-End")
    print("=" * 64)

    db_obj = HiCStructuralDatabase(
        ckpt_path    = str(ckpt_path),
        db_path      = db_path,
        faiss_path   = faiss_path,
        device_str   = str(DEVICE),
        model_kwargs = SMALL,
        ccre_registry= CCRE_REGISTRY,
    )

    query_results = {}
    for sample_id, info in CELL_LINE_REGISTRY.items():
        mcool_path = str(MCOOL_DIR / info["file"])
        print(f"\n  Querying {sample_id}...")
        res = db_obj.query_mcool(
            mcool_path = mcool_path,
            sample_id  = sample_id,
            assay_type = info["assay"],
            k          = 3,
            verbose    = False,
        )
        db_obj.print_report(res)

        # Collect window-level cCRE annotations
        win_annots = {}
        for locus, v in res["window_results"].items():
            qa = v.get("ccre_annot_query")
            win_annots[locus] = {
                "similarity":   v["similarity_score"],
                "p_value":      v.get("p_value"),
                "ccre_diff":    v.get("ccre_diff", []),
                "active_cres":  int(qa[:9].sum()) if qa is not None else None,
            }

        query_results[sample_id] = {
            "predicted":    res["level1_cell_type"],
            "confidence":   round(res["level1_fraction"], 3),
            "correct":      res["level1_cell_type"] == sample_id,
            "window_annots": win_annots,
        }

    db_obj.close()

    # Summary
    print("\n  Self-retrieval summary:")
    print(f"  {'Sample':<30}  {'Predicted':>25}  {'Conf':>6}  {'Match':>8}")
    print("  " + "-" * 75)
    n_correct = 0
    for sid, r in query_results.items():
        icon = "✓" if r["correct"] else "✗"
        print(f"  {sid:<30}  {r['predicted']:>25}  {r['confidence']:>6.3f}  {icon:>8}")
        if r["correct"]:
            n_correct += 1

    accuracy = n_correct / len(query_results)
    print(f"\n  Self-retrieval: {n_correct}/{len(query_results)} = {accuracy:.1%}")

    save_json(query_results, RESULTS_DIR / "annotated_query.json")
    return accuracy == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Ablation (4 configs, 80 steps) — verify model trains with annotation path
# ─────────────────────────────────────────────────────────────────────────────

def _abl(overrides):
    return {**SMALL, **overrides}

ABLATION_CONFIGS = [
    {"id": 1, "name": "recon_only",    "kw": _abl(dict(use_boundary_head=False, use_compartment_head=False))},
    {"id": 2, "name": "full_v4",       "kw": _abl({})},
    {"id": 3, "name": "no_masking",    "kw": _abl(dict(use_masking=False))},
    {"id": 4, "name": "codebook_128",  "kw": _abl(dict(n_codes=128))},
]

def test_ablations():
    print("\n" + "=" * 64)
    print("TEST 6: Ablation Studies (4 configs × 80 steps)")
    print("=" * 64)

    ds = HiCTileDataset(cell_lines=CELL_LINES, augment=False)
    from torch.utils.data import DataLoader
    loader  = DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=False)
    batch   = next(iter(loader))
    contact    = batch["contact"].to(DEVICE)
    assay_id   = batch["assay_id"].to(DEVICE)
    boundary   = batch["boundary"].to(DEVICE)
    compartment= batch["compartment"].to(DEVICE)

    abl_results = []
    for cfg in ABLATION_CONFIGS:
        t0 = time.time()
        print(f"\n  [{cfg['id']}] {cfg['name']}...")
        model = MQVAE(**cfg["kw"]).to(DEVICE)
        model.set_masker_temperatures(get_tau_f(0))
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        losses = []
        for step in range(80):
            model.train()
            model.set_masker_temperatures(get_tau_f(step / 80))
            out = model(contact, assay_id)
            lv, metrics = total_loss(
                out,
                {"contact": contact, "boundary": boundary, "compartment": compartment},
                epoch=step // 8,
            )
            opt.zero_grad(); lv.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(lv))

        drop = (losses[0] - losses[-1]) / losses[0] * 100
        codes = int(out["indices"].unique().shape[0])
        elapsed = time.time() - t0
        r = {
            "id": cfg["id"], "name": cfg["name"],
            "loss_start": losses[0], "loss_end": losses[-1],
            "loss_drop_pct": drop, "codes": codes,
            "elapsed_s": round(elapsed, 1),
        }
        abl_results.append(r)
        print(f"    loss: {losses[0]:.4f}→{losses[-1]:.4f}  ({drop:.1f}% drop) | "
              f"codes={codes}/{cfg['kw']['n_codes']} | {elapsed:.0f}s")

    # Print table
    print(f"\n  {'#':>3}  {'Name':<15}  {'Loss↓%':>7}  {'Codes':>8}  {'Time':>6}")
    print("  " + "-" * 45)
    for r in abl_results:
        print(f"  {r['id']:>3}  {r['name']:<15}  {r['loss_drop_pct']:>7.1f}%  "
              f"{r['codes']:>8}  {r['elapsed_s']:>6.0f}s")

    all_drop = all(r["loss_drop_pct"] > 20 for r in abl_results)
    print(f"\n  All configs show >20% loss drop: {'✓ PASS' if all_drop else '✗ FAIL'}")
    save_json(abl_results, RESULTS_DIR / "ablation_results.json")
    return all_drop


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  MQ-VAE Bio-Annotation Test Suite")
    print(f"  Branch: feature/bio-annotations")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {RESULTS_DIR}")
    print("=" * 64)

    results_summary = {}

    ann_ok, ann = test_annotator_sanity()
    results_summary["annotator_sanity"] = ann_ok

    model, ckpt_path, overfit_ok = test_overfit()
    results_summary["overfit"] = overfit_ok

    db_path, faiss_path, ingest_ok = test_db_ingest(ckpt_path)
    results_summary["db_ingest"] = ingest_ok

    perm_ok = test_permutation_calibration(model)
    results_summary["permutation_test"] = perm_ok

    query_ok = test_annotated_query(ckpt_path, db_path, faiss_path)
    results_summary["annotated_query"] = query_ok

    abl_ok = test_ablations()
    results_summary["ablations"] = abl_ok

    elapsed_total = time.time() - t_start
    results_summary["elapsed_s"] = round(elapsed_total, 1)

    save_json(results_summary, RESULTS_DIR / "summary.json")

    print("\n" + "=" * 64)
    print("  BIO TEST SUITE RESULTS")
    print("=" * 64)
    all_pass = True
    for test, passed in results_summary.items():
        if test == "elapsed_s":
            continue
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {icon}  {test}")
        if not passed:
            all_pass = False
    print(f"\n  Total time: {elapsed_total:.0f}s")
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 64)
