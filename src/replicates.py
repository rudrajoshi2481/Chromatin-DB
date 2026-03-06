"""
replicates.py — Replicate management for MQ-VAE training and querying.

Auto-detects replicate naming conventions:
  K562_rep1, K562_rep2, K562_rep3   → group: K562
  IMR-90_r1, IMR-90_r2              → group: IMR-90
  HeLa-S3_batch_A, HeLa-S3_batch_B → group: HeLa-S3
  K562                               → group: K562 (single sample, unchanged)

Also handles:
  - Missing cCRE annotation (structural-only mode per sample)
  - Missing mcool file (skip with warning)
  - Averaging fingerprints within replicate groups
  - Confidence intervals from replicate variance
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Patterns that indicate replicate suffixes
_REP_PATTERNS = [
    r"_rep(\d+)$",           # _rep1, _rep2
    r"_r(\d+)$",             # _r1, _r2
    r"_replicate(\d+)$",     # _replicate1
    r"_batch_?([A-Za-z\d]+)$",  # _batch_A, _batchA
    r"_([12345])$",           # trailing single digit 1-5
    r"\.([12345])$",          # .1, .2 suffix
]

_REP_RE = re.compile("|".join(_REP_PATTERNS))


def detect_replicate_group(sample_id: str) -> str:
    """
    Strip replicate suffix to get biological group name.

    Examples:
        "K562_rep1"     → "K562"
        "IMR-90_r2"     → "IMR-90"
        "HeLa-S3"       → "HeLa-S3"
        "K562_batch_A"  → "K562"
    """
    m = _REP_RE.search(sample_id)
    if m:
        return sample_id[: m.start()]
    return sample_id


def group_registry(registry: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Group registry entries by biological replicate group.

    Returns: {group_name: [sample_id, ...]}

    Example:
        {
            "K562":   ["K562_rep1", "K562_rep2", "K562_rep3"],
            "IMR-90": ["IMR-90"],
        }
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for sample_id in registry:
        group = detect_replicate_group(sample_id)
        groups[group].append(sample_id)
    return dict(groups)


def validate_registry(
    registry:    Dict[str, Dict],
    mcool_dir:   Path,
    ccre_registry: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Validate all registry entries. Returns annotated registry with flags:
        {
            sample_id: {
                ...original config...,
                "mcool_path":     str,        # resolved absolute path
                "mcool_exists":   bool,
                "ccre_path":      str | None,
                "ccre_exists":    bool,
                "group":          str,         # replicate group name
                "mode":           "full" | "structural_only",
                "skip":           bool,         # True if mcool missing
                "warnings":       [str, ...],
            }
        }
    """
    validated = {}

    for sample_id, cfg in registry.items():
        warns   = []
        entry   = dict(cfg)
        group   = detect_replicate_group(sample_id)
        entry["group"] = group

        # ── Resolve mcool path ─────────────────────────────────────────────
        mcool_file = cfg.get("file", "")
        if Path(mcool_file).is_absolute():
            mcool_path = Path(mcool_file)
        else:
            mcool_path = mcool_dir / mcool_file

        entry["mcool_path"]   = str(mcool_path)
        entry["mcool_exists"] = mcool_path.exists()

        if not entry["mcool_exists"]:
            warns.append(
                f"mcool file not found: {mcool_path}  — sample will be SKIPPED"
            )
            entry["skip"] = True
        else:
            entry["skip"] = False

        # ── Resolve cCRE path ──────────────────────────────────────────────
        ccre_path = ccre_registry.get(sample_id) or ccre_registry.get(group)
        entry["ccre_path"]   = ccre_path
        entry["ccre_exists"] = bool(ccre_path and Path(ccre_path).exists())

        if not entry["ccre_exists"]:
            if ccre_path:
                warns.append(
                    f"cCRE file not found: {ccre_path}  — regulatory analysis DISABLED"
                )
            else:
                warns.append(
                    f"No cCRE annotation registered for {sample_id}  "
                    f"(group: {group})  — regulatory analysis DISABLED"
                )

        entry["mode"]     = "full" if entry["ccre_exists"] else "structural_only"
        entry["warnings"] = warns

        if warns:
            for w in warns:
                print(f"[replicates] WARNING  {sample_id}: {w}")

        validated[sample_id] = entry

    # Print summary
    n_ok   = sum(1 for e in validated.values() if not e["skip"] and e["mode"] == "full")
    n_struct = sum(1 for e in validated.values() if not e["skip"] and e["mode"] == "structural_only")
    n_skip = sum(1 for e in validated.values() if e["skip"])
    groups = group_registry(registry)

    print(f"\n[replicates] Registry summary:")
    print(f"  Total samples:    {len(validated)}")
    print(f"  Replicate groups: {len(groups)}")
    print(f"  Full mode:        {n_ok}")
    print(f"  Structural only:  {n_struct}")
    print(f"  Skipped:          {n_skip}")
    print(f"  Groups: {list(groups.keys())}\n")

    return validated


def active_samples(validated_registry: Dict[str, Dict]) -> Dict[str, Dict]:
    """Return only non-skipped entries."""
    return {k: v for k, v in validated_registry.items() if not v.get("skip", False)}


def build_ccre_registry_from_validated(
    validated: Dict[str, Dict]
) -> Dict[str, str]:
    """
    Build a cCRE registry containing only samples that have valid cCRE files.
    Used to initialize MultiSampleAnnotator.
    """
    return {
        sid: entry["ccre_path"]
        for sid, entry in validated.items()
        if entry.get("ccre_exists") and not entry.get("skip")
    }


# ── Fingerprint averaging ─────────────────────────────────────────────────────

def average_group_fingerprints(
    fingerprints: Dict[str, np.ndarray],   # {sample_id: [N_windows, FP_DIM]}
    groups:       Dict[str, List[str]],    # {group: [sample_id, ...]}
) -> Dict[str, np.ndarray]:
    """
    Average fingerprints within each replicate group.
    Only averages windows that exist in ALL replicates of the group.

    Returns: {group_name: averaged_fingerprints [N_windows, FP_DIM]}
    """
    averaged = {}
    for group, members in groups.items():
        available = [m for m in members if m in fingerprints]
        if not available:
            continue
        if len(available) == 1:
            averaged[group] = fingerprints[available[0]]
        else:
            stacked = np.stack([fingerprints[m] for m in available], axis=0)
            averaged[group] = stacked.mean(axis=0)
    return averaged


def replicate_confidence(
    fingerprints: Dict[str, np.ndarray],   # {sample_id: [N_windows, FP_DIM]}
    groups:       Dict[str, List[str]],
) -> Dict[str, Dict]:
    """
    Compute within-group replicate consistency metrics.

    Returns: {
        group_name: {
            "n_replicates": int,
            "mean_cosine_similarity": float,  # average pairwise cosine sim
            "std_cosine_similarity": float,
            "confidence": "high" | "medium" | "low",
        }
    }
    """
    results = {}
    for group, members in groups.items():
        available = [m for m in members if m in fingerprints]
        if len(available) < 2:
            results[group] = {
                "n_replicates": len(available),
                "mean_cosine_similarity": 1.0,
                "std_cosine_similarity": 0.0,
                "confidence": "single_sample",
            }
            continue

        sims = []
        fps  = [fingerprints[m] for m in available]

        # Pairwise cosine similarity (mean over windows)
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                a = fps[i] / (np.linalg.norm(fps[i], axis=-1, keepdims=True) + 1e-8)
                b = fps[j] / (np.linalg.norm(fps[j], axis=-1, keepdims=True) + 1e-8)
                sim = float((a * b).sum(axis=-1).mean())
                sims.append(sim)

        mean_sim = float(np.mean(sims))
        std_sim  = float(np.std(sims))

        if mean_sim > 0.8:
            confidence = "high"
        elif mean_sim > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        results[group] = {
            "n_replicates":           len(available),
            "mean_cosine_similarity": round(mean_sim, 4),
            "std_cosine_similarity":  round(std_sim, 4),
            "confidence":             confidence,
        }

    return results


# ── Query-time replicate matching ─────────────────────────────────────────────

def group_query_results(
    matches:  List[Dict],    # from _search_locus_numpy
    registry: Dict[str, Dict],
) -> List[Dict]:
    """
    Augment query matches with replicate group information.
    Groups matches by biological group and reports group-level similarity.

    Input matches: [{sample_id, score, cell_type, ...}, ...]
    Output:        Same, with added:
                    "group":           biological group name
                    "group_sim_mean":  mean similarity over all group replicates
                    "group_sim_std":   std of similarities over group replicates
                    "n_replicates":    how many replicates matched
                    "confidence":      "high" | "medium" | "low" | "single_sample"
    """
    # Build group → scores map
    group_scores: Dict[str, List[float]] = defaultdict(list)
    for m in matches:
        sid   = m.get("sample_id", "")
        group = detect_replicate_group(sid)
        group_scores[group].append(m["score"])

    # Augment each match
    for m in matches:
        sid    = m.get("sample_id", "")
        group  = detect_replicate_group(sid)
        scores = group_scores[group]
        mean_s = float(np.mean(scores))
        std_s  = float(np.std(scores))

        m["group"]          = group
        m["group_sim_mean"] = round(mean_s, 4)
        m["group_sim_std"]  = round(std_s, 4)
        m["n_replicates"]   = len(scores)
        m["confidence"]     = (
            "high"   if std_s < 0.05 else
            "medium" if std_s < 0.15 else
            "low"
        ) if len(scores) > 1 else "single_sample"

    return matches


def best_group_match(
    matches:  List[Dict],
    registry: Dict[str, Dict],
) -> Tuple[str, float, str]:
    """
    Return (group_name, mean_similarity, confidence) for the best-matching
    replicate group.
    """
    augmented = group_query_results(matches, registry)

    # Aggregate per group
    group_agg: Dict[str, List[float]] = defaultdict(list)
    for m in augmented:
        group_agg[m["group"]].append(m["score"])

    best_group = max(group_agg, key=lambda g: np.mean(group_agg[g]))
    scores     = group_agg[best_group]
    mean_sim   = float(np.mean(scores))
    std_sim    = float(np.std(scores))
    conf       = (
        "high"   if std_sim < 0.05 or len(scores) == 1 else
        "medium" if std_sim < 0.15 else
        "low"
    )
    return best_group, round(mean_sim, 4), conf


def print_registry_status(validated: Dict[str, Dict]):
    """Print a formatted table of registry status."""
    print("\n" + "=" * 72)
    print("  Sample Registry Status")
    print("=" * 72)
    print(f"  {'Sample ID':<30}  {'Group':<20}  {'mcool':>5}  {'cCRE':>5}  {'Mode':<16}")
    print("  " + "-" * 68)
    for sid, entry in validated.items():
        skip = "SKIP" if entry.get("skip") else ("✓" if entry["mcool_exists"] else "✗")
        ccre = "✓" if entry["ccre_exists"] else "✗"
        mode = entry.get("mode", "unknown")
        grp  = entry.get("group", sid)
        print(f"  {sid:<30}  {grp:<20}  {skip:>5}  {ccre:>5}  {mode:<16}")
    print("=" * 72 + "\n")
