"""
Phase 7: Evaluation & Submission Packaging Utilities for Prior Case Retrieval.
Strictly conforms to official evaluate_phase7.py standards.
"""

import os
import json
import zipfile
from typing import Dict, List, Set, Tuple

K = 7


def compute_pcr_metrics(
    gold: Dict[str, List[str]],
    pred: Dict[str, List[str]]
) -> Dict[str, float]:
    """Computes Micro-F1@7, Micro-Precision@7, and Micro-Recall@7 matching evaluate_phase7.py."""
    tp = fp = fn = 0
    for q, labels in gold.items():
        g = set(labels)
        p = set(pred.get(q, [])[:K])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "micro_f1_7": f1,
        "micro_precision_7": precision,
        "micro_recall_7": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def export_pcr_submission(
    pred_dict: Dict[str, List[str]],
    output_zip_path: str,
    allowed_candidates: Set[str],
    expected_queries: Set[str]
) -> None:
    """
    Validates and packages prediction.json into compliant submission.zip.
    """
    errors = []
    pred_queries = set(pred_dict.keys())

    missing = expected_queries - pred_queries
    extra = pred_queries - expected_queries

    if missing:
        errors.append(f"Missing {len(missing)} query IDs: {sorted(list(missing))[:5]}")
    if extra:
        errors.append(f"Unexpected {len(extra)} query IDs: {sorted(list(extra))[:5]}")

    cleaned_pred = {}
    for q in expected_queries:
        cands = pred_dict.get(q, [])
        if not isinstance(cands, list):
            errors.append(f"Query {q}: prediction is not a list")
            continue
        if len(cands) != K:
            errors.append(f"Query {q}: expected {K} candidates, got {len(cands)}")
            continue
        seen = set()
        for idx, c in enumerate(cands):
            if not isinstance(c, str) or not c.strip():
                errors.append(f"Query {q} idx {idx}: invalid candidate ID '{c}'")
            if c in seen:
                errors.append(f"Query {q}: duplicate candidate ID '{c}'")
            seen.add(c)
            if c not in allowed_candidates:
                errors.append(f"Query {q}: candidate ID '{c}' is outside candidate pool")

        cleaned_pred[q] = cands[:K]

    if errors:
        raise ValueError(f"Submission validation failed with {len(errors)} errors:\n" + "\n".join(errors[:10]))

    out_dir = os.path.dirname(output_zip_path)
    os.makedirs(out_dir, exist_ok=True)
    pred_json_path = os.path.join(out_dir, "prediction.json")

    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_pred, f, indent=2)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_json_path, arcname="prediction.json")

    print(f"Verified & created submission archive: {output_zip_path} ({len(cleaned_pred)} queries, {os.path.getsize(output_zip_path):,} bytes)")
