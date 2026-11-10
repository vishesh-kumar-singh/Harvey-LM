"""
Phase 4: Evaluation and Threshold Calibration Utilities.

1. Evaluates binary predictions using standard Macro-F1, Precision, and Recall matching evaluate_phase4.py.
2. Performs optimal 1D threshold search on Dev predictions to directly maximize unweighted binary Macro-F1.
3. Packages verified prediction.json into submission.zip archives.
"""

import os
import json
import zipfile
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Tuple


def compute_cjpe_metrics(golds: List[int], preds: List[int]) -> Dict[str, float]:
    """Computes Macro-F1, Precision, Recall, and Accuracy matching official evaluator."""
    macro_f1 = f1_score(golds, preds, average="macro", labels=[0, 1], zero_division=0)
    f1_0 = f1_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    f1_1 = f1_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    prec_1 = precision_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    rec_1 = recall_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    acc = np.mean(np.array(golds) == np.array(preds))

    return {
        "macro_f1": float(macro_f1),
        "f1_0": float(f1_0),
        "f1_1": float(f1_1),
        "precision_1": float(prec_1),
        "recall_1": float(rec_1),
        "accuracy": float(acc)
    }


def find_optimal_threshold(
    golds: List[int],
    logits: np.ndarray,
    delta_range: Tuple[float, float] = (-1.0, 1.0),
    num_steps: int = 201
) -> Tuple[float, float, Dict[str, float]]:
    """
    Finds optimal decision offset delta* on validation logits difference (z_1 - z_0)
    that strictly maximizes unweighted Macro-F1.
    """
    diffs = logits[:, 1] - logits[:, 0]
    best_delta = 0.0
    best_f1 = -1.0
    best_metrics = {}

    deltas = np.linspace(delta_range[0], delta_range[1], num_steps)
    for d in deltas:
        preds = (diffs + d > 0).astype(int).tolist()
        score = f1_score(golds, preds, average="macro", labels=[0, 1], zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_delta = float(d)

    optimal_preds = (diffs + best_delta > 0).astype(int).tolist()
    best_metrics = compute_cjpe_metrics(golds, optimal_preds)
    return best_delta, best_f1, best_metrics


def export_cjpe_submission(
    case_ids: List[str],
    predictions: List[int],
    out_dir: str,
    expected_count: int = 1517
) -> Tuple[str, str]:
    """
    Exports prediction.json and packages it into submission.zip.
    Verifies that all predictions are integers (0 or 1) and exact case count matches.
    """
    os.makedirs(out_dir, exist_ok=True)
    assert len(case_ids) == len(predictions), f"Length mismatch: {len(case_ids)} ids vs {len(predictions)} preds"
    if expected_count is not None:
        assert len(case_ids) == expected_count, f"Expected {expected_count} cases, but got {len(case_ids)}"

    records = []
    for cid, p in zip(case_ids, predictions):
        assert int(p) in (0, 1), f"Invalid prediction value: {p}"
        records.append({
            "id": str(cid),
            "pred": int(p)
        })

    json_path = os.path.join(out_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "prediction.json")

    print(f"Exported submission: {zip_path} ({len(records)} records, {os.path.getsize(zip_path)/1024:.1f} KB)")
    return json_path, zip_path


def load_ground_truth(path: str) -> Dict[str, int]:
    """Loads ground truth JSON file mapping case ID to integer label (0 or 1)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data:
        out[r["id"]] = int(r["gold"])
    return out
