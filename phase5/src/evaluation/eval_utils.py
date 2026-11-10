"""
Phase 5: Evaluation and Threshold Optimization Utilities for Multi-Label LSI.

Features:
1. Exact replication of official evaluate_phase5.py metrics (Macro-F1, Micro-F1, Subset Accuracy).
2. Per-Class Optimal Threshold Search (theta_c*) on Dev set to prevent rare class collapse and maximize Macro-F1.
3. Fallback to top-1 argmax for empty predictions to ensure every document has >= 1 predicted statute.
4. Export and validation of official submission.zip (containing prediction.json).
"""

import os
import json
import zipfile
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Tuple, Optional


def compute_lsi_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes exact metrics matching evaluate_phase5.py.
    y_true: binary matrix [N, 100]
    y_pred: binary matrix [N, 100]
    """
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    num_classes = y_true.shape[1]
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = float(np.mean(per_class_f1))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    subset_acc = float(accuracy_score(y_true, y_pred))
    
    # Check for class collapses (F1 == 0.0)
    collapsed_classes = int(np.sum(per_class_f1 == 0.0))
    active_classes = num_classes - collapsed_classes
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_acc": subset_acc,
        "active_classes": active_classes,
        "collapsed_classes": collapsed_classes,
        "min_class_f1": float(np.min(per_class_f1)),
        "max_class_f1": float(np.max(per_class_f1)),
        "per_class_f1": per_class_f1
    }


def optimize_per_class_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_th: float = 0.10,
    max_th: float = 0.65,
    num_steps: int = 56,
    default_th: float = 0.35
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Finds the optimal decision threshold theta_c* for each class c in [0, 99]
    independently to strictly maximize Macro-F1 across all 100 statute classes.
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, default_th, dtype=np.float32)
    threshold_grid = np.linspace(min_th, max_th, num_steps)
    
    for c in range(num_classes):
        y_t_c = y_true[:, c]
        y_p_c = y_probs[:, c]
        
        # If class has 0 positives in validation, keep default
        if np.sum(y_t_c) == 0:
            continue
            
        best_f1_c = -1.0
        best_th_c = default_th
        for th in threshold_grid:
            pred_c = (y_p_c >= th).astype(np.uint8)
            score = f1_score(y_t_c, pred_c, average="binary", zero_division=0)
            if score > best_f1_c:
                best_f1_c = score
                best_th_c = th
                
        # If all searched thresholds yielded 0 F1 (extremely rare), pick threshold with highest recall
        if best_f1_c <= 0.0:
            best_th_c = min_th
            
        best_thresholds[c] = best_th_c
        
    # Apply optimal per-class thresholds
    y_pred_opt = (y_probs >= best_thresholds.reshape(1, -1)).astype(np.uint8)
    
    # For any sample with 0 predicted labels, backfill with argmax class
    zero_pred_indices = np.where(y_pred_opt.sum(axis=1) == 0)[0]
    for idx in zero_pred_indices:
        best_c = np.argmax(y_probs[idx])
        y_pred_opt[idx, best_c] = 1
        
    metrics = compute_lsi_metrics(y_true, y_pred_opt)
    return best_thresholds, metrics


def probs_to_predictions(
    y_probs: np.ndarray,
    thresholds: np.ndarray,
    ensure_non_empty: bool = True
) -> List[List[int]]:
    """
    Converts continuous prediction probabilities to integer list of predicted statute IDs.
    """
    num_samples = y_probs.shape[0]
    binary_preds = (y_probs >= thresholds.reshape(1, -1)).astype(np.uint8)
    
    results = []
    for i in range(num_samples):
        row = binary_preds[i]
        labels = [int(c) for c in np.where(row == 1)[0]]
        
        # Fallback to top-1 if empty
        if not labels and ensure_non_empty:
            labels = [int(np.argmax(y_probs[i]))]
            
        results.append(sorted(list(set(labels))))
        
    return results


def export_lsi_submission(
    case_ids: List[str],
    predictions: List[List[int]],
    out_dir: str,
    expected_count: int = 13019
) -> Tuple[str, str]:
    """
    Exports predictions to prediction.json and archives to submission.zip.
    Validates formatting, uniqueness, and record count.
    """
    assert len(case_ids) == len(predictions), f"Count mismatch: {len(case_ids)} vs {len(predictions)}"
    if expected_count is not None and len(case_ids) != expected_count:
        print(f"Warning: Expected {expected_count} records, got {len(case_ids)}")
        
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "prediction.json")
    zip_path = os.path.join(out_dir, "submission.zip")
    
    records = []
    for cid, preds in zip(case_ids, predictions):
        clean_preds = sorted(list(set(int(x) for x in preds if 0 <= int(x) <= 99)))
        records.append({
            "id": str(cid),
            "pred": clean_preds
        })
        
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction.json")
        
    print(f"Exported submission: {zip_path} ({len(records)} records, {os.path.getsize(zip_path)/1024:.1f} KB)")
    return json_path, zip_path
