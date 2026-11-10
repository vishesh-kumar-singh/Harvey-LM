#!/usr/bin/env python3
"""
Phase 5: Multi-Architecture Grand Consensus Ensemble for LSI.

Averages predicted probabilities across all distinct foundational models,
optimizes ensemble per-class thresholds theta_c* on Dev, and packages
an elite consensus submission.zip.
"""

import os
import sys
import glob
import json
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from phase5.src.evaluation.eval_utils import (
    compute_lsi_metrics,
    optimize_per_class_thresholds,
    probs_to_predictions,
    export_lsi_submission
)

OUT_SUB_DIR     = os.path.join(CODE_DIR, "phase5/submissions/ensemble/grand_consensus")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase5/results.txt")


def main():
    print("=" * 80)
    print("Phase 5 Multi-Architecture Grand Consensus Ensemble")
    print("=" * 80)

    pattern = os.path.join(CODE_DIR, "phase5/models/**/test_dev_logits.pt")
    found_files = glob.glob(pattern, recursive=True)

    print(f"Discovered {len(found_files)} candidate model logit caches:")
    for fp in found_files:
        print(f" - {os.path.relpath(fp, CODE_DIR)}")

    if not found_files:
        print("No test_dev_logits.pt files found! Run training models first.")
        return

    dev_probs_list = []
    test_probs_list = []
    common_dev_golds = None
    common_dev_ids = None
    common_test_ids = None

    for fp in found_files:
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            d_p = data["dev_probs"]
            t_p = data["test_probs"]

            dev_probs_list.append(d_p)
            test_probs_list.append(t_p)

            if common_dev_golds is None:
                common_dev_golds = data["dev_golds"]
                common_dev_ids = data["dev_ids"]
                common_test_ids = data["test_ids"]

            print(f"Successfully integrated {os.path.basename(os.path.dirname(fp))}")
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")

    if not dev_probs_list:
        print("Error: No valid models loaded.")
        return

    # Average probabilities
    mean_dev_probs = np.mean(dev_probs_list, axis=0)
    mean_test_probs = np.mean(test_probs_list, axis=0)

    print(f"\nEnsembled {len(dev_probs_list)} models across {len(common_dev_ids)} dev cases and {len(common_test_ids)} test cases.")

    # Calibrate ensemble per-class thresholds
    opt_thresholds, ensemble_m = optimize_per_class_thresholds(common_dev_golds, mean_dev_probs)

    print(f"\nEnsemble Dev Evaluation:")
    print(f"Ensemble Dev Macro-F1:    {ensemble_m['macro_f1']:.4f}")
    print(f"Ensemble Dev Micro-F1:    {ensemble_m['micro_f1']:.4f}")
    print(f"Ensemble Dev Subset Acc:  {ensemble_m['subset_acc']*100:.2f}%")
    print(f"Active Classes:           {ensemble_m['active_classes']}/100 (Collapsed: {ensemble_m['collapsed_classes']})")

    # Generate final test predictions
    final_test_preds = probs_to_predictions(mean_test_probs, opt_thresholds, ensure_non_empty=True)

    json_path, zip_path = export_lsi_submission(
        case_ids=common_test_ids,
        predictions=final_test_preds,
        out_dir=OUT_SUB_DIR,
        expected_count=len(common_test_ids)
    )

    # Log to results.txt
    res_line = f"{'ENSEMBLE':<16} {'Grand Consensus Multi-Arch':<25} {ensemble_m['macro_f1']:<14.6f} {ensemble_m['micro_f1']:<14.6f} [SUBMITTED]    {zip_path}\n"
    with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
        f.write(res_line)
    print(f"Logged result to {RESULTS_TRACKER}")

    print("\nGrand Consensus Ensemble Completed Successfully!")


if __name__ == "__main__":
    main()
