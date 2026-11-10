#!/usr/bin/env python3
"""
Phase 4: Multi-Architecture Grand Consensus Ensemble for CJPE.

Combines calibrated test-case decision probabilities across all 4 foundational backbones:
1. HarveyLM LORMA Head-Tail (0.7712 Test Macro-F1)
2. HarveyLM QLoRA Head-Tail
3. BaselineLM LORMA Head-Tail
4. BaselineLM LoRA Head-Tail
5. HarveyMoE LORMA Head-Tail
6. LegalEncoderDecoder LORMA Head-Tail

Performs:
- Soft probability averaging across diverse architectural inductive biases
- Post-hoc threshold calibration delta* on unified dev logits
- Evaluation against test_truth.json benchmark
- Packaging of verified submission.zip
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

from phase4.src.evaluation.eval_utils import (
    compute_cjpe_metrics,
    find_optimal_threshold,
    export_cjpe_submission,
    load_ground_truth
)

TEST_TRUTH_PATH = os.path.join(CODE_DIR, "phase4/data/eval_truth/test_truth.json")
OUT_SUB_DIR     = os.path.join(CODE_DIR, "phase4/submissions/ensemble/grand_consensus")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase4/results.txt")

LOGIT_PATHS = [
    os.path.join(CODE_DIR, "code/phase4/models/harveylm/lorma_headtail/test_dev_logits.pt"),
    os.path.join(CODE_DIR, "code/phase4/models/harveylm/qlora_headtail/test_dev_logits.pt"),
    os.path.join(CODE_DIR, "code/phase4/models/baseline/lorma_headtail/test_dev_logits.pt"),
    os.path.join(CODE_DIR, "code/phase4/models/baseline/lora_headtail/test_dev_logits.pt"),
    os.path.join(CODE_DIR, "code/phase4/models/moe/lorma_headtail/test_dev_logits.pt"),
    os.path.join(CODE_DIR, "code/phase4/models/enc_dec/lorma_headtail/test_dev_logits.pt"),
]


def main():
    print("=" * 80)
    print("Phase 4 Multi-Architecture Grand Consensus Ensemble")
    print("=" * 80)

    available_models = []
    # Search for all existing test_dev_logits.pt files in phase4
    pattern = os.path.join(CODE_DIR, "phase4/models/**/test_dev_logits.pt")
    found_files = glob.glob(pattern, recursive=True)

    print(f"Discovered {len(found_files)} candidate model logit caches:")
    for fp in found_files:
        print(f" - {os.path.relpath(fp, CODE_DIR)}")

    if not found_files:
        print("No test_dev_logits.pt files found! Run training models first.")
        return

    dev_logits_list = []
    test_logits_list = []
    common_dev_golds = None
    common_dev_ids = None
    common_test_ids = None

    for fp in found_files:
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            d_logits = torch.tensor(data["dev_logits"]) if not isinstance(data["dev_logits"], torch.Tensor) else data["dev_logits"]
            t_logits = torch.tensor(data["test_logits"]) if not isinstance(data["test_logits"], torch.Tensor) else data["test_logits"]

            # Convert to softmax probabilities
            d_probs = torch.softmax(d_logits.float(), dim=-1).numpy()
            t_probs = torch.softmax(t_logits.float(), dim=-1).numpy()

            dev_logits_list.append(d_probs)
            test_logits_list.append(t_probs)

            if common_dev_golds is None:
                common_dev_golds = data["dev_golds"]
                common_dev_ids = data["dev_ids"]
                common_test_ids = data["test_ids"]

            print(f"Successfully integrated {os.path.basename(os.path.dirname(fp))}")
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")

    if not dev_logits_list:
        print("Error: No valid models loaded.")
        return

    # Average soft probabilities across models
    mean_dev_probs = np.mean(dev_logits_list, axis=0)
    mean_test_probs = np.mean(test_logits_list, axis=0)

    print(f"\nEnsembled {len(dev_logits_list)} models across {len(common_dev_ids)} dev cases and {len(common_test_ids)} test cases.")

    # Convert mean probabilities back to log-odds for threshold calibration
    eps = 1e-6
    clamped_dev_probs = np.clip(mean_dev_probs, eps, 1.0 - eps)
    dev_log_odds = np.log(clamped_dev_probs[:, 1] / clamped_dev_probs[:, 0])

    clamped_test_probs = np.clip(mean_test_probs, eps, 1.0 - eps)
    test_log_odds = np.log(clamped_test_probs[:, 1] / clamped_test_probs[:, 0])

    # Find optimal threshold delta*
    dev_synthetic_logits = np.stack([-0.5 * dev_log_odds, 0.5 * dev_log_odds], axis=-1)
    delta_star, calib_f1, calib_m = find_optimal_threshold(common_dev_golds, dev_synthetic_logits)

    print(f"\nEnsemble Dev Calibration:")
    print(f"Optimal Threshold delta*: {delta_star:+.4f}")
    print(f"Ensemble Dev Macro-F1:    {calib_m['macro_f1']:.4f}")
    print(f"Ensemble Dev Accuracy:    {calib_m['accuracy']*100:.2f}%")

    # Generate final test predictions
    final_test_preds = (test_log_odds + delta_star > 0).astype(int).tolist()
    c0 = sum(1 for p in final_test_preds if p == 0)
    c1 = sum(1 for p in final_test_preds if p == 1)
    print(f"\nFinal Ensemble Test Predictions:")
    print(f"Class 0 (Dismissed): {c0} ({c0/len(final_test_preds)*100:.2f}%)")
    print(f"Class 1 (Allowed):   {c1} ({c1/len(final_test_preds)*100:.2f}%)")

    # Export submission
    os.makedirs(OUT_SUB_DIR, exist_ok=True)
    json_path, zip_path = export_cjpe_submission(
        case_ids=common_test_ids,
        predictions=final_test_preds,
        out_dir=OUT_SUB_DIR,
        expected_count=1517
    )

    # Evaluate against test benchmark ground truth
    if os.path.exists(TEST_TRUTH_PATH):
        truth_dict = load_ground_truth(TEST_TRUTH_PATH)
        test_golds = [truth_dict[cid] for cid in common_test_ids]
        test_m = compute_cjpe_metrics(test_golds, final_test_preds)

        print("\n" + "=" * 60)
        print("OFFICIAL TEST BENCHMARK EVALUATION (GRAND ENSEMBLE):")
        print(f"Test Macro-F1:  {test_m['macro_f1']:.6f}")
        print(f"Test Precision: {test_m['precision_1']:.6f}")
        print(f"Test Recall:    {test_m['recall_1']:.6f}")
        print(f"Test Accuracy:  {test_m['accuracy']*100:.2f}%")
        print("=" * 60)

        res_line = f"{'ENSEMBLE':<16} {'GRAND_CONSENSUS':<25} {calib_m['macro_f1']:<14.6f} {test_m['macro_f1']:<14.6f} {test_m['accuracy']*100:.2f}%        {zip_path}\n"
        with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
            f.write(res_line)
        print(f"Logged ensemble result to {RESULTS_TRACKER}")


if __name__ == "__main__":
    main()
