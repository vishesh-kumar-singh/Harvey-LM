#!/usr/bin/env python3
"""
Generate Official Submission for HarveyLM Hierarchical Chunk-Transformer (LORMA).
Loads the saved checkpoint from code/phase4/models/harveylm/lorma_hierarchical/best_model.pt,
performs evaluation on dev and test sets, packages submission.zip, and logs test metrics.
"""

import os
import sys


import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from phase4.src.data.cjpe_dataset import (
    load_raw_cjpe_data,
    CJPEDataset,
    cjpe_collate_fn_chunked
)
from phase4.src.model.cjpe_classifier import (
    HarveyHierarchicalClassifier,
    apply_lorma_to_model,
    mark_only_lorma_as_trainable
)
from phase4.src.evaluation.eval_utils import (
    compute_cjpe_metrics,
    find_optimal_threshold,
    export_cjpe_submission,
    load_ground_truth
)

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
HARVEYLM_CKPT  = os.path.join(CODE_DIR, "phase1/models/final_architectures/2_HarveyLM/epoch_99")
DATA_DIR       = os.path.join(CODE_DIR, "phase4/data/raw/Phase_4_CJPE")
TEST_TRUTH     = os.path.join(CODE_DIR, "phase4/data/eval_truth/test_truth.json")

CKPT_PATH      = os.path.join(CODE_DIR, "phase4/models/harveylm/lorma_hierarchical/best_model.pt")
OUT_DIR        = os.path.join(CODE_DIR, "phase4/models/harveylm/lorma_hierarchical")
SUB_DIR        = os.path.join(CODE_DIR, "phase4/submissions/harveylm/lorma_hierarchical")
RESULTS_TXT    = os.path.join(CODE_DIR, "phase4/results.txt")


def evaluate_hierarchical(model, dataloader, device):
    model.eval()
    all_logits = []
    all_golds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            chunk_ids = batch["chunk_input_ids"].to(device)
            chunk_masks = batch["chunk_masks"].to(device)
            doc_masks = batch["doc_chunk_masks"].to(device)
            out = model(chunk_input_ids=chunk_ids, chunk_masks=chunk_masks, doc_chunk_masks=doc_masks)

            logits = out["logits"].float().cpu()
            all_logits.append(logits)
            all_ids.extend(batch["ids"])
            if "labels" in batch:
                all_golds.extend(batch["labels"].cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    return all_ids, all_logits, all_golds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Generating Submission for HarveyLM Hierarchical (8x1024 tok, LORMA)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Submission Output: {SUB_DIR}")
    print("=" * 80)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")

    os.makedirs(SUB_DIR, exist_ok=True)

    # 1. Load Tokenizer
    print("\n[1/6] Loading legal BPE tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 3
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 1
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 2

    # 2. Load Data
    print("\n[2/6] Loading Dev and Test data...")
    dev_path = os.path.join(DATA_DIR, "Train/dev.json")
    test_path = os.path.join(DATA_DIR, "Test/test.json")

    raw_dev = load_raw_cjpe_data(dev_path, deduplicate=True)
    raw_test = load_raw_cjpe_data(test_path, deduplicate=False)
    print(f"Loaded: Dev={len(raw_dev)} unique cases | Test={len(raw_test)} cases")

    dev_dataset = CJPEDataset(
        raw_dev, tokenizer,
        mode="hierarchical", chunk_len=1024, max_chunks=8,
        is_training=False, anti_leakage=False
    )
    test_dataset = CJPEDataset(
        raw_test, tokenizer,
        mode="hierarchical", chunk_len=1024, max_chunks=8,
        is_training=False, anti_leakage=False
    )

    dev_loader = DataLoader(
        dev_dataset, batch_size=4, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=lambda b: cjpe_collate_fn_chunked(b, pad_token_id=tokenizer.pad_token_id)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=lambda b: cjpe_collate_fn_chunked(b, pad_token_id=tokenizer.pad_token_id)
    )

    # 3. Instantiate Model & Inject LORMA
    print("\n[3/6] Instantiating HarveyHierarchicalClassifier...")
    model = HarveyHierarchicalClassifier(
        ckpt_dir=HARVEYLM_CKPT,
        num_classes=2,
        max_chunks=8,
        num_queries=4,
        dropout=0.15,
        tau_la=1.0
    )

    all_targets = ["w1", "w2", "w3", "wq", "wk", "wv", "wo"]
    model.backbone = apply_lorma_to_model(
        model.backbone,
        target_modules=all_targets,
        r=32,
        lorma_alpha=64,
        dropout=0.05
    )
    mark_only_lorma_as_trainable(model.backbone)

    # 4. Load Saved Weights
    print(f"\n[4/6] Loading weights from {CKPT_PATH}...")
    best_ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.to(device)

    best_delta = best_ckpt.get("best_delta", -0.030)
    best_dev_f1 = best_ckpt.get("best_macro_f1", 0.5102)
    saved_epoch = best_ckpt.get("epoch", 1)
    print(f"Loaded checkpoint: Epoch {saved_epoch} (Saved Dev Macro-F1: {best_dev_f1:.4f}, Delta*: {best_delta:+.4f})")

    # 5. Evaluate on Dev
    print("\n[5/6] Verifying evaluation on Dev set (2,511 unique cases)...")
    dev_ids, dev_logits, dev_golds = evaluate_hierarchical(model, dev_loader, device)

    # Optimal threshold search on dev
    opt_delta, calib_f1, calib_m = find_optimal_threshold(dev_golds, dev_logits)
    print(f"Dev Evaluation: Calibrated Macro-F1: {calib_m['macro_f1']:.4f} (delta*={opt_delta:+.3f}) | Acc: {calib_m['accuracy']*100:.2f}%")

    use_delta = opt_delta if abs(opt_delta) < 2.0 else best_delta

    # 6. Evaluate on Hidden Test Cases
    print("\n[6/6] Generating Test Predictions (1,517 hidden cases)...")
    test_ids, test_logits, _ = evaluate_hierarchical(model, test_loader, device)

    diffs = test_logits[:, 1] - test_logits[:, 0]
    final_test_preds = (diffs + use_delta > 0).astype(int).tolist()

    class_0_cnt = sum(1 for p in final_test_preds if p == 0)
    class_1_cnt = sum(1 for p in final_test_preds if p == 1)
    print(f"Test Class Distribution: 0={class_0_cnt} ({class_0_cnt/len(final_test_preds)*100:.2f}%) | 1={class_1_cnt} ({class_1_cnt/len(final_test_preds)*100:.2f}%)")

    # Export submission
    json_path, zip_path = export_cjpe_submission(
        case_ids=test_ids,
        predictions=final_test_preds,
        out_dir=SUB_DIR,
        expected_count=1517
    )

    # Cache logits for ensembling
    logits_cache_path = os.path.join(OUT_DIR, "test_dev_logits.pt")
    torch.save({
        "dev_ids": dev_ids,
        "dev_logits": dev_logits,
        "dev_golds": dev_golds,
        "test_ids": test_ids,
        "test_logits": test_logits,
        "best_delta": use_delta,
        "metrics": calib_m
    }, logits_cache_path)
    print(f"Saved evaluation logits cache: {logits_cache_path}")

    # Verify against official test benchmark ground truth
    if os.path.exists(TEST_TRUTH):
        truth_dict = load_ground_truth(TEST_TRUTH)
        test_golds = [truth_dict[cid] for cid in test_ids]
        test_m = compute_cjpe_metrics(test_golds, final_test_preds)

        print("\n" + "=" * 60)
        print("OFFICIAL TEST BENCHMARK RESULTS (HARVEYLM HIERARCHICAL):")
        print(f"Test Macro-F1:  {test_m['macro_f1']:.6f}")
        print(f"Test Precision: {test_m['precision_1']:.6f}")
        print(f"Test Recall:    {test_m['recall_1']:.6f}")
        print(f"Test Accuracy:  {test_m['accuracy']*100:.2f}%")
        print("=" * 60)

        # Update results.txt
        res_line = f"{'HARVEYLM':<16} {'LORMA_HIERARCHICAL':<25} {calib_m['macro_f1']:<14.6f} {test_m['macro_f1']:<14.6f} {test_m['accuracy']*100:.2f}%        {zip_path}\n"
        with open(RESULTS_TXT, "a", encoding="utf-8") as f:
            f.write(res_line)
        print(f"Logged result to {RESULTS_TXT}")

    print("\nHierarchical Submission Generation Completed Successfully!")


if __name__ == "__main__":
    main()
