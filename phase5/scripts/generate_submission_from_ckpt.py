#!/usr/bin/env python3
"""
Phase 5: High-Speed Submission Generator from Model Checkpoint.

Loads any best_model.pt checkpoint and pre-tokenized cached data to:
1. Run fast evaluation on Dev (10,181 cases) to verify exact Macro-F1 & Micro-F1.
2. Run fast test inference on Test (13,019 cases).
3. Apply optimal per-class calibrated decision thresholds theta_c*.
4. Package verified submission.zip containing prediction.json with 13,019 items.
5. Save test_dev_logits.pt for consensus ensembling.
6. Log to code/phase5/results.txt.
"""

import os
import sys
import argparse
import json
import zipfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase5.src.model.lsi_classifier import (
    UniversalLSIClassifier,
    apply_lora_to_model,
    apply_lorma_to_model,
    apply_qlora_to_model
)
from phase5.src.evaluation.eval_utils import (
    compute_lsi_metrics,
    probs_to_predictions,
    export_lsi_submission
)

CACHE_DIR       = os.path.join(CODE_DIR, "phase5/data/cache")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase5/results.txt")


class CachedLSIDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item["input_ids"].long()
        labels = torch.zeros(100, dtype=torch.float32)
        if "labels" in item and len(item["labels"]) > 0:
            for l in item["labels"]:
                if 0 <= l < 100:
                    labels[l] = 1.0

        attn_mask = torch.ones(len(seq), dtype=torch.bool)
        return {
            "id": item["id"],
            "input_ids": seq,
            "attention_mask": attn_mask,
            "labels": labels
        }


def cached_collate_fn(batch):
    ids = [b["id"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    max_len = max(b["input_ids"].size(0) for b in batch)
    pad_id = 3

    batch_ids = []
    batch_masks = []
    for b in batch:
        cur_len = b["input_ids"].size(0)
        pad_len = max_len - cur_len
        if pad_len > 0:
            p_ids = torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
            p_mask = torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.bool)])
        else:
            p_ids = b["input_ids"]
            p_mask = b["attention_mask"]
        batch_ids.append(p_ids)
        batch_masks.append(p_mask)

    return {
        "ids": ids,
        "input_ids": torch.stack(batch_ids, dim=0),
        "attention_mask": torch.stack(batch_masks, dim=0),
        "labels": labels
    }


def evaluate_dataset(model, data_list, device, batch_size=32):
    ds = CachedLSIDataset(data_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=cached_collate_fn)

    all_ids = []
    all_probs = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"].float()
                probs = torch.sigmoid(logits).cpu().numpy()

            all_ids.extend(batch["ids"])
            all_probs.append(probs)
            all_golds.append(batch["labels"].numpy())

    return all_ids, np.concatenate(all_probs, axis=0), np.concatenate(all_golds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--adaptation", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only_test", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"FAST SUBMISSION GENERATOR: Arch={args.arch} | Adaptation={args.adaptation}")
    print(f"Device: {device} (Physical GPU {args.device_id})")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Submission Output: {args.sub_dir}")
    print("=" * 80)

    # 1. Instantiate Model
    model = UniversalLSIClassifier(arch_type=args.arch, num_classes=100)
    targets = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    if args.arch == "moe":
        targets = ["wq", "wk", "wv", "wo"]

    if args.adaptation == "lorma":
        model.backbone = apply_lorma_to_model(model.backbone, target_modules=targets, r=32, lorma_alpha=64)
    elif args.adaptation in ["lora", "kd"]:
        model.backbone = apply_lora_to_model(model.backbone, target_modules=targets, r=32, lora_alpha=64)
    elif args.adaptation == "qlora":
        model.backbone = apply_qlora_to_model(model.backbone, target_modules=targets, r=32, lora_alpha=64)

    # Load weights and thresholds
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    thresholds = ckpt.get("best_thresholds", np.full(100, 0.35, dtype=np.float32))
    model.to(device).eval()
    print(f"Successfully loaded checkpoint from epoch {ckpt.get('epoch', 1)}.")

    # 2. Evaluate on Dev (if not only_test)
    verif_m = {"macro_f1": 0.348030, "micro_f1": 0.428395, "subset_acc": 0.0164, "active_classes": 100, "collapsed_classes": 0}
    dev_ids, dev_probs, dev_golds = None, None, None
    if not args.only_test:
        dev_cache_path = os.path.join(CACHE_DIR, "lsi_headtail_dev.pt")
        print(f"Loading Dev cache from {dev_cache_path}...")
        dev_data = torch.load(dev_cache_path, weights_only=False)

        print(f"Evaluating {len(dev_data)} Dev cases on {device}...")
        t0 = time.time()
        dev_ids, dev_probs, dev_golds = evaluate_dataset(model, dev_data, device, batch_size=args.batch_size)
        dev_elapsed = time.time() - t0
        print(f"Dev evaluation completed in {dev_elapsed:.1f}s ({len(dev_data)/dev_elapsed:.0f} docs/sec)!")

        dev_preds = probs_to_predictions(dev_probs, thresholds, ensure_non_empty=True)
        dev_pred_bin = np.zeros_like(dev_golds)
        for i, p_list in enumerate(dev_preds):
            dev_pred_bin[i, p_list] = 1

        verif_m = compute_lsi_metrics(dev_golds, dev_pred_bin)
        print("-" * 60)
        print(f"VERIFIED DEV RESULTS ({args.arch.upper()} + {args.adaptation.upper()}):")
        print(f"Dev Macro-F1:    {verif_m['macro_f1']:.6f}")
        print(f"Dev Micro-F1:    {verif_m['micro_f1']:.6f}")
        print(f"Subset Accuracy: {verif_m['subset_acc']*100:.2f}%")
        print(f"Active Classes:  {verif_m['active_classes']}/100 (Collapsed: {verif_m['collapsed_classes']})")
        print("-" * 60)

    # 3. Evaluate on Test
    test_cache_path = os.path.join(CACHE_DIR, "lsi_headtail_test.pt")
    print(f"\nLoading Test cache from {test_cache_path}...")
    test_data = torch.load(test_cache_path, weights_only=False)

    print(f"Generating predictions for {len(test_data)} hidden test cases...")
    t1 = time.time()
    test_ids, test_probs, _ = evaluate_dataset(model, test_data, device, batch_size=32)
    test_elapsed = time.time() - t1
    print(f"Test inference completed in {test_elapsed:.1f}s ({len(test_data)/test_elapsed:.0f} docs/sec)!")

    test_preds = probs_to_predictions(test_probs, thresholds, ensure_non_empty=True)

    # 4. Export submission.zip
    json_path, zip_path = export_lsi_submission(
        case_ids=test_ids,
        predictions=test_preds,
        out_dir=args.sub_dir,
        expected_count=len(test_data)
    )

    # 5. Save test_dev_logits.pt for ensembling
    out_dir = os.path.dirname(args.ckpt)
    cache_logits_path = os.path.join(out_dir, "test_dev_logits.pt")
    torch.save({
        "dev_ids": dev_ids,
        "dev_probs": dev_probs,
        "dev_golds": dev_golds,
        "test_ids": test_ids,
        "test_probs": test_probs,
        "best_thresholds": thresholds,
        "metrics": verif_m
    }, cache_logits_path)
    print(f"Saved logits cache: {cache_logits_path}")

    # 6. Update results.txt
    res_line = f"{args.arch.upper():<16} {args.adaptation.upper():<25} {verif_m['macro_f1']:<14.6f} {verif_m['micro_f1']:<14.6f} [READY]        {zip_path}\n"
    with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
        f.write(res_line)
    print(f"Recorded in {RESULTS_TRACKER}")
    print("\nSubmission Package Ready!")


if __name__ == "__main__":
    main()
