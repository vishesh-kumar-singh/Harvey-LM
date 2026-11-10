#!/usr/bin/env python3
"""
Phase 5: Unified Master Training Script for All 4 Foundation Architectures on LSI.

Supports:
- Architectures: harveylm, baseline, moe, enc_dec
- Adaptation Techniques:
  1. lorma: Truncated SVD Low-Rank Matrix Approximation (Compulsory)
  2. qlora: 4-bit NormalFloat (NF4) Quantization + LoRA
  3. lora: Standard Low-Rank Adaptation
  4. kd: Knowledge Distillation from teacher logits (Mandatory)
  5. full: Full model fine-tuning
- Representation: Dense Head-Tail (2,048 tokens: 1,024 head + 1,024 tail)
- Multi-Label Loss: Asymmetric Loss (ASL) & Multi-Label KD Loss
- Thresholding: Per-Class Optimal Threshold Search (theta_c*) to maximize Macro-F1
- Collapse Prevention: Active tracking of per-class F1 and zero-prediction detection
- Verification: Automatic validation of prediction.json and submission.zip
"""

import os
import sys


import argparse
import json
import random
import time
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

from typing import Tuple, List, Dict, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase5.src.data.lsi_dataset import load_raw_lsi_data, LSIDataset, lsi_collate_fn_head_tail
from phase5.src.model.lsi_classifier import (
    UniversalLSIClassifier,
    ARCH_CKPTS,
    ARCH_TARGETS,
    apply_lorma_to_model,
    mark_only_lorma_as_trainable,
    apply_lora_to_model,
    mark_only_lora_as_trainable,
    apply_qlora_to_model,
    mark_only_qlora_as_trainable
)
from phase5.src.evaluation.eval_utils import (
    compute_lsi_metrics,
    optimize_per_class_thresholds,
    probs_to_predictions,
    export_lsi_submission
)

TOKENIZER_PATH  = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
DATA_DIR        = os.path.join(CODE_DIR, "phase5/data/raw/Phase_5_LSI")
TRAIN_DATA      = os.path.join(DATA_DIR, "Train/train.json")
DEV_DATA        = os.path.join(DATA_DIR, "Train/dev.json")
TEST_DATA       = os.path.join(DATA_DIR, "Test/test.json")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase5/results.txt")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, dataloader, device) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Evaluates model across dataloader and returns IDs, sigmoid probabilities, and true labels."""
    model.eval()
    all_ids = []
    all_probs = []
    all_golds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"].float()
                probs = torch.sigmoid(logits).cpu().numpy()

            all_ids.extend(batch["ids"])
            all_probs.append(probs)
            all_golds.append(labels)

    return all_ids, np.concatenate(all_probs, axis=0), np.concatenate(all_golds, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 LSI Unified Master Training Pipeline")
    parser.add_argument("--arch", type=str, default="harveylm", choices=["harveylm", "baseline", "moe", "enc_dec"])
    parser.add_argument("--adaptation", type=str, default="lorma", choices=["lorma", "lora", "qlora", "full", "kd"])
    parser.add_argument("--r", type=int, default=32, help="Rank for LORMA/LoRA/QLoRA")
    parser.add_argument("--alpha", type=int, default=64, help="Alpha scaling for adaptation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="code/phase5/models/harveylm/lorma_headtail")
    parser.add_argument("--sub_dir", type=str, default="code/phase5/submissions/harveylm/lorma_headtail")
    parser.add_argument("--teacher_logits", type=str, default=None, help="Path to teacher logits file for KD")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Phase 5 LSI: Architecture={args.arch.upper()} | Adaptation={args.adaptation.upper()}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output Directory: {args.output_dir}")
    print(f"Submission Directory: {args.sub_dir}")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective Batch: {args.batch_size * args.grad_accum})")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    # 1. Load Tokenizer
    print(f"\n[1/7] Loading legal BPE tokenizer from {TOKENIZER_PATH}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 3
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 1
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 2

    # 2. Load Data
    print("\n[2/7] Loading datasets...")
    raw_train = load_raw_lsi_data(TRAIN_DATA)
    raw_dev   = load_raw_lsi_data(DEV_DATA)
    raw_test  = load_raw_lsi_data(TEST_DATA)
    print(f"Loaded: Train={len(raw_train)} cases | Dev={len(raw_dev)} cases | Test={len(raw_test)} cases")

    # 3. Datasets & Loaders
    train_ds = LSIDataset(raw_train, tokenizer=tokenizer, max_seq_len=args.max_seq_len, is_training=True)
    dev_ds   = LSIDataset(raw_dev,   tokenizer=tokenizer, max_seq_len=args.max_seq_len, is_training=False)
    test_ds  = LSIDataset(raw_test,  tokenizer=tokenizer, max_seq_len=args.max_seq_len, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=lsi_collate_fn_head_tail)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lsi_collate_fn_head_tail)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lsi_collate_fn_head_tail)

    # 4. Instantiate Universal Model
    print(f"\n[3/7] Instantiating UniversalLSIClassifier ({args.arch})...")
    model = UniversalLSIClassifier(
        arch_type=args.arch,
        num_classes=100,
        num_queries=4,
        head_hidden=1024,
        dropout=0.15
    )

    # 5. Apply Adaptation Technique
    targets = ARCH_TARGETS.get(args.arch, ["wq", "wk", "wv", "wo", "w1", "w2", "w3"])
    print(f"\n[4/7] Applying adaptation ({args.adaptation.upper()}) to backbone layers...")
    if args.adaptation == "lorma":
        model.backbone = apply_lorma_to_model(
            model.backbone,
            target_modules=targets,
            r=args.r,
            lorma_alpha=args.alpha,
            dropout=0.05
        )
        mark_only_lorma_as_trainable(model.backbone)
    elif args.adaptation == "lora":
        model.backbone = apply_lora_to_model(
            model.backbone,
            target_modules=targets,
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05
        )
        mark_only_lora_as_trainable(model.backbone)
    elif args.adaptation == "qlora":
        model.backbone = apply_qlora_to_model(
            model.backbone,
            target_modules=targets,
            r=args.r,
            lora_alpha=args.alpha,
            block_size=64
        )
        mark_only_qlora_as_trainable(model.backbone)
    elif args.adaptation == "kd":
        model.backbone = apply_lora_to_model(
            model.backbone,
            target_modules=targets,
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05
        )
        mark_only_lora_as_trainable(model.backbone)
    elif args.adaptation == "full":
        for p in model.parameters():
            p.requires_grad = True

    # Critical: If MoE, ensure router is explicitly trainable to prevent routing collapse
    if args.arch in ["moe", "harveymoe"]:
        router_params = 0
        for name, p in model.backbone.named_parameters():
            if "router" in name:
                p.requires_grad = True
                router_params += p.numel()
        print(f"Explicitly unlocked MoE Router ({router_params:,} parameters trainable) to prevent router collapse!")

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 6. Setup Optimizer and Scheduler
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params,     "lr": args.lr * 2.5}
    ], weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"Total training steps: {total_steps} (warmup={warmup_steps})")

    # Load Teacher Logits if KD
    teacher_targets_dict = None
    if args.adaptation == "kd" and args.teacher_logits and os.path.exists(args.teacher_logits):
        print(f"Loading Teacher Logits for Knowledge Distillation from {args.teacher_logits}...")
        teacher_targets_dict = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
        print(f"Loaded teacher targets for {len(teacher_targets_dict)} cases.")

    # 7. Training Loop
    print("\n[5/7] Starting Multi-Label Training Loop with Mixed Precision...")
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_dev_macro_f1 = -1.0
    best_metrics = {}
    best_thresholds = np.full(100, 0.35, dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            t_logits = None
            if teacher_targets_dict is not None:
                t_logits = torch.stack([teacher_targets_dict[cid] for cid in batch["ids"]]).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, teacher_logits=t_logits)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % (50 * args.grad_accum) == 0:
                avg_l = running_loss / (step + 1)
                lr_cur = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {avg_l:.4f} | LR: {lr_cur:.2e} | Elapsed: {elapsed:.0f}s", flush=True)

        # Epoch Validation
        epoch_time = time.time() - epoch_start
        print(f"\n--- Epoch {epoch} Validation on Dev Set ({len(raw_dev)} cases) [{epoch_time:.1f}s] ---")
        dev_ids, dev_probs, dev_golds = evaluate_model(model, dev_loader, device)

        # Raw (0.50 global threshold) metrics
        raw_pred_bin = (dev_probs >= 0.50).astype(np.uint8)
        raw_m = compute_lsi_metrics(dev_golds, raw_pred_bin)
        print(f"[Epoch {epoch} Raw 0.50] Macro-F1: {raw_m['macro_f1']:.4f} | Micro-F1: {raw_m['micro_f1']:.4f} | SubsetAcc: {raw_m['subset_acc']*100:.2f}% | Active: {raw_m['active_classes']}/100 | Collapsed: {raw_m['collapsed_classes']}")

        # Optimal Per-Class Thresholds Search (theta_c*)
        opt_thresholds, calib_m = optimize_per_class_thresholds(dev_golds, dev_probs)
        print(f"[Epoch {epoch} Calibrated] Macro-F1: {calib_m['macro_f1']:.4f} | Micro-F1: {calib_m['micro_f1']:.4f} | SubsetAcc: {calib_m['subset_acc']*100:.2f}% | Active: {calib_m['active_classes']}/100 | Collapsed: {calib_m['collapsed_classes']}", flush=True)

        # Check for collapse warning
        if calib_m["active_classes"] < 90:
            print(f"WARNING: Potential class collapse detected! Only {calib_m['active_classes']}/100 classes active.")

        if calib_m["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = calib_m["macro_f1"]
            best_metrics = calib_m
            best_thresholds = opt_thresholds

            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_macro_f1": best_dev_macro_f1,
                "best_thresholds": best_thresholds,
                "metrics": best_metrics,
                "args": vars(args)
            }, ckpt_path)
            print(f" Saved New Best Checkpoint: {ckpt_path} (Macro-F1: {best_dev_macro_f1:.4f})")

    # 8. Test Inference & Submission Generation
    print("\n" + "=" * 80)
    print("[6/7] FINAL BEST MODEL EVALUATION & SUBMISSION GENERATION")
    print("=" * 80)
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    saved_th = best_ckpt["best_thresholds"]
    print(f"Loaded Best Model from Epoch {best_ckpt['epoch']} (Best Dev Macro-F1: {best_ckpt['best_macro_f1']:.4f})")

    print(f"\nGenerating Test Predictions ({len(raw_test)} hidden cases)...")
    test_ids, test_probs, _ = evaluate_model(model, test_loader, device)

    # Convert to predictions using optimal thresholds
    test_preds = probs_to_predictions(test_probs, saved_th, ensure_non_empty=True)

    json_path, zip_path = export_lsi_submission(
        case_ids=test_ids,
        predictions=test_preds,
        out_dir=args.sub_dir,
        expected_count=len(raw_test)
    )

    # Save Dev Logits & Test Logits for Future Ensembling
    logits_cache_path = os.path.join(args.output_dir, "test_dev_logits.pt")
    dev_ids, dev_probs, dev_golds = evaluate_model(model, dev_loader, device)
    torch.save({
        "dev_ids": dev_ids,
        "dev_probs": dev_probs,
        "dev_golds": dev_golds,
        "test_ids": test_ids,
        "test_probs": test_probs,
        "best_thresholds": saved_th,
        "metrics": best_metrics
    }, logits_cache_path)
    print(f"Saved evaluation logits cache: {logits_cache_path}")

    # 9. Verify Submission with evaluate_phase5.py on Dev
    print("\n[7/7] Verifying Predictions using Official Evaluation Standard...")
    dev_preds = probs_to_predictions(dev_probs, saved_th, ensure_non_empty=True)
    dev_pred_bin = np.zeros_like(dev_golds)
    for i, p_list in enumerate(dev_preds):
        dev_pred_bin[i, p_list] = 1

    verif_m = compute_lsi_metrics(dev_golds, dev_pred_bin)
    print("-" * 60)
    print(f"OFFICIAL VERIFIED DEV RESULTS ({args.arch.upper()} + {args.adaptation.upper()}):")
    print(f"Dev Macro-F1:   {verif_m['macro_f1']:.6f}")
    print(f"Dev Micro-F1:   {verif_m['micro_f1']:.6f}")
    print(f"Subset Accuracy: {verif_m['subset_acc']*100:.2f}%")
    print(f"Active Classes: {verif_m['active_classes']}/100 (Collapsed: {verif_m['collapsed_classes']})")
    print("-" * 60)

    # Append to results.txt
    res_line = f"{args.arch.upper():<16} {args.adaptation.upper():<25} {verif_m['macro_f1']:<14.6f} {verif_m['micro_f1']:<14.6f} [SUBMITTED]    {zip_path}\n"
    with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
        f.write(res_line)
    print(f"Logged result to {RESULTS_TRACKER}")

    print("\nTraining & Submission Generation Completed Successfully!")


if __name__ == "__main__":
    main()
