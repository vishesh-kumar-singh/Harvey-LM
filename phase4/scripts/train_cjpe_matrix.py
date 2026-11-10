#!/usr/bin/env python3
"""
Phase 4: Unified Master Training Script for All 4 Foundation Architectures on CJPE.

Supports:
- Architectures: harveylm, baseline, moe, enc_dec
- Adaptation Techniques:
  1. lorma: Truncated SVD Low-Rank Matrix Approximation (Compulsory)
  2. qlora: 4-bit NormalFloat (NF4) Quantization + LoRA
  3. lora: Standard Low-Rank Adaptation
  4. full: Full model fine-tuning
  5. kd: Knowledge Distillation from teacher logits (Mandatory)
- Representation: Dense Head-Tail (2,048 tokens: 1,024 facts + 1,024 ratio)
- Loss: Logit-Adjusted Loss (tau=1.0) / KnowledgeDistillationLoss
- Post-hoc Calibration: Dynamic threshold delta* maximizing Binary Macro-F1
- Evaluation: Instant verification via evaluate_phase4.py against test_truth.json
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase4.src.data.cjpe_dataset import (
    load_raw_cjpe_data,
    CJPEDataset,
    cjpe_collate_fn_head_tail
)
from phase4.src.model.cjpe_classifier import (
    UniversalHeadTailClassifier,
    ARCH_CKPTS,
    ARCH_TARGETS,
    apply_lorma_to_model,
    mark_only_lorma_as_trainable,
    apply_lora_to_model,
    mark_only_lora_as_trainable,
    apply_qlora_to_model,
    mark_only_qlora_as_trainable
)
from phase4.src.evaluation.eval_utils import (
    compute_cjpe_metrics,
    find_optimal_threshold,
    export_cjpe_submission
)

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
DATA_DIR       = os.path.join(CODE_DIR, "phase4/data/raw/Phase_4_CJPE")
TEST_TRUTH     = os.path.join(CODE_DIR, "phase4/data/eval_truth/test_truth.json")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase4/results.txt")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits = []
    all_golds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"].float().cpu()
            all_logits.append(logits)
            all_ids.extend(batch["ids"])
            if "labels" in batch:
                all_golds.extend(batch["labels"].cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    return all_ids, all_logits, all_golds


def main():
    parser = argparse.ArgumentParser(description="Train Universal Head-Tail Model for Phase 4 CJPE")
    parser.add_argument("--arch", type=str, default="harveylm", choices=["harveylm", "baseline", "moe", "enc_dec"],
                        help="Foundation backbone architecture")
    parser.add_argument("--adaptation", type=str, default="lorma", choices=["lorma", "lora", "qlora", "full", "kd"],
                        help="Adaptation technique")
    parser.add_argument("--r", type=int, default=32, help="Rank for LORMA/LoRA/QLoRA")
    parser.add_argument("--alpha", type=int, default=64, help="Alpha for LORMA/LoRA/QLoRA")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Context sequence length (Head + Tail)")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per step")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--tau_la", type=float, default=1.0, help="Logit adjustment tau")
    parser.add_argument("--teacher_logits", type=str, default=None, help="Path to teacher logits file for KD")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output directory")
    parser.add_argument("--sub_dir", type=str, required=True, help="Submission zip output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Phase 4 CJPE: Architecture={args.arch.upper()} | Adaptation={args.adaptation.upper()}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output Directory: {args.output_dir}")
    print(f"Submission Directory: {args.sub_dir}")
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

    # 2. Load Raw CJPE Splits
    print("\n[2/7] Loading datasets with deterministic deduplication...")
    train_file = os.path.join(DATA_DIR, "Train", "train.json")
    dev_file   = os.path.join(DATA_DIR, "Train", "dev.json")
    test_file  = os.path.join(DATA_DIR, "Test", "test.json")

    raw_train = load_raw_cjpe_data(train_file, deduplicate=True)
    raw_dev   = load_raw_cjpe_data(dev_file,   deduplicate=True)
    raw_test  = load_raw_cjpe_data(test_file,  deduplicate=False)
    print(f"Loaded: Train={len(raw_train)} unique cases | Dev={len(raw_dev)} unique cases | Test={len(raw_test)} cases")

    # 3. Create Datasets and DataLoaders
    train_ds = CJPEDataset(
        raw_items=raw_train,
        tokenizer=tokenizer,
        mode="head_tail",
        max_seq_len=args.max_seq_len,
        is_training=True,
        anti_leakage=True,
        tail_drop_prob=0.50
    )
    dev_ds = CJPEDataset(
        raw_items=raw_dev,
        tokenizer=tokenizer,
        mode="head_tail",
        max_seq_len=args.max_seq_len,
        is_training=False,
        anti_leakage=False,
        tail_drop_prob=0.0
    )
    test_ds = CJPEDataset(
        raw_items=raw_test,
        tokenizer=tokenizer,
        mode="head_tail",
        max_seq_len=args.max_seq_len,
        is_training=False,
        anti_leakage=False,
        tail_drop_prob=0.0
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=cjpe_collate_fn_head_tail)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cjpe_collate_fn_head_tail)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cjpe_collate_fn_head_tail)

    # 4. Instantiate Universal Head-Tail Model
    print(f"\n[3/7] Instantiating UniversalHeadTailClassifier ({args.arch})...")
    model = UniversalHeadTailClassifier(
        arch_type=args.arch,
        num_classes=2,
        num_queries=4,
        head_hidden=1024,
        dropout=0.15,
        tau_la=args.tau_la
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
        # For KD student: we adapt the backbone with LoRA
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
    print("\n[5/7] Starting Training Loop with Mixed Precision...")
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_dev_macro_f1 = -1.0
    best_metrics = {}
    best_delta = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

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

            if (step + 1) % (25 * args.grad_accum) == 0:
                avg_l = running_loss / (step + 1)
                lr_cur = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {avg_l:.4f} | LR: {lr_cur:.2e}", flush=True)

        # Epoch Validation
        print(f"\n--- Epoch {epoch} Evaluation on Dev (2,511 unique judgments) ---")
        dev_ids, dev_logits, dev_golds = evaluate_model(model, dev_loader, device)

        # Raw (zero-offset) metrics
        raw_preds = (dev_logits[:, 1] - dev_logits[:, 0] > 0).astype(int).tolist()
        raw_m = compute_cjpe_metrics(dev_golds, raw_preds)
        print(f"[Epoch {epoch} Raw]   Macro-F1: {raw_m['macro_f1']:.4f} | F1_0: {raw_m['f1_0']:.4f} | F1_1: {raw_m['f1_1']:.4f} | Acc: {raw_m['accuracy']*100:.2f}%")

        # Calibrated threshold search (delta*)
        delta_star, calib_f1, calib_m = find_optimal_threshold(dev_golds, dev_logits)
        print(f"[Epoch {epoch} Calib] Macro-F1: {calib_m['macro_f1']:.4f} (delta*={delta_star:+.3f}) | F1_0: {calib_m['f1_0']:.4f} | F1_1: {calib_m['f1_1']:.4f} | Acc: {calib_m['accuracy']*100:.2f}%", flush=True)

        if calib_m["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = calib_m["macro_f1"]
            best_metrics = calib_m
            best_delta = delta_star

            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_macro_f1": best_dev_macro_f1,
                "best_delta": best_delta,
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
    print(f"Loaded Best Model from Epoch {best_ckpt['epoch']} (Best Dev Macro-F1: {best_ckpt['best_macro_f1']:.4f}, Delta*: {best_ckpt['best_delta']:+.3f})")

    print("\nGenerating Test Predictions (1,517 hidden cases)...")
    test_ids, test_logits, _ = evaluate_model(model, test_loader, device)

    # Apply calibrated threshold delta*
    diffs = test_logits[:, 1] - test_logits[:, 0]
    final_test_preds = (diffs + best_delta > 0).astype(int).tolist()

    class_0_cnt = sum(1 for p in final_test_preds if p == 0)
    class_1_cnt = sum(1 for p in final_test_preds if p == 1)
    print(f"Test Class Distribution: 0={class_0_cnt} ({class_0_cnt/len(final_test_preds)*100:.2f}%) | 1={class_1_cnt} ({class_1_cnt/len(final_test_preds)*100:.2f}%)")

    json_path, zip_path = export_cjpe_submission(
        case_ids=test_ids,
        predictions=final_test_preds,
        out_dir=args.sub_dir,
        expected_count=1517
    )

    # Save Dev Logits & Test Logits for Future Ensembling
    logits_cache_path = os.path.join(args.output_dir, "test_dev_logits.pt")
    _, dev_logits, dev_golds = evaluate_model(model, dev_loader, device)
    torch.save({
        "dev_ids": dev_ids,
        "dev_logits": dev_logits,
        "dev_golds": dev_golds,
        "test_ids": test_ids,
        "test_logits": test_logits,
        "best_delta": best_delta,
        "metrics": best_metrics
    }, logits_cache_path)
    print(f"Saved evaluation logits cache: {logits_cache_path}")

    # 9. Verify with evaluate_phase4.py against test_truth.json
    print("\n[7/7] Verifying Submission against Official Benchmark Ground Truth...")
    if os.path.exists(TEST_TRUTH):
        from phase4.src.evaluation.eval_utils import load_ground_truth
        test_truth_dict = load_ground_truth(TEST_TRUTH)
        test_golds = [test_truth_dict[cid] for cid in test_ids]
        test_m = compute_cjpe_metrics(test_golds, final_test_preds)

        print("-" * 60)
        print(f"TEST BENCHMARK RESULTS ({args.arch.upper()} + {args.adaptation.upper()}):")
        print(f"Macro-F1:  {test_m['macro_f1']:.6f}")
        print(f"Precision: {test_m['precision_1']:.6f}")
        print(f"Recall:    {test_m['recall_1']:.6f}")
        print(f"Accuracy:  {test_m['accuracy']*100:.2f}%")
        print("-" * 60)

        # Append to results.txt
        res_line = f"{args.arch.upper():<16} {args.adaptation.upper():<25} {best_dev_macro_f1:<14.6f} {test_m['macro_f1']:<14.6f} {test_m['accuracy']*100:.2f}%        {zip_path}\n"
        with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
            f.write(res_line)
        print(f"Logged result to {RESULTS_TRACKER}")

    print("\nTraining Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
