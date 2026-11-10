#!/usr/bin/env python3
"""
Phase 4: Master Training Script for HarveyLM on Court Judgment Prediction (CJPE).

Key Capabilities:
1. Long-Document Architecture:
   - "head_tail": Dense 2,048-token Head-Tail Slicing + Multi-Query Attention Pooling.
   - "hierarchical": 8 x 1,024-token Multi-Chunk Transformer + Multi-Query Bottleneck.
2. Compulsory LORMA (Truncated SVD) and LoRA / QLoRA parameter-efficient adaptation.
3. Anti-Leakage Scrubbing: Dispositive verdict trigger masking + stochastic tail-drop.
4. Logit-Adjusted Loss (tau=1.0) under Train (58.6/41.4) vs Dev (50/50) prior shift.
5. Post-Hoc Threshold Optimization (delta*) directly maximizing unweighted binary Macro-F1.
6. Local Verification against code/Evaluation Script/evaluate_phase4.py and export of submission.zip.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

# Set project paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, os.path.join(ROOT_DIR, "code"))

from phase4.src.data.cjpe_dataset import (
    load_raw_cjpe_data,
    CJPEDataset,
    cjpe_collate_fn_head_tail,
    cjpe_collate_fn_chunked
)
from phase4.src.model.cjpe_classifier import (
    HarveyHeadTailClassifier,
    HarveyHierarchicalClassifier,
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

HARVEYLM_CKPT = os.path.join(ROOT_DIR, "code/phase1/models/final_architectures/2_HarveyLM/epoch_99")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
DATA_DIR = os.path.join(ROOT_DIR, "code/phase4/data/raw/Phase_4_CJPE")
DEV_TRUTH_PATH = os.path.join(ROOT_DIR, "code/phase4/data/eval_truth/dev_truth.json")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, dataloader, device, is_hierarchical: bool = False):
    model.eval()
    all_logits = []
    all_golds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if not is_hierarchical:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
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
    parser = argparse.ArgumentParser(description="Train HarveyLM for Court Judgment Prediction (Phase 4)")
    parser.add_argument("--mode", type=str, default="head_tail", choices=["head_tail", "hierarchical"],
                        help="Document encoding mode: head_tail (2048 tok) or hierarchical (M x 1024 tok)")
    parser.add_argument("--adaptation", type=str, default="lorma", choices=["lorma", "lora", "qlora"],
                        help="Parameter-efficient adaptation method (LORMA is compulsory)")
    parser.add_argument("--r", type=int, default=32, help="Rank for LORMA/LoRA adapters")
    parser.add_argument("--alpha", type=int, default=64, help="Alpha scale for LORMA/LoRA")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for head_tail mode")
    parser.add_argument("--max_chunks", type=int, default=8, help="Max chunks for hierarchical mode")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--tau_la", type=float, default=1.0, help="Logit adjustment tau for class imbalance")
    parser.add_argument("--output_dir", type=str, default="code/phase4/models/harveylm/lorma_headtail",
                        help="Model checkpoint output directory")
    parser.add_argument("--sub_dir", type=str, default="code/phase4/submissions/harveylm/lorma_headtail",
                        help="Submission archive directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Phase 4 Court Judgment Prediction: HarveyLM ({args.mode.upper()} + {args.adaptation.upper()})")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output Directory: {args.output_dir}")
    print(f"Submission Directory: {args.sub_dir}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    # 1. Load Tokenizer
    print(f"Loading legal BPE tokenizer from {TOKENIZER_PATH}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 3
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 1
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 2

    # 2. Load Raw CJPE Splits with Deduplication
    train_file = os.path.join(DATA_DIR, "Train", "train.json")
    dev_file   = os.path.join(DATA_DIR, "Train", "dev.json")
    test_file  = os.path.join(DATA_DIR, "Test", "test.json")

    print("\nLoading datasets with deterministic deduplication...")
    raw_train = load_raw_cjpe_data(train_file, deduplicate=True)
    raw_dev   = load_raw_cjpe_data(dev_file,   deduplicate=True)
    raw_test  = load_raw_cjpe_data(test_file,  deduplicate=False) # test is already unique 1,517 cases

    print(f"Loaded: Train={len(raw_train)} unique cases | Dev={len(raw_dev)} unique cases | Test={len(raw_test)} cases")

    # 3. Create Datasets and DataLoaders
    is_hier = (args.mode == "hierarchical")
    train_ds = CJPEDataset(
        raw_items=raw_train,
        tokenizer=tokenizer,
        mode=args.mode,
        max_seq_len=args.max_seq_len,
        max_chunks=args.max_chunks,
        is_training=True,
        anti_leakage=True,
        tail_drop_prob=0.50
    )
    dev_ds = CJPEDataset(
        raw_items=raw_dev,
        tokenizer=tokenizer,
        mode=args.mode,
        max_seq_len=args.max_seq_len,
        max_chunks=args.max_chunks,
        is_training=False,
        anti_leakage=False,
        tail_drop_prob=0.0
    )
    test_ds = CJPEDataset(
        raw_items=raw_test,
        tokenizer=tokenizer,
        mode=args.mode,
        max_seq_len=args.max_seq_len,
        max_chunks=args.max_chunks,
        is_training=False,
        anti_leakage=False,
        tail_drop_prob=0.0
    )

    collate_fn = cjpe_collate_fn_chunked if is_hier else cjpe_collate_fn_head_tail

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 4. Instantiate Model
    print(f"\nInstantiating HarveyLM model (mode={args.mode})...")
    if not is_hier:
        model = HarveyHeadTailClassifier(
            ckpt_dir=HARVEYLM_CKPT,
            num_classes=2,
            num_queries=4,
            dropout=0.15,
            tau_la=args.tau_la
        )
    else:
        model = HarveyHierarchicalClassifier(
            ckpt_dir=HARVEYLM_CKPT,
            num_classes=2,
            max_chunks=args.max_chunks,
            num_queries=4,
            dropout=0.15,
            tau_la=args.tau_la
        )

    # 5. Inject Parameter-Efficient Adaptation (LORMA / LoRA / QLoRA)
    target_modules = ["w1", "w2", "w3"]
    # attention projections in TransformerBlock.attention
    # In HarveyLM: attention has wq, wk, wv, wo
    attn_targets = ["wq", "wk", "wv", "wo"]
    all_targets = target_modules + attn_targets

    print(f"\nApplying adaptation ({args.adaptation.upper()}) to backbone layers...")
    if args.adaptation == "lorma":
        model.backbone = apply_lorma_to_model(
            model.backbone,
            target_modules=all_targets,
            r=args.r,
            lorma_alpha=args.alpha,
            dropout=0.05
        )
        mark_only_lorma_as_trainable(model.backbone)
    elif args.adaptation == "lora":
        model.backbone = apply_lora_to_model(
            model.backbone,
            target_modules=all_targets,
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05
        )
        mark_only_lora_as_trainable(model.backbone)
    elif args.adaptation == "qlora":
        model.backbone = apply_qlora_to_model(
            model.backbone,
            target_modules=all_targets,
            r=args.r,
            lora_alpha=args.alpha,
            block_size=64
        )
        mark_only_qlora_as_trainable(model.backbone)

    model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 6. Optimizer and LR Scheduler
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 2.5}
    ], weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Total training steps: {total_steps} (warmup={warmup_steps})")

    # 7. Training Loop with Mixed Precision
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_dev_macro_f1 = -1.0
    best_metrics = {}
    best_delta = 0.0

    print("\nStarting Training Pipeline...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if not is_hier:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    chunk_ids = batch["chunk_input_ids"].to(device)
                    chunk_masks = batch["chunk_masks"].to(device)
                    doc_masks = batch["doc_chunk_masks"].to(device)
                    out = model(chunk_input_ids=chunk_ids, chunk_masks=chunk_masks, doc_chunk_masks=doc_masks, labels=labels)

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
                step_count += 1

            if (step + 1) % (25 * args.grad_accum) == 0:
                avg_l = running_loss / (step + 1)
                lr_cur = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {avg_l:.4f} | LR: {lr_cur:.2e}", flush=True)

        # Epoch Validation
        print(f"\n--- Epoch {epoch} Evaluation on Dev (2,511 unique judgments) ---")
        dev_ids, dev_logits, dev_golds = evaluate_model(model, dev_loader, device, is_hierarchical=is_hier)

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

    # 8. Final Best Model Test Inference & Submission Generation
    print("\n" + "=" * 80)
    print("FINAL BEST MODEL EVALUATION & TEST SUBMISSION GENERATION")
    print("=" * 80)
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded Best Model from Epoch {best_ckpt['epoch']} (Best Dev Macro-F1: {best_ckpt['best_macro_f1']:.4f}, Delta*: {best_ckpt['best_delta']:+.3f})")

    print("\nGenerating Test Predictions (1,517 hidden cases)...")
    test_ids, test_logits, _ = evaluate_model(model, test_loader, device, is_hierarchical=is_hier)

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
    _, dev_logits, dev_golds = evaluate_model(model, dev_loader, device, is_hierarchical=is_hier)
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
    print("\n Training Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
