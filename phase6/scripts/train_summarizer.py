#!/usr/bin/env python3
"""
Phase 6: Unified Master Training Script for Abstractive Legal Summarization.

Supports:
- Backbones: enc_dec (LegalEncoderDecoder), harveylm, moe, baseline
- Adaptation: lorma (Compulsory), lora, qlora, kd, full
- Sequence Budgeting: Head-Tail (1024 head + 1024 tail tokens = 2048 input context)
- Controlled Generation: Repetition penalty + Trigram blocking + EOS stopping
- Metric Tracking: BERTScore F1 (Primary), ROUGE-1 F1, ROUGE-L F1 (Secondary)
- Automatic Packaging & Verification: prediction.json & submission.zip
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
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase6.src.data.summarization_dataset import (
    load_summarization_data,
    LegalSummarizationDataset,
    summarization_collate_fn
)
from phase6.src.model.summarizer import UniversalLegalSummarizer, ARCH_CKPTS
from phase6.src.evaluation.eval_utils import compute_summarization_metrics, export_phase6_submission

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
TRAIN_DATA_PATH = os.path.join(CODE_DIR, "phase6/data/processed/train.json")
DEV_DATA_PATH = os.path.join(CODE_DIR, "phase6/data/processed/dev.json")
DEV_TRUTH_PATH = os.path.join(CODE_DIR, "phase6/data/eval_truth/dev_truth.json")
TEST_DATA_PATH = os.path.join(CODE_DIR, "phase6/data/raw/Phase_6_Summ/Test/test.json")
RESULTS_TRACKER = os.path.join(CODE_DIR, "phase6/results.txt")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_summaries_from_loader(
    model: UniversalLegalSummarizer,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    max_new_tokens: int = 768,
    max_samples: Optional[int] = None,
    desc: str = "Generating",
    dynamic_holding_injection: bool = True
) -> Dict[str, str]:
    """Generates decoded text summaries across batches with tqdm and live per-sample logging."""
    model.eval()
    predictions = {}
    total_processed = 0
    t_start = time.time()
    target_total = max_samples if max_samples else len(dataloader)

    holding_prompt = "\n\nDecision and Holding:\n"
    holding_prompt_ids = tokenizer.encode(holding_prompt, add_special_tokens=False)

    pbar = tqdm(total=target_total, desc=desc, unit="doc", dynamic_ncols=True)

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_token_lists = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    dynamic_holding_injection=dynamic_holding_injection,
                    holding_prompt_ids=holding_prompt_ids,
                    holding_injection_step=min(620, max(50, max_new_tokens - 120))
                )

            for doc_id, tokens in zip(ids, generated_token_lists):
                decoded_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if not decoded_text:
                    decoded_text = "The court delivered its decision on the legal appeal."
                # Clean punctuation spacing and tokenization artifacts
                decoded_text = decoded_text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
                decoded_text = decoded_text.replace(" ( ", " (").replace(" ) ", ") ").replace(" )", ")")
                decoded_text = decoded_text.replace(" ' ", "'").replace(" s ", "'s ").replace(" 's", "'s")
                decoded_text = decoded_text.replace("Un divided", "Undivided")
                if len(decoded_text) > 200:
                    last_p = max(decoded_text.rfind("."), decoded_text.rfind(";"))
                    if last_p > len(decoded_text) * 0.75:
                        decoded_text = decoded_text[: last_p + 1].strip()
                predictions[doc_id] = decoded_text
                total_processed += 1
                pbar.update(1)

                elapsed = time.time() - t_start
                avg_speed = elapsed / total_processed
                remaining = (target_total - total_processed) * avg_speed
                print(f"  [{desc} {total_processed}/{target_total}] Doc: {doc_id} | Tokens: {len(tokens)} | Speed: {avg_speed:.1f}s/doc | ETA: {remaining:.0f}s", flush=True)

                if max_samples and total_processed >= max_samples:
                    pbar.close()
                    return predictions

    pbar.close()
    return predictions


def log_result_to_tracker(
    arch: str,
    adaptation: str,
    bertscore: float,
    rouge1: float,
    rougel: float,
    zip_path: str
):
    """Appends evaluated experiment score to code/phase6/results.txt."""
    if not os.path.exists(RESULTS_TRACKER):
        return

    rel_zip = os.path.relpath(zip_path, ROOT_DIR)
    record_line = f"{arch.upper():<16} {adaptation.upper():<25} {bertscore:<14.6f} {rouge1:<12.6f} {rougel:<12.6f} {'[SUBMITTED]':<14} {rel_zip}\n"

    try:
        with open(RESULTS_TRACKER, "a", encoding="utf-8") as f:
            f.write(record_line)
        print(f"Logged score to {RESULTS_TRACKER}")
    except Exception as e:
        print(f"Warning: Failed to update tracker: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Legal Summarization Master Trainer")
    parser.add_argument("--arch", type=str, default="enc_dec", choices=["enc_dec", "harveylm", "moe", "baseline"])
    parser.add_argument("--adaptation", type=str, default="lorma", choices=["lorma", "lora", "qlora", "kd", "full"])
    parser.add_argument("--r", type=int, default=32, help="Rank for LORMA/LoRA/QLoRA")
    parser.add_argument("--alpha", type=int, default=64, help="Alpha scaling")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--max_doc_len", type=int, default=2048)
    parser.add_argument("--head_tokens", type=int, default=1024)
    parser.add_argument("--tail_tokens", type=int, default=1024)
    parser.add_argument("--use_triple_anchor", action="store_true", default=True, help="Use 3-anchor salience budgeting")
    parser.add_argument("--max_summary_len", type=int, default=768)
    parser.add_argument("--gen_tokens", type=int, default=768)
    parser.add_argument("--eval_samples", type=int, default=20, help="Dev samples for mid-training epoch validation")
    parser.add_argument("--val_tokens", type=int, default=128, help="Max generated tokens for mid-epoch validation")
    parser.add_argument("--final_dev_samples", type=int, default=100, help="Dev samples for final evaluation scorecard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for dry runs")
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="Path to teacher checkpoint for KD")
    parser.add_argument("--teacher_arch", type=str, default="harveylm", help="Teacher architecture for KD")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"PHASE 6: ABSTRACTIVE LEGAL SUMMARIZATION MASTER RUNNER")
    print(f"Arch: {args.arch} | Adaptation: {args.adaptation} | Rank: {args.r} | Alpha: {args.alpha}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} (Accum: {args.grad_accum}) | LR: {args.lr}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output: {args.output_dir} | Submissions: {args.sub_dir}")
    print("=" * 80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    # 1. Load Tokenizer
    print(f"Loading course BPE tokenizer: {TOKENIZER_PATH}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 3
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    # 2. Load Datasets
    print("Loading Train, Dev, and Test data...")
    raw_train = load_summarization_data(TRAIN_DATA_PATH)
    raw_dev = load_summarization_data(DEV_DATA_PATH)
    raw_test = load_summarization_data(TEST_DATA_PATH)
    print(f"Loaded: Train={len(raw_train)} cases | Dev={len(raw_dev)} cases | Test={len(raw_test)} cases")

    if args.max_train_samples:
        raw_train = raw_train[:args.max_train_samples]
        print(f"Truncated training data to {len(raw_train)} cases for dry run.")

    train_dataset = LegalSummarizationDataset(
        raw_train,
        tokenizer=tokenizer,
        model_type=args.arch,
        max_doc_len=args.max_doc_len,
        head_tokens=args.head_tokens,
        tail_tokens=args.tail_tokens,
        max_summary_len=args.max_summary_len,
        use_triple_anchor=args.use_triple_anchor,
        is_test=False
    )
    dev_dataset = LegalSummarizationDataset(
        raw_dev,
        tokenizer=tokenizer,
        model_type=args.arch,
        max_doc_len=args.max_doc_len,
        head_tokens=args.head_tokens,
        tail_tokens=args.tail_tokens,
        max_summary_len=args.max_summary_len,
        use_triple_anchor=args.use_triple_anchor,
        is_test=True  # Generation mode: feed document prompt only (prevents full 2048 token prompt overflow)
    )
    test_dataset = LegalSummarizationDataset(
        raw_test,
        tokenizer=tokenizer,
        model_type=args.arch,
        max_doc_len=args.max_doc_len,
        head_tokens=args.head_tokens,
        tail_tokens=args.tail_tokens,
        max_summary_len=args.max_summary_len,
        use_triple_anchor=args.use_triple_anchor,
        is_test=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=summarization_collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=summarization_collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=summarization_collate_fn, num_workers=2)

    # Load Ground Truth Dev References for BERTScore evaluation
    with open(DEV_TRUTH_PATH, "r", encoding="utf-8") as f:
        dev_gold_truth = json.load(f)

    # 3. Instantiate Universal Legal Summarizer
    model = UniversalLegalSummarizer(
        arch_type=args.arch,
        adaptation=args.adaptation,
        r=args.r,
        lorma_alpha=args.alpha
    ).to(device)

    # Optional Teacher Model for Knowledge Distillation
    teacher_model = None
    if args.adaptation == "kd" and args.teacher_ckpt:
        print(f"Loading Teacher Model ({args.teacher_arch}) from {args.teacher_ckpt}...")
        teacher_model = UniversalLegalSummarizer(arch_type=args.teacher_arch, adaptation="lorma").to(device)
        t_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        teacher_model.load_state_dict(t_ckpt["model_state_dict"], strict=False)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print("Teacher model loaded and frozen successfully.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameter Profile:")
    print(f"  Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda")
    best_bertscore = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    # 5. Training Loop
    print(f"\nStarting Training: {args.epochs} Epochs | Total Optimization Steps: {total_steps}...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else None
            dec_inputs = batch["decoder_input_ids"].to(device) if "decoder_input_ids" in batch else None

            teacher_logits = None
            if teacher_model is not None:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        t_out = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=dec_inputs
                        )
                        teacher_logits = t_out["logits"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_inputs,
                    labels=labels,
                    teacher_logits=teacher_logits
                )
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

            if step % (args.grad_accum * 50) == 0:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                avg_step_loss = epoch_loss / step
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.0f}s")

        # 6. Mid-training Validation on Dev Set
        print(f"\n--- Epoch {epoch} Validation on Dev Set ({args.eval_samples} samples, max {args.val_tokens} tokens) ---", flush=True)
        val_start = time.time()
        dev_preds = generate_summaries_from_loader(
            model,
            dev_loader,
            tokenizer,
            device,
            max_new_tokens=args.val_tokens,
            max_samples=args.eval_samples,
            desc=f"Epoch {epoch} Val"
        )
        val_metrics = compute_summarization_metrics(dev_preds, dev_gold_truth, device="cuda")
        val_elapsed = time.time() - val_start
        print(f"[Epoch {epoch} Dev Metrics] ({val_elapsed:.1f}s):", flush=True)
        print(f"  BERTScore F1: {val_metrics['bertscore_f1']:.6f} (Primary)", flush=True)
        print(f"  ROUGE-1 F1:   {val_metrics['rouge1_f1']:.6f}", flush=True)
        print(f"  ROUGE-L F1:   {val_metrics['rougel_f1']:.6f}", flush=True)

        if val_metrics["bertscore_f1"] > best_bertscore:
            best_bertscore = val_metrics["bertscore_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad or "W_res" in k or "lora" in k or "lorma" in k},
                "bertscore_f1": best_bertscore,
                "val_metrics": val_metrics,
                "args": vars(args)
            }, best_ckpt_path)
            print(f" Saved New Best Checkpoint: {best_ckpt_path} (BERTScore F1: {best_bertscore:.6f})", flush=True)

    # 7. Final Test Generation & Submission Packaging
    print("\n" + "=" * 80, flush=True)
    print("FINAL EVALUATION & OFFICIAL SUBMISSION PACKAGING", flush=True)
    print("=" * 80, flush=True)

    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from: {best_ckpt_path}", flush=True)
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print(f"\nGenerating Summaries for 100 Hidden Test Cases (max_tokens={args.gen_tokens})...", flush=True)
    test_gen_start = time.time()
    test_predictions = generate_summaries_from_loader(
        model,
        test_loader,
        tokenizer,
        device,
        max_new_tokens=args.gen_tokens,
        max_samples=None,
        desc="Test Set"
    )
    print(f"Generated {len(test_predictions)} test summaries in {time.time() - test_gen_start:.1f}s.", flush=True)

    # Export submission.zip directly at root
    zip_path = export_phase6_submission(test_predictions, args.sub_dir)

    # 8. Complete Dev Set Evaluation & Logging
    print(f"\nEvaluating Held-Out Dev Set ({args.final_dev_samples} cases) for Final Benchmark Scorecard...", flush=True)
    full_dev_preds = generate_summaries_from_loader(
        model,
        dev_loader,
        tokenizer,
        device,
        max_new_tokens=args.gen_tokens,
        max_samples=args.final_dev_samples,
        desc="Dev Scorecard"
    )
    dev_preds_path = os.path.join(args.output_dir, "dev_preds.json")
    with open(dev_preds_path, "w", encoding="utf-8") as f:
        json.dump(full_dev_preds, f, indent=2)
    print(f"Saved Dev Predictions to: {dev_preds_path}")

    final_metrics = compute_summarization_metrics(full_dev_preds, dev_gold_truth, device="cuda")

    print("\n" + "-" * 60)
    print(f"OFFICIAL VERIFIED RESULTS ({args.arch.upper()} + {args.adaptation.upper()}):")
    print(f"  Dev BERTScore F1: {final_metrics['bertscore_f1']:.6f}")
    print(f"  Dev ROUGE-1 F1:   {final_metrics['rouge1_f1']:.6f}")
    print(f"  Dev ROUGE-L F1:   {final_metrics['rougel_f1']:.6f}")
    print("-" * 60)

    log_result_to_tracker(
        arch=args.arch,
        adaptation=args.adaptation,
        bertscore=final_metrics["bertscore_f1"],
        rouge1=final_metrics["rouge1_f1"],
        rougel=final_metrics["rougel_f1"],
        zip_path=zip_path
    )

    print(f"\nExperiment Completed Successfully in {time.time() - start_time:.1f}s.")
    print(f"Submission Package Ready: {zip_path}")


if __name__ == "__main__":
    main()
