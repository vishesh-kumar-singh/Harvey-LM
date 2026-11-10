#!/usr/bin/env python3
"""
Phase 7: Master Prior Case Retrieval Training & Evaluation Pipeline.
Supports:
- Foundation Backbones: HarveyLM, BaselineLM, HarveyMoE, LegalEncoderDecoder
- Adaptation: LORMA (Compulsory), LoRA, QLoRA, Knowledge Distillation, BM25, Hybrid RRF
- Evaluates on Dev Split (Micro-F1@7) using official evaluation standards
- Generates compliant submission.zip for Hidden Test Split (237 queries)
"""

import os
import sys
import json
import time
import argparse
import subprocess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase7.src.model.pcr_retriever import UniversalLegalRetriever
from phase7.src.data.pcr_dataset import (
    PCRTrainPairDataset,
    PCRDocDataset,
    collate_train_pairs,
    collate_docs
)
from phase7.src.evaluation.eval_utils import (
    compute_pcr_metrics,
    export_pcr_submission
)
from phase7.src.baselines.bm25 import LegalBM25

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
EVAL_DIR = os.path.join(ROOT_CODE, "phase7/data/eval_truth")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
EVAL_SCRIPT_PATH = os.path.join(ROOT_CODE, "Evaluation Script/evaluate_phase7.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def encode_docs(model, doc_dict, tokenizer, max_seq_len, batch_size, device):
    """Encodes a dictionary of documents into an L2-normalized embedding matrix."""
    model.eval()
    dataset = PCRDocDataset(doc_dict, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_docs, num_workers=2)

    doc_ids_all = []
    embeds_list = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.encode(ids, mask)
        embeds_list.append(embeds.float().cpu())
        doc_ids_all.extend(batch["doc_ids"])

    return doc_ids_all, torch.cat(embeds_list, dim=0)


def rank_candidates(query_embeds, query_ids, cand_embeds, cand_ids, top_k=7):
    """Computes cosine similarity matrix and extracts top-K candidate IDs per query."""
    # query_embeds: [N_q, D], cand_embeds: [N_c, D]
    sim = torch.matmul(query_embeds, cand_embeds.T)  # [N_q, N_c]
    top_indices = torch.topk(sim, k=top_k, dim=-1).indices.cpu().numpy()

    pred = {}
    for i, q_id in enumerate(query_ids):
        pred[q_id] = [cand_ids[idx] for idx in top_indices[i]]
    return pred


def run_pipeline(args):
    print("==========================================================================")
    print(f"PHASE 7 PCR PIPELINE: ARCH='{args.arch}', ADAPTATION='{args.adaptation}'")
    print(f"Hardware Device: {args.device} | Context Length: {args.max_seq_len}")
    print(f"Submissions Dir: {args.sub_dir}")
    print("==========================================================================")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Raw Splits
    print("Loading datasets...")
    train_q = load_json(os.path.join(DATA_DIR, "Train/train.json"))
    train_c = load_json(os.path.join(DATA_DIR, "Train/train_candidates.json"))
    dev_q = load_json(os.path.join(DATA_DIR, "Dev/dev.json"))
    dev_c = load_json(os.path.join(DATA_DIR, "Dev/dev_candidates.json"))
    test_q = load_json(os.path.join(DATA_DIR, "Test/test.json"))
    test_c = load_json(os.path.join(DATA_DIR, "Test/test_candidates.json"))

    dev_truth = load_json(os.path.join(EVAL_DIR, "dev_truth.json"))
    dev_cand_ids = load_json(os.path.join(EVAL_DIR, "dev_candidate_ids.json"))
    test_cand_ids = load_json(os.path.join(EVAL_DIR, "test_candidate_ids.json"))

    # 1. SPECIAL CASE: BM25 Lexical Baseline
    if args.adaptation == "bm25" or args.arch == "bm25":
        print("\n--- Running BM25 Lexical Baseline ---")
        # Fit on Dev pool
        bm25_dev = LegalBM25(k1=1.5, b=0.75)
        bm25_dev.fit(dev_c)
        dev_pred = {}
        for q_id, q_val in dev_q.items():
            top7 = bm25_dev.query(q_val["text"], top_k=7)
            dev_pred[q_id] = [c for c, _ in top7]

        # Fit on Test pool
        bm25_test = LegalBM25(k1=1.5, b=0.75)
        bm25_test.fit(test_c)
        test_pred = {}
        for q_id, q_val in test_q.items():
            top7 = bm25_test.query(q_val["text"], top_k=7)
            test_pred[q_id] = [c for c, _ in top7]

    # 2. SPECIAL CASE: Reciprocal Rank Fusion (Hybrid)
    elif args.adaptation == "hybrid":
        print("\n--- Running Reciprocal Rank Fusion (Hybrid Dense + BM25) ---")
        # Build paths for dev and test predictions
        dev_dense_path = os.path.join(os.path.dirname(args.dense_pred_file), "dev_prediction.json")
        dense_dev_pred = load_json(dev_dense_path) if os.path.exists(dev_dense_path) else {}
        dense_test_pred = load_json(args.dense_pred_file)

        # Build BM25 rankings
        bm25_dev = LegalBM25()
        bm25_dev.fit(dev_c)
        bm25_test = LegalBM25()
        bm25_test.fit(test_c)

        def fuse_ranks(dense_p, bm25_engine, queries_dict, cand_pool, top_k=7):
            fused = {}
            w_dense = args.w_dense
            w_bm25 = 1.0 - w_dense
            for q_id, q_val in queries_dict.items():
                d_cands = dense_p.get(q_id, [])
                b_cands = [c for c, _ in bm25_engine.query(q_val["text"], top_k=50)]

                scores = {}
                for rank, c_id in enumerate(d_cands):
                    scores[c_id] = scores.get(c_id, 0.0) + w_dense / (60.0 + rank + 1)
                for rank, c_id in enumerate(b_cands):
                    scores[c_id] = scores.get(c_id, 0.0) + w_bm25 / (60.0 + rank + 1)

                sorted_cands = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
                # Fill to top_k if needed
                for c in cand_pool:
                    if c not in scores:
                        sorted_cands.append(c)
                        if len(sorted_cands) >= top_k:
                            break
                fused[q_id] = sorted_cands[:top_k]
            return fused

        dev_pred = fuse_ranks(dense_dev_pred, bm25_dev, dev_q, dev_cand_ids)
        test_pred = fuse_ranks(dense_test_pred, bm25_test, test_q, test_cand_ids)

    # 3. DENSE NEURAL RETRIEVERS (HarveyLM, BaselineLM, HarveyMoE, LegalEncoderDecoder)
    else:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        print(f"\nInitializing UniversalLegalRetriever ({args.arch}, adaptation={args.adaptation})...")
        model = UniversalLegalRetriever(
            arch_type=args.arch,
            adaptation=args.adaptation,
            r=args.r,
            alpha=args.alpha,
            temperature=args.temperature,
            kd_temperature=args.kd_temperature,
            kd_alpha=args.kd_alpha
        ).to(device)

        # Optional Teacher Model for Knowledge Distillation
        teacher_model = None
        if args.adaptation in ["kd", "distillation"] and args.teacher_ckpt:
            print(f"Loading Teacher Model from {args.teacher_ckpt}...")
            tk_state = torch.load(args.teacher_ckpt, map_location=device)
            tk_args = tk_state.get("args", {})
            t_arch = tk_args.get("arch", "harveylm")
            t_adapt = tk_args.get("adaptation", "lorma")
            t_r = tk_args.get("r", args.r)
            t_alpha = tk_args.get("alpha", args.alpha)
            teacher_model = UniversalLegalRetriever(
                arch_type=t_arch,
                adaptation=t_adapt,
                r=t_r,
                alpha=t_alpha
            ).to(device)
            teacher_model.load_state_dict(tk_state.get("model_state_dict", tk_state), strict=False)
            teacher_model.eval()

        # Training Phase
        if args.epochs > 0:
            print("\nPreparing Training Pairs Dataset...")
            train_ds = PCRTrainPairDataset(train_q, train_c, tokenizer, max_seq_len=args.max_seq_len)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_train_pairs,
                num_workers=2,
                pin_memory=True
            )
            print(f"Total Training Positive Pairs: {len(train_ds):,} | Batches per Epoch: {len(train_loader)}")

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=0.01
            )
            scaler = torch.amp.GradScaler("cuda")

            print("\nBeginning Contrastive InfoNCE Fine-Tuning...")
            model.train()
            step = 0
            for epoch in range(1, args.epochs + 1):
                t_start = time.time()
                epoch_loss = 0.0
                for batch in train_loader:
                    step += 1
                    optimizer.zero_grad()
                    q_ids = batch["q_ids"].to(device)
                    q_mask = batch["q_mask"].to(device)
                    c_ids = batch["c_ids"].to(device)
                    c_mask = batch["c_mask"].to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss, _, _ = model(q_ids, q_mask, c_ids, c_mask, teacher_model=teacher_model)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    if step % 25 == 0 or step == len(train_loader):
                        print(f" Epoch {epoch}/{args.epochs} | Step {step:3d}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

                avg_loss = epoch_loss / len(train_loader)
                elapsed = time.time() - t_start
                print(f"--> Epoch {epoch} Completed in {elapsed:.1f}s | Average Contrastive Loss: {avg_loss:.4f}")

            # Save checkpoint
            model_dir = args.sub_dir.replace("/submissions/", "/models/")
            os.makedirs(model_dir, exist_ok=True)
            ckpt_path = os.path.join(model_dir, "best_model.pt")
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"Saved adapted retriever checkpoint to {ckpt_path}")

        # Evaluation & Encoding Phase
        print("\n--- Encoding Candidate Pools & Query Sets ---")
        t0 = time.time()
        dev_c_ids, dev_c_embeds = encode_docs(model, dev_c, tokenizer, args.max_seq_len, args.eval_batch_size, device)
        dev_q_ids, dev_q_embeds = encode_docs(model, dev_q, tokenizer, args.max_seq_len, args.eval_batch_size, device)
        print(f"Encoded Dev set ({len(dev_c_ids)} cands, {len(dev_q_ids)} queries) in {time.time() - t0:.2f}s")
        dev_pred = rank_candidates(dev_q_embeds, dev_q_ids, dev_c_embeds, dev_c_ids, top_k=7)

        t0 = time.time()
        test_c_ids, test_c_embeds = encode_docs(model, test_c, tokenizer, args.max_seq_len, args.eval_batch_size, device)
        test_q_ids, test_q_embeds = encode_docs(model, test_q, tokenizer, args.max_seq_len, args.eval_batch_size, device)
        print(f"Encoded Test set ({len(test_c_ids)} cands, {len(test_q_ids)} queries) in {time.time() - t0:.2f}s")
        test_pred = rank_candidates(test_q_embeds, test_q_ids, test_c_embeds, test_c_ids, top_k=7)

    # 4. COMPUTE DEV METRICS
    print("\n=======================================================")
    print("DEV SPLIT EVALUATION RESULTS")
    print("=======================================================")
    dev_metrics = compute_pcr_metrics(dev_truth, dev_pred)
    print(f"Micro-F1@7:        {dev_metrics['micro_f1_7']:.6f}")
    print(f"Micro-Precision@7: {dev_metrics['micro_precision_7']:.6f}")
    print(f"Micro-Recall@7:    {dev_metrics['micro_recall_7']:.6f}")
    print(f"TP={dev_metrics['tp']}, FP={dev_metrics['fp']}, FN={dev_metrics['fn']}")

    # Save Dev Predictions for Hybrid Ensembling
    os.makedirs(args.sub_dir, exist_ok=True)
    dev_pred_path = os.path.join(args.sub_dir, "dev_prediction.json")
    with open(dev_pred_path, "w", encoding="utf-8") as f:
        json.dump(dev_pred, f, indent=2)

    # 5. PACKAGE TEST SUBMISSION
    print("\n=======================================================")
    print("PACKAGING OFFICIAL TEST SUBMISSION")
    print("=======================================================")
    sub_zip_path = os.path.join(args.sub_dir, "submission.zip")
    export_pcr_submission(
        pred_dict=test_pred,
        output_zip_path=sub_zip_path,
        allowed_candidates=set(test_cand_ids),
        expected_queries=set(test_q.keys())
    )

    # Verify with official evaluate_phase7.py
    cmd = [
        sys.executable,
        EVAL_SCRIPT_PATH,
        "--truth", os.path.join(EVAL_DIR, "dev_truth.json"),
        "--candidate-ids", os.path.join(EVAL_DIR, "dev_candidate_ids.json"),
        "--prediction", dev_pred_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n[Official Evaluator Output on Dev]:")
    print(res.stdout.strip())
    if res.returncode != 0:
        print("[Evaluator Error]:", res.stderr)

    # 6. LOG TO RESULTS TRACKER
    tracker_path = os.path.join(ROOT_CODE, "phase7/results.txt")
    log_line = f"{args.arch.upper():<16} {args.adaptation.upper():<20} {dev_metrics['micro_f1_7']:<14.6f} {dev_metrics['micro_precision_7']:<11.6f} {dev_metrics['micro_recall_7']:<12.6f} [READY]        {sub_zip_path}\n"
    with open(tracker_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\nLogged results to {tracker_path}")

    return dev_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7 PCR Pipeline")
    parser.add_argument("--arch", type=str, default="harveylm", choices=["harveylm", "baseline", "moe", "enc_dec", "bm25", "hybrid"])
    parser.add_argument("--adaptation", type=str, default="lorma", choices=["lorma", "lora", "qlora", "kd", "partial", "full", "bm25", "hybrid"])
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--dense_pred_file", type=str, default=None)
    parser.add_argument("--w_dense", type=float, default=0.60)
    parser.add_argument("--sub_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
