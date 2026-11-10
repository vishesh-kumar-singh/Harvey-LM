"""
Phase 3 - Unified Adaptation and Knowledge Distillation Training Script.

Supports:
- Architectures: BaselineLM, HarveyLM, HarveyMoE, LegalEncoderDecoder
- Modes:
  1. lora: Low-Rank Adaptation (A, B matrices)
  2. lorma: Truncated SVD Low-Rank Matrix Approximation
  3. qlora: 4-bit NormalFloat (NF4) Quantized Low-Rank Adaptation
  4. kd: Knowledge Distillation from the 0.4785 LB Grand Consensus teacher

Evaluation Standard: 13-Class Macro-F1 across CL and IT legal domains.
"""
import os
import sys


import argparse
import json
import time
import zipfile
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code/phase1")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))
from phase3.src.data.rr_dataset import load_label_vocab, load_rr_data
from phase3.src.data.rr_context_dataset_v3 import ChronoContextRRDataset
from phase3.src.model.rr_universal_classifier import UniversalLegalClassifier
from phase3.src.model.qlora_from_scratch import apply_qlora_to_model, mark_only_qlora_as_trainable
from phase2.src.model.lora_from_scratch import apply_lora_to_model, mark_only_lora_as_trainable
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase3.src.evaluation.viterbi_decoder import LegalViterbiDecoder

ARCH_CKPTS = {
    "baseline": "code/phase1/models/final_architectures/1_Baseline/epoch_5",
    "harveylm": "code/phase1/models/final_architectures/2_HarveyLM/epoch_99",
    "moe":      "code/phase1/models/final_architectures/3_HarveyMoE/epoch_10",
    "enc_dec":  "code/phase1/models/final_architectures/4_EncoderDecoder/epoch_10"
}

ARCH_TARGETS = {
    "baseline": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "harveylm": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "moe":      ["wq", "wk", "wv", "wo"],
    "enc_dec":  ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
}

DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"
TEACHER_CACHE = "code/phase3/data/cache/teacher_soft_targets.pt"


def evaluate_doc_viterbi(model, docs, dataset, viterbi_decoder, device, id2label, num_labels):
    model.eval()
    doc_examples = defaultdict(list)
    for ex in dataset.examples:
        doc_examples[ex["doc_id"]].append(ex)

    all_raw_preds = []
    all_vit_preds = []
    all_golds = []

    with torch.no_grad():
        for doc in docs:
            doc_id = doc["id"]
            exs = doc_examples.get(doc_id, [])
            if not exs:
                continue

            CHUNK_SIZE = 32
            all_doc_logits = []
            for k in range(0, len(exs), CHUNK_SIZE):
                chunk = exs[k: k + CHUNK_SIZE]
                input_ids   = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask   = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos     = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id   = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        target_mask=target_mask,
                        rel_pos=rel_pos,
                        domain_id=domain_id
                    )
                all_doc_logits.append(out["logits"].float().cpu())

            doc_logits = torch.cat(all_doc_logits, dim=0)
            doc_probs = torch.softmax(doc_logits, dim=-1)
            raw_doc_preds = torch.argmax(doc_logits, dim=-1).tolist()

            vit_doc_preds = viterbi_decoder.decode_document(doc_logits) if viterbi_decoder else raw_doc_preds

            gold_labels = [e["label"].item() for e in exs if e["label"].item() >= 0]
            if len(gold_labels) == len(raw_doc_preds):
                all_raw_preds.extend(raw_doc_preds)
                all_vit_preds.extend(vit_doc_preds)
                all_golds.extend(gold_labels)

    if all_golds:
        raw_macro_f1 = f1_score(all_golds, all_raw_preds, average="macro", zero_division=0)
        vit_macro_f1 = f1_score(all_golds, all_vit_preds, average="macro", zero_division=0)
        vit_acc = (torch.tensor(all_vit_preds) == torch.tensor(all_golds)).float().mean().item()
        return raw_macro_f1, vit_macro_f1, vit_acc, all_vit_preds, all_golds
    return 0.0, 0.0, 0.0, [], []


def generate_test_predictions(model, test_docs, test_dataset, viterbi_decoder, device, id2label, out_path, zip_path):
    model.eval()
    doc_examples = defaultdict(list)
    for ex in test_dataset.examples:
        doc_examples[ex["doc_id"]].append(ex)

    predictions = {}
    total_sents = 0

    with torch.no_grad():
        for doc in test_docs:
            doc_id = doc["id"]
            exs = doc_examples.get(doc_id, [])
            if not exs:
                continue

            CHUNK_SIZE = 32
            all_doc_logits = []
            for k in range(0, len(exs), CHUNK_SIZE):
                chunk = exs[k: k + CHUNK_SIZE]
                input_ids   = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask   = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos     = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id   = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        target_mask=target_mask,
                        rel_pos=rel_pos,
                        domain_id=domain_id
                    )
                all_doc_logits.append(out["logits"].float().cpu())

            doc_logits = torch.cat(all_doc_logits, dim=0)
            doc_probs = torch.softmax(doc_logits, dim=-1)
            raw_preds = torch.argmax(doc_logits, dim=-1).tolist()
            vit_preds = viterbi_decoder.decode_document(doc_logits) if viterbi_decoder else raw_preds

            pred_label_names = [id2label[p] for p in vit_preds]
            predictions[doc_id] = pred_label_names
            total_sents += len(pred_label_names)

    assert total_sents == 2197, f"Expected 2197 test sentences, got {total_sents}!"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname="prediction.json")

    print(f"Successfully generated verified test submission: {zip_path} ({total_sents} sentences).", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, choices=["baseline", "harveylm", "moe", "enc_dec"])
    parser.add_argument("--mode", type=str, required=True, choices=["lora", "lorma", "qlora", "kd"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sub_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"Phase 3 - Running {args.arch.upper()} [{args.mode.upper()}] on {device}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch Size: {args.batch_size} | Rank: {args.r}")
    print("=" * 75, flush=True)

    # 1. Load Tokenizer & Vocab
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_DIR,
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )
    label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
    label2id, id2label = load_label_vocab(label_vocab_path)
    num_labels = len(label2id)

    # 2. Load Data
    train_docs = (
        load_rr_data(os.path.join(DATA_DIR, "Train", "CL_train.json")) +
        load_rr_data(os.path.join(DATA_DIR, "Train", "IT_train.json"))
    )
    dev_docs = (
        load_rr_data(os.path.join(DATA_DIR, "Train", "CL_dev.json")) +
        load_rr_data(os.path.join(DATA_DIR, "Train", "IT_dev.json"))
    )
    test_docs = (
        load_rr_data(os.path.join(DATA_DIR, "Test", "CL_test.json")) +
        load_rr_data(os.path.join(DATA_DIR, "Test", "IT_test.json"))
    )

    train_dataset = ChronoContextRRDataset(train_docs, tokenizer, label2id, max_length=256)
    dev_dataset   = ChronoContextRRDataset(dev_docs,   tokenizer, label2id, max_length=256)
    test_dataset  = ChronoContextRRDataset(test_docs,  tokenizer, label2id, max_length=256)

    # If KD mode, attach teacher soft targets to train examples
    teacher_targets_map = {}
    if args.mode == "kd":
        print(f"Loading Teacher Soft Targets from {TEACHER_CACHE}...")
        teacher_targets_map = torch.load(TEACHER_CACHE, map_location="cpu")
        print(f"Loaded soft targets for {len(teacher_targets_map)} documents.")

        # Assign per-sentence teacher vector to each example
        doc_sent_idx = defaultdict(int)
        for ex in train_dataset.examples:
            did = ex["doc_id"]
            idx = doc_sent_idx[did]
            doc_sent_idx[did] += 1
            if did in teacher_targets_map and idx < teacher_targets_map[did].shape[0]:
                ex["teacher_target"] = teacher_targets_map[did][idx]
            else:
                one_hot = torch.zeros(num_labels)
                one_hot[ex["label"].item()] = 1.0
                ex["teacher_target"] = one_hot

    # 3. Class Weights for Loss
    counts = Counter(ex["label"].item() for ex in train_dataset.examples)
    total_samples = len(train_dataset)
    class_weights = torch.zeros(num_labels)
    for c in range(num_labels):
        cnt = max(counts[c], 1)
        w = (total_samples / (num_labels * cnt)) ** 0.60
        class_weights[c] = min(w, 15.0)
    class_weights = class_weights / class_weights.mean()

    train_files = [os.path.join(DATA_DIR, "Train", f) for f in ["CL_train.json", "IT_train.json"]]
    viterbi_decoder = LegalViterbiDecoder(train_files, label2id, lambda_trans=0.35)

    # 4. Initialize Universal Legal Classifier
    ckpt_path = ARCH_CKPTS[args.arch]
    print(f"\nInstantiating UniversalLegalClassifier for {args.arch} from {ckpt_path}...")
    model = UniversalLegalClassifier(
        arch_type=args.arch,
        ckpt_dir=ckpt_path,
        num_labels=num_labels,
        pos_dim=128,
        domain_dim=32,
        hidden_dropout=0.15,
        class_weights=class_weights,
        gamma=2.0
    )

    # 5. Inject Adaptation
    target_modules = ARCH_TARGETS[args.arch]
    if args.mode == "lora" or args.mode == "kd":
        print(f"Applying custom LoRA (r={args.r}, alpha={args.alpha}) to {target_modules}...")
        model.backbone = apply_lora_to_model(
            model.backbone, target_modules=target_modules,
            r=args.r, lora_alpha=args.alpha, lora_dropout=0.10
        )
        mark_only_lora_as_trainable(model, ["pos_enc", "domain_enc", "pre_classifier", "classifier"])
    elif args.mode == "lorma":
        print(f"Applying custom LORMA (r={args.r}, alpha={args.alpha}) to {target_modules}...")
        model.backbone = apply_lorma_to_model(
            model.backbone, target_modules=target_modules,
            r=args.r, lorma_alpha=args.alpha, dropout=0.10
        )
        mark_only_lorma_as_trainable(model, ["pos_enc", "domain_enc", "pre_classifier", "classifier"])
    elif args.mode == "qlora":
        print(f"Applying custom QLoRA (NF4 4-bit, r={args.r}, alpha={args.alpha}) to {target_modules}...")
        model.backbone = apply_qlora_to_model(
            model.backbone, target_modules=target_modules,
            r=args.r, lora_alpha=args.alpha, lora_dropout=0.10
        )
        mark_only_qlora_as_trainable(model, ["pos_enc", "domain_enc", "pre_classifier", "classifier"])

    model.to(device).bfloat16()

    # 6. Dataloader & Optimizer
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_macro_f1 = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            rel_pos     = batch["rel_pos"].to(device)
            domain_id   = batch["domain_id"].to(device)
            labels      = batch["label"].to(device)
            teacher_tgt = batch["teacher_target"].to(device) if "teacher_target" in batch else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    target_mask=target_mask,
                    rel_pos=rel_pos,
                    domain_id=domain_id,
                    labels=labels,
                    teacher_targets=teacher_tgt
                )
                loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time

        # Dev Evaluation
        raw_f1, vit_f1, vit_acc, _, _ = evaluate_doc_viterbi(
            model, dev_docs, dev_dataset, viterbi_decoder, device, id2label, num_labels
        )
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Dev Raw-F1: {raw_f1:.4f} | Dev Vit-F1: {vit_f1:.4f} | Dev Acc: {vit_acc*100:.2f}% | Time: {elapsed:.1f}s", flush=True)

        if vit_f1 > best_macro_f1:
            best_macro_f1 = vit_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint (Macro-F1: {best_macro_f1:.4f})")

    # If all epochs complete and no checkpoint saved (or equal), save final
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    # 7. Final Test Submission Generation
    print(f"\nLoading best checkpoint from epoch {best_epoch} (Macro-F1: {best_macro_f1:.4f}) for test submission...")
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.to(device).bfloat16()

    pred_out_path = os.path.join(args.sub_dir, "prediction.json")
    zip_out_path  = os.path.join(args.sub_dir, "submission.zip")
    generate_test_predictions(
        model, test_docs, test_dataset, viterbi_decoder, device, id2label, pred_out_path, zip_out_path
    )

    print("=" * 75)
    print(f"Completed {args.arch.upper()} [{args.mode.upper()}] | Best Dev Macro-F1: {best_macro_f1:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"Submission saved to: {zip_out_path}")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
