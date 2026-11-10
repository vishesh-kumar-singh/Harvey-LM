"""
Phase 3 - Train Hierarchical Legal Document BiLSTM-Transformer-CRF on HarveyLM Features.

Trains a Document Sequential Contextualizer on cached HarveyLM embeddings to optimize
Macro-F1 across all 13 rhetorical role classes.
"""
import os
import sys


import json
import time
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))

from phase3.src.model.hierarchical_crf import HierarchicalLegalDocCRF

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache-file", default="code/phase3/data/cache/harveylm_doc_features.pt")
parser.add_argument("--model-dir", default="code/phase3/models/hierarchical_crf")
parser.add_argument("--sub-dir", default="code/phase3/submissions/harveylm/hierarchical_crf")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=4e-4)
args, _ = parser.parse_known_args()

CACHE_FILE  = args.cache_file
MODEL_DIR   = args.model_dir
SUB_DIR     = args.sub_dir
LOG_DIR     = "code/phase3/models/logs"

EPOCHS      = args.epochs
LR          = args.lr
WEIGHT_DECAY = 0.01
BATCH_SIZE  = 4
SEED        = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class DocDataset(Dataset):
    def __init__(self, doc_list):
        self.docs = doc_list

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


def collate_docs(batch):
    max_len = max(d["n_sents"] for d in batch)
    B = len(batch)
    D = batch[0]["embeddings"].shape[-1]

    padded_embeddings = torch.zeros(B, max_len, D, dtype=torch.float32)
    padded_positions  = torch.zeros(B, max_len, 1, dtype=torch.float32)
    padded_labels     = torch.full((B, max_len), -1, dtype=torch.long)
    mask              = torch.zeros(B, max_len, dtype=torch.bool)
    doc_ids           = [d["id"] for d in batch]
    n_sents_list      = [d["n_sents"] for d in batch]

    for b, d in enumerate(batch):
        n = d["n_sents"]
        padded_embeddings[b, :n] = d["embeddings"]
        padded_positions[b, :n]  = d["positions"]
        padded_labels[b, :n]     = d["labels"]
        mask[b, :n]              = True

    return {
        "embeddings": padded_embeddings,
        "positions":  padded_positions,
        "labels":     padded_labels,
        "mask":       mask,
        "doc_ids":    doc_ids,
        "n_sents":    n_sents_list
    }


def evaluate(model, loader, device, id2label):
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            pos = batch["positions"].to(device)
            msk = batch["mask"].to(device)
            lbl = batch["labels"]
            n_sents = batch["n_sents"]

            decoded_paths, _ = model.decode(emb, pos, mask=msk)

            for b in range(len(decoded_paths)):
                path = decoded_paths[b]
                gold = lbl[b, :n_sents[b]].tolist()
                all_preds.extend(path)
                all_golds.extend(gold)

    labels_all = list(range(len(id2label)))
    macro_f1 = f1_score(all_golds, all_preds, labels=labels_all, average="macro", zero_division=0)
    per_class = f1_score(all_golds, all_preds, labels=labels_all, average=None, zero_division=0)
    return macro_f1, per_class, all_preds, all_golds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Training Hierarchical Document BiLSTM-Transformer-CRF on HarveyLM")
    print(f"Device: {device}")
    print("=" * 70)

    print(f"Loading cached features from: {CACHE_FILE}")
    cached = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
    train_docs = cached["train"]
    dev_docs   = cached["dev"]
    test_docs  = cached["test"]
    label2id   = cached["label2id"]
    id2label   = cached["id2label"]
    num_labels = len(label2id)

    print(f"Train docs: {len(train_docs)} | Dev docs: {len(dev_docs)} | Test docs: {len(test_docs)}")

    # Compute balanced class weights for Focal Loss
    from collections import Counter
    counts = Counter()
    for d in train_docs:
        counts.update(d["labels"].tolist())

    total = sum(counts.values())
    class_weights = torch.zeros(num_labels)
    for c in range(num_labels):
        cnt = max(counts[c], 1)
        w = (total / (num_labels * cnt)) ** 0.60
        class_weights[c] = min(w, 15.0)
    class_weights = class_weights / class_weights.mean()

    print("\nClass weights (exponent=0.60):")
    for c in range(num_labels):
        print(f"  {id2label[c]:24s} (cnt={counts[c]:5d}): weight = {class_weights[c].item():.4f}")

    train_loader = DataLoader(DocDataset(train_docs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_docs)
    dev_loader   = DataLoader(DocDataset(dev_docs),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_docs)
    test_loader  = DataLoader(DocDataset(test_docs),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_docs)

    feat_dim = train_docs[0]["embeddings"].shape[-1]
    model = HierarchicalLegalDocCRF(
        embed_dim=feat_dim,
        pos_dim=128,
        hidden_dim=512,
        num_labels=num_labels,
        class_weights=class_weights,
        focal_gamma=2.0,
        focal_weight=0.6
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHierarchical Document Model Trainable Parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_macro_f1 = 0.0
    best_epoch = 0

    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            pos = batch["positions"].to(device)
            lbl = batch["labels"].to(device)
            msk = batch["mask"].to(device)

            out = model(emb, pos, mask=msk, labels=lbl)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0

        val_f1, per_class, _, _ = evaluate(model, dev_loader, device, id2label)

        if epoch % 5 == 0 or val_f1 > best_macro_f1:
            lr_curr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:2d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Dev Macro-F1: {val_f1:.4f} | LR: {lr_curr:.2e}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            save_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> NEW BEST MODEL! (Dev Macro-F1: {best_macro_f1:.4f} at Epoch {best_epoch})")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Dev Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")
    print("=" * 70)

    # Load best model and evaluate in full detail
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_f1, per_class, val_preds, val_golds = evaluate(model, dev_loader, device, id2label)

    print("\nBest Model Classification Report on Dev:")
    target_names = [id2label[i] for i in range(num_labels)]
    print(classification_report(val_golds, val_preds, labels=list(range(num_labels)), target_names=target_names, digits=4, zero_division=0))

    # Generate Dev Predictions and Run Official Evaluation
    print("\nGenerating Dev Predictions for Official Evaluation...")
    model.eval()
    dev_predictions = {}
    with torch.no_grad():
        for batch in dev_loader:
            emb = batch["embeddings"].to(device)
            pos = batch["positions"].to(device)
            msk = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]
            n_sents = batch["n_sents"]

            decoded_paths, _ = model.decode(emb, pos, mask=msk)
            for b in range(len(decoded_paths)):
                dev_predictions[doc_ids[b]] = [id2label[p] for p in decoded_paths[b]]

    dev_pred_file = os.path.join(MODEL_DIR, "dev_prediction.json")
    with open(dev_pred_file, "w", encoding="utf-8") as f:
        json.dump(dev_predictions, f, indent=2)
    print(f"Saved: {dev_pred_file}")

    print("\n" + "=" * 70)
    print("OFFICIAL PHASE 3 EVALUATION ON DEV SET:")
    print("=" * 70)
    import subprocess
    eval_script = "code/Evaluation Script/evaluate_phase3.py"
    subprocess.run([
        sys.executable, eval_script,
        "--truth-cl", "code/phase3/data/eval_truth/cl_dev_truth.json",
        "--truth-it", "code/phase3/data/eval_truth/it_dev_truth.json",
        "--prediction", dev_pred_file,
        "--labels", "code/phase3/data/eval_truth/labels.json"
    ])
    print("=" * 70)

    # Generate Test Submission
    print("\nGenerating Test Predictions from Best Hierarchical Model...")
    test_predictions = {}
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embeddings"].to(device)
            pos = batch["positions"].to(device)
            msk = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]
            n_sents = batch["n_sents"]

            decoded_paths, _ = model.decode(emb, pos, mask=msk)
            for b in range(len(decoded_paths)):
                doc_id = doc_ids[b]
                pred_labels = [id2label[p] for p in decoded_paths[b]]
                assert len(pred_labels) == n_sents[b]
                test_predictions[doc_id] = pred_labels

    pred_file = os.path.join(SUB_DIR, "prediction.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2)
    print(f"Saved: {pred_file}")

    zip_file = os.path.join(SUB_DIR, "submission.zip")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, "prediction.json")
    print(f"Saved: {zip_file}")
    print("Done!")


if __name__ == "__main__":
    main()
