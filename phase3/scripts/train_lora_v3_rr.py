"""
Phase 3 - High-Performance Chronological Contextual LoRA (V3) on HarveyLM.

Innovations:
1. Chronological Natural Context: [PREV] + [EOS] + [TARGET] + [EOS] + [NEXT]
2. Target-Span Isolation with Symmetric ALiBi (Fixed distance penalty)
3. Document Depth (Fourier MLP) + Domain Embedding (CL vs IT)
4. Class-Balanced Focal Loss (gamma=2.0, weights exponent=0.60)
5. Legal Viterbi Sequence Decoding for document coherence
6. Automatic Feature Caching for Hierarchical CRF downstream

"""
import os
import sys


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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))

from phase3.src.data.rr_dataset import load_label_vocab, load_rr_data
from phase3.src.data.rr_context_dataset_v3 import ChronoContextRRDataset
from phase3.src.model.rr_harveylm_v3 import HarveyForRhetoricalRoleV3
from phase3.src.evaluation.viterbi_decoder import LegalViterbiDecoder
from phase2.src.model.lora_from_scratch import apply_lora_to_model

# ── Hyperparameters ───────────────────────────────────────────────────────────
HARVEY_CKPT   = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
OUT_DIR       = "code/phase3/models/lora_v2_harveylm"
SUB_DIR       = "code/phase3/submissions/harveylm/lora_v2"
LOG_DIR       = "code/phase3/models/logs"
CACHE_OUT     = "code/phase3/data/cache/harveylm_context_features.pt"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"

MAX_LEN       = 256
BATCH_SIZE    = 32
EPOCHS        = 8
LR            = 2.5e-4
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
LORA_R        = 64
LORA_ALPHA    = 128
LORA_DROPOUT  = 0.10
POS_DIM       = 128
DOM_DIM       = 32
SEED          = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("code/phase3/data/cache", exist_ok=True)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate_doc_viterbi(model, docs, dataset, viterbi_decoder, device, id2label, num_labels):
    model.eval()
    # Group dataset examples by doc_id
    doc_examples = defaultdict(list)
    for ex in dataset.examples:
        doc_examples[ex["doc_id"]].append(ex)

    all_raw_preds = []
    all_vit_preds = []
    all_golds = []

    with torch.no_grad():
        for doc in docs:
            doc_id = doc["id"]
            exs = doc_examples[doc_id]
            if not exs:
                continue

            # Chunk evaluate sentences to prevent OOM on long documents
            CHUNK_SIZE = 32
            all_doc_logits = []
            for k in range(0, len(exs), CHUNK_SIZE):
                chunk = exs[k: k + CHUNK_SIZE]
                input_ids = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        target_mask=target_mask,
                        rel_pos=rel_pos,
                        domain_id=domain_id
                    )
                all_doc_logits.append(out["logits"].float().cpu())

            logits = torch.cat(all_doc_logits, dim=0).numpy()
            raw_p = logits.argmax(axis=-1).tolist()
            vit_p = viterbi_decoder.decode_document(logits)

            golds = [e["label"].item() for e in exs]

            all_raw_preds.extend(raw_p)
            all_vit_preds.extend(vit_p)
            all_golds.extend(golds)

    labels_all = list(range(num_labels))
    raw_f1 = f1_score(all_golds, all_raw_preds, labels=labels_all, average="macro", zero_division=0)
    vit_f1 = f1_score(all_golds, all_vit_preds, labels=labels_all, average="macro", zero_division=0)
    per_class_vit = f1_score(all_golds, all_vit_preds, labels=labels_all, average=None, zero_division=0)

    return raw_f1, vit_f1, per_class_vit, all_vit_preds, all_golds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Phase 3 - Chronological Contextual LoRA V3 on HarveyLM")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_DIR,
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )

    label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
    label2id, id2label = load_label_vocab(label_vocab_path)
    num_labels = len(label2id)

    train_docs = []
    for fname in ["CL_train.json", "IT_train.json"]:
        docs = load_rr_data(os.path.join(DATA_DIR, "Train", fname))
        train_docs.extend(docs)
        print(f"  Loaded {len(docs)} documents from {fname}")

    dev_docs = []
    for fname in ["CL_dev.json", "IT_dev.json"]:
        docs = load_rr_data(os.path.join(DATA_DIR, "Train", fname))
        dev_docs.extend(docs)
        print(f"  Loaded {len(docs)} documents from {fname}")

    print("\nBuilding ChronoContextRRDataset for training...")
    t0 = time.time()
    train_dataset = ChronoContextRRDataset(train_docs, tokenizer, label2id, max_length=MAX_LEN)
    print(f"Train sentences: {len(train_dataset):,} (tokenized in {time.time()-t0:.2f}s)")

    print("Building ChronoContextRRDataset for dev...")
    t0 = time.time()
    dev_dataset = ChronoContextRRDataset(dev_docs, tokenizer, label2id, max_length=MAX_LEN)
    print(f"Dev sentences:   {len(dev_dataset):,} (tokenized in {time.time()-t0:.2f}s)")

    # Balanced class weights (exponent 0.60)
    counts = Counter()
    for ex in train_dataset.examples:
        lbl = ex["label"].item()
        if lbl >= 0:
            counts[lbl] += 1
    total_sents = sum(counts.values())
    class_weights = torch.zeros(num_labels)
    for c in range(num_labels):
        cnt = max(counts[c], 1)
        w = (total_sents / (num_labels * cnt)) ** 0.60
        class_weights[c] = min(w, 15.0)
    class_weights = class_weights / class_weights.mean()

    print("\nClass weights (exponent=0.60):")
    for c in range(num_labels):
        print(f"  {id2label[c]:24s} (cnt={counts[c]:5d}): weight = {class_weights[c].item():.4f}")

    train_files = [os.path.join(DATA_DIR, "Train", f) for f in ["CL_train.json", "IT_train.json"]]
    viterbi_decoder = LegalViterbiDecoder(train_files, label2id, lambda_trans=0.35)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\nLoading HarveyLM from: {HARVEY_CKPT}")
    model = HarveyForRhetoricalRoleV3.from_pretrained_harvey(
        HARVEY_CKPT, num_labels=num_labels, pos_dim=POS_DIM, domain_dim=DOM_DIM,
        hidden_dropout=0.15, class_weights=class_weights, gamma=2.0
    )

    # Invert LoRA into attention and FFN
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    model = apply_lora_to_model(
        model, target_modules=target_modules,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT
    )

    # Unfreeze LoRA adapters + Position Encoder + Domain Encoder + PreClassifier + Classifier
    for name, param in model.named_parameters():
        if any(x in name for x in ["lora_A", "lora_B", "pos_enc", "domain_enc", "pre_classifier", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model.to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_macro_f1 = 0.0
    best_epoch = 0

    print(f"Total training steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")
    print("\nStarting LoRA V3 training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids   = batch["input_ids"].to(device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            rel_pos     = batch["rel_pos"].to(device, non_blocking=True)
            domain_id   = batch["domain_id"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids, attention_mask=attn_mask,
                    target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id,
                    labels=labels
                )
                loss = outputs["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        elapsed = time.time() - start_time
        avg_train_loss = epoch_loss / len(train_loader)

        # Evaluate on dev
        raw_f1, vit_f1, per_class, val_preds, val_golds = evaluate_doc_viterbi(
            model, dev_docs, dev_dataset, viterbi_decoder, device, id2label, num_labels
        )

        lr_curr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {avg_train_loss:.4f} | "
              f"Dev Raw-F1: {raw_f1:.4f} | Dev Viterbi-F1: {vit_f1:.4f} | LR: {lr_curr:.2e}")

        # Track per class
        for c in range(num_labels):
            if c in [5, 8]:  # Dissent, PrecedentOverruled have 0 in dev
                continue
            print(f"   {id2label[c]:24s}: {per_class[c]:.4f}")

        if vit_f1 > best_macro_f1:
            best_macro_f1 = vit_f1
            best_epoch = epoch
            save_path = os.path.join(OUT_DIR, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> NEW BEST MODEL! (Dev Viterbi-F1: {best_macro_f1:.4f} at Epoch {best_epoch}) -> {save_path}")

    print("\n" + "=" * 70)
    print(f"LoRA V3 Training Complete! Best Dev Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})")
    print("=" * 70)

    # Load best model
    best_path = os.path.join(OUT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    raw_f1, vit_f1, per_class, val_preds, val_golds = evaluate_doc_viterbi(
        model, dev_docs, dev_dataset, viterbi_decoder, device, id2label, num_labels
    )
    print("\nFinal Best Model Classification Report on Dev:")
    target_names = [id2label[i] for i in range(num_labels)]
    print(classification_report(val_golds, val_preds, labels=list(range(num_labels)), target_names=target_names, digits=4, zero_division=0))

    # Generate Test Submission
    print("\nGenerating Test Predictions...")
    test_docs = []
    for fname in ["CL_test.json", "IT_test.json"]:
        docs = load_rr_data(os.path.join(DATA_DIR, "Test", fname))
        test_docs.extend(docs)
        print(f"  Loaded {len(docs)} test documents from {fname}")

    test_dataset = ChronoContextRRDataset(test_docs, tokenizer, label2id, max_length=MAX_LEN)
    test_examples = defaultdict(list)
    for ex in test_dataset.examples:
        test_examples[ex["doc_id"]].append(ex)

    test_predictions = {}
    model.eval()
    with torch.no_grad():
        CHUNK_SIZE = 32
        torch.cuda.empty_cache()
        for doc in test_docs:
            doc_id = doc["id"]
            exs = test_examples[doc_id]
            all_doc_logits = []
            for k in range(0, len(exs), CHUNK_SIZE):
                chunk = exs[k: k + CHUNK_SIZE]
                input_ids = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids, attention_mask=attn_mask, target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id)
                all_doc_logits.append(out["logits"].float().cpu())

            logits = torch.cat(all_doc_logits, dim=0).numpy()
            vit_p = viterbi_decoder.decode_document(logits)
            pred_labels = [id2label[p] for p in vit_p]
            assert len(pred_labels) == len(doc["sentences"])
            test_predictions[doc_id] = pred_labels

    pred_file = os.path.join(SUB_DIR, "prediction.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2)
    print(f"Saved: {pred_file}")

    zip_file = os.path.join(SUB_DIR, "submission.zip")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, arcname="prediction.json")
    print(f"Saved: {zip_file}")

    # Now extract and cache contextual sentence representations for Hierarchical CRF!
    print(f"\nCaching fine-tuned contextual sentence embeddings to: {CACHE_OUT}...")
    cached_splits = {}
    for split_name, split_docs, split_ds in [("train", train_docs, train_dataset), ("dev", dev_docs, dev_dataset), ("test", test_docs, test_dataset)]:
        split_exs = defaultdict(list)
        for ex in split_ds.examples:
            split_exs[ex["doc_id"]].append(ex)

        doc_feature_list = []
        for doc in split_docs:
            doc_id = doc["id"]
            exs = split_exs[doc_id]
            n_sents = len(exs)

            all_embs = []
            # Process in sub-batches of 32 sentences
            SUB_B = 32
            torch.cuda.empty_cache()
            for s_idx in range(0, n_sents, SUB_B):
                chunk = exs[s_idx: s_idx + SUB_B]
                input_ids = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target_pooled, _ = model.extract_features(
                        input_ids, attention_mask=attn_mask,
                        target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id
                    )
                all_embs.append(target_pooled.float().cpu())

            doc_embs = torch.cat(all_embs, dim=0)   # (N_sents, 1024)
            pos_tensor = torch.tensor([e["rel_pos"].item() for e in exs], dtype=torch.float32).unsqueeze(-1)
            lbl_tensor = torch.tensor([e["label"].item() for e in exs], dtype=torch.long)

            doc_feature_list.append({
                "id": doc_id,
                "n_sents": n_sents,
                "embeddings": doc_embs,
                "positions": pos_tensor,
                "labels": lbl_tensor
            })
        cached_splits[split_name] = doc_feature_list
        print(f"  Processed {len(doc_feature_list)} {split_name} documents.")

    cached_splits["label2id"] = label2id
    cached_splits["id2label"] = id2label
    torch.save(cached_splits, CACHE_OUT)
    print(f"Features successfully saved to: {CACHE_OUT}")


if __name__ == "__main__":
    main()
