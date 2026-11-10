"""
Phase 3 - Training HarveyLM V4 with Attentive Token-Level Representation Learning.

Innovations:
1. Multi-Head Self-Attentive Pooling over Target Tokens (4 attention heads)
2. Stopword & Punctuation Attenuation in Attention Logits
3. Max-Over-Time + Mean Token Pooling (Multi-View Aggregation: 3,232 dimensions)
4. Warm-Started from Peak LORMA Backbone Checkpoint (0.4599 LB)
5. Legal Viterbi Sequence Decoding & Feature Caching

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
from phase3.src.model.rr_harveylm_v4 import HarveyForRhetoricalRoleV4
from phase3.src.evaluation.viterbi_decoder import LegalViterbiDecoder
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable

# ── Hyperparameters ───────────────────────────────────────────────────────────
HARVEY_CKPT   = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
PREV_LORMA_PT = "code/phase3/models/lorma_v3_harveylm/best_model.pt"
DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
OUT_DIR       = "code/phase3/models/attentive_lorma_harveylm"
SUB_DIR       = "code/phase3/submissions/harveylm/attentive_lorma"
LOG_DIR       = "code/phase3/models/logs"
CACHE_OUT     = "code/phase3/data/cache/harveylm_attentive_lorma_features.pt"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"

MAX_LEN       = 256
BATCH_SIZE    = 16
ACCUM_STEPS   = 2
EPOCHS        = 6
LR_HEAD       = 3.5e-4
LR_LORMA      = 4e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
LORMA_R       = 64
LORMA_ALPHA   = 128
LORMA_DROPOUT = 0.10
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


def build_stopword_ids(tokenizer):
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "was", "are", "were", "be", "been", "by", "with", "from", "as",
        "that", "this", "it", "its", "which", "who", "whom", "their", "there",
        "has", "have", "had", "will", "would", "shall", "should", "can", "could",
        "may", "might", ",", ".", ";", ":", "-", "(", ")", "[", "]", "\"", "'",
        "``", "''"
    }
    vocab = tokenizer.get_vocab()
    sw_ids = set()
    for word, idx in vocab.items():
        clean = word.replace("Ġ", "").replace(" ", "").lower().strip()
        if clean in stopwords:
            sw_ids.add(idx)
    return sw_ids


def get_stopword_mask(input_ids, sw_ids):
    # input_ids: (B, L)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sw in sw_ids:
        mask |= (input_ids == sw)
    return mask


def evaluate_doc_viterbi(model, docs, dataset, viterbi_decoder, sw_ids, device, id2label, num_labels):
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
            exs = doc_examples[doc_id]
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
                sw_mask     = get_stopword_mask(input_ids, sw_ids).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        target_mask=target_mask,
                        rel_pos=rel_pos,
                        domain_id=domain_id,
                        stopword_mask=sw_mask
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
    raw_macro_f1 = f1_score(all_golds, all_raw_preds, labels=labels_all, average="macro", zero_division=0)
    vit_macro_f1 = f1_score(all_golds, all_vit_preds, labels=labels_all, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_golds, all_vit_preds, labels=labels_all, average=None, zero_division=0)

    return raw_macro_f1, vit_macro_f1, per_class_f1, all_vit_preds, all_golds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print(f"Training Attentive Token Head HarveyLM V4 on {device}", flush=True)
    print(f"Backbone: {HARVEY_CKPT}", flush=True)
    print(f"Rank: {LORMA_R} | Alpha: {LORMA_ALPHA} | Dropout: {LORMA_DROPOUT}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Tokenizer & Vocab
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_DIR,
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )
    sw_ids = build_stopword_ids(tokenizer)
    print(f"Loaded Tokenizer. Detected {len(sw_ids)} stopword/punctuation token IDs.", flush=True)

    label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
    label2id, id2label = load_label_vocab(label_vocab_path)
    num_labels = len(label2id)

    # 2. Load Raw Splits
    train_docs = []
    for fname in ["CL_train.json", "IT_train.json"]:
        train_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Train", fname)))

    dev_docs = []
    for fname in ["CL_dev.json", "IT_dev.json"]:
        dev_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Train", fname)))

    test_docs = []
    for fname in ["CL_test.json", "IT_test.json"]:
        test_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Test", fname)))

    print(f"Documents - Train: {len(train_docs)} | Dev: {len(dev_docs)} | Test: {len(test_docs)}", flush=True)

    # 3. Create Datasets
    print("\nTokenizing and building ChronoContextRRDataset...", flush=True)
    train_dataset = ChronoContextRRDataset(train_docs, tokenizer, label2id, max_length=MAX_LEN)
    dev_dataset   = ChronoContextRRDataset(dev_docs,   tokenizer, label2id, max_length=MAX_LEN)
    test_dataset  = ChronoContextRRDataset(test_docs,  tokenizer, label2id, max_length=MAX_LEN)

    print(f"Sentences - Train: {len(train_dataset):,} | Dev: {len(dev_dataset):,} | Test: {len(test_dataset):,}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # 4. Class Weights for Focal Loss
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

    # 5. Load Pretrained HarveyLM V4 & Inject LORMA
    print(f"\nInstantiating HarveyLM V4 with Attentive Token Pooling Head...", flush=True)
    model = HarveyForRhetoricalRoleV4.from_pretrained_harvey(
        HARVEY_CKPT, num_labels=num_labels, pos_dim=POS_DIM, domain_dim=DOM_DIM,
        num_attn_heads=4, hidden_dropout=0.15, class_weights=class_weights, gamma=2.0
    )

    print("Injecting Low-Rank Matrix Approximation (LORMA)...", flush=True)
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    model = apply_lorma_to_model(
        model, target_modules=target_modules,
        r=LORMA_R, lorma_alpha=LORMA_ALPHA, dropout=LORMA_DROPOUT
    )

    # Warm-start LORMA backbone weights from our peak 0.4599 model
    if os.path.exists(PREV_LORMA_PT):
        print(f"Warm-starting backbone and positional weights from {PREV_LORMA_PT}...", flush=True)
        prev_state = torch.load(PREV_LORMA_PT, map_location="cpu", weights_only=True)
        # Load only backbone/LORMA weights, leaving new Attentive Token Head initialized freshly
        model_state = model.state_dict()
        loaded_keys = 0
        for k, v in prev_state.items():
            if k in model_state and "classifier" not in k and "pre_classifier" not in k:
                if model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_keys += 1
        model.load_state_dict(model_state)
        print(f"Successfully warm-started {loaded_keys} weight tensors from peak LORMA checkpoint!", flush=True)

    mark_only_lorma_as_trainable(
        model, additional_trainable=["pos_enc", "domain_enc", "token_head"]
    )

    model.to(device)

    # 6. Differential Learning Rates: Fast for new Token Head, gentle for LORMA backbone
    head_params = [p for n, p in model.named_parameters() if ("token_head" in n or "pos_enc" in n or "domain_enc" in n) and p.requires_grad]
    lorma_params = [p for n, p in model.named_parameters() if ("lorma" in n) and p.requires_grad]

    optimizer = AdamW([
        {"params": head_params,  "lr": LR_HEAD,  "weight_decay": WEIGHT_DECAY},
        {"params": lorma_params, "lr": LR_LORMA, "weight_decay": WEIGHT_DECAY * 0.1}
    ])

    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_macro_f1 = 0.0
    best_epoch = 0

    print(f"\nTotal optimization steps: {total_steps:,} | Warmup steps: {warmup_steps:,}", flush=True)
    print("Starting training of Attentive Token Head + LORMA V4...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids   = batch["input_ids"].to(device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            rel_pos     = batch["rel_pos"].to(device, non_blocking=True)
            domain_id   = batch["domain_id"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)
            sw_mask     = get_stopword_mask(input_ids, sw_ids).to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids, attention_mask=attn_mask,
                    target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id,
                    stopword_mask=sw_mask, labels=labels
                )
                loss = outputs["loss"] / ACCUM_STEPS

            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % 200 == 0:
                print(f"  Epoch {epoch:2d} | Step {step+1:4d}/{len(train_loader)} | Loss: {loss.item()*ACCUM_STEPS:.4f}", flush=True)

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # Evaluate on Dev with Viterbi Decoding
        raw_f1, vit_f1, per_class, val_preds, val_golds = evaluate_doc_viterbi(
            model, dev_docs, dev_dataset, viterbi_decoder, sw_ids, device, id2label, num_labels
        )

        lr_head_curr = scheduler.get_last_lr()[0]
        print(f"\n>>> Epoch {epoch:2d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {avg_loss:.4f} | "
              f"Dev Raw-F1: {raw_f1:.4f} | Dev Viterbi-F1: {vit_f1:.4f} | Head LR: {lr_head_curr:.2e}", flush=True)

        if vit_f1 > best_macro_f1:
            best_macro_f1 = vit_f1
            best_epoch = epoch
            save_path = os.path.join(OUT_DIR, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> NEW BEST MODEL! (Dev Viterbi Macro-F1: {best_macro_f1:.4f} at Epoch {best_epoch})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Training Complete! Best Dev Viterbi Macro-F1: {best_macro_f1:.4f} (Epoch {best_epoch})", flush=True)
    print("=" * 70, flush=True)

    # 7. Load Best Checkpoint & Full Dev Evaluation
    best_path = os.path.join(OUT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    raw_f1, vit_f1, per_class, val_preds, val_golds = evaluate_doc_viterbi(
        model, dev_docs, dev_dataset, viterbi_decoder, sw_ids, device, id2label, num_labels
    )

    print("\nDetailed Dev Classification Report (Attentive Token Head V4):", flush=True)
    target_names = [id2label[i] for i in range(num_labels)]
    print(classification_report(val_golds, val_preds, labels=list(range(num_labels)), target_names=target_names, digits=4, zero_division=0), flush=True)

    # 8. Generate Dev Predictions & Run Official Evaluation
    print("Generating Dev Predictions for Official Evaluation...", flush=True)
    dev_predictions = {}
    doc_dev_examples = defaultdict(list)
    for ex in dev_dataset.examples:
        doc_dev_examples[ex["doc_id"]].append(ex)

    with torch.no_grad():
        for doc in dev_docs:
            doc_id = doc["id"]
            exs = doc_dev_examples[doc_id]
            all_doc_logits = []
            for k in range(0, len(exs), 32):
                chunk = exs[k: k + 32]
                input_ids   = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask   = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos     = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id   = torch.stack([e["domain_id"] for e in chunk]).to(device)
                sw_mask     = get_stopword_mask(input_ids, sw_ids).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids, attention_mask=attn_mask, target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id, stopword_mask=sw_mask)
                all_doc_logits.append(out["logits"].float().cpu())

            logits = torch.cat(all_doc_logits, dim=0).numpy()
            vit_p = viterbi_decoder.decode_document(logits)
            dev_predictions[doc_id] = [id2label[p] for p in vit_p]

    dev_pred_file = os.path.join(OUT_DIR, "dev_prediction.json")
    with open(dev_pred_file, "w", encoding="utf-8") as f:
        json.dump(dev_predictions, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("OFFICIAL PHASE 3 EVALUATION ON DEV SET:", flush=True)
    print("=" * 70, flush=True)
    import subprocess
    eval_script = "code/Evaluation Script/evaluate_phase3.py"
    subprocess.run([
        sys.executable, eval_script,
        "--truth-cl", "code/phase3/data/eval_truth/cl_dev_truth.json",
        "--truth-it", "code/phase3/data/eval_truth/it_dev_truth.json",
        "--prediction", dev_pred_file,
        "--labels", "code/phase3/data/eval_truth/labels.json"
    ])
    print("=" * 70, flush=True)

    # 9. Generate Test Submission
    print("\nGenerating Official Test Submission...", flush=True)
    test_predictions = {}
    doc_test_examples = defaultdict(list)
    for ex in test_dataset.examples:
        doc_test_examples[ex["doc_id"]].append(ex)

    total_test_sents = 0
    with torch.no_grad():
        for doc in test_docs:
            doc_id = doc["id"]
            exs = doc_test_examples[doc_id]
            total_test_sents += len(exs)

            all_doc_logits = []
            for k in range(0, len(exs), 32):
                chunk = exs[k: k + 32]
                input_ids   = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask   = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                rel_pos     = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                domain_id   = torch.stack([e["domain_id"] for e in chunk]).to(device)
                sw_mask     = get_stopword_mask(input_ids, sw_ids).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids, attention_mask=attn_mask, target_mask=target_mask, rel_pos=rel_pos, domain_id=domain_id, stopword_mask=sw_mask)
                all_doc_logits.append(out["logits"].float().cpu())

            logits = torch.cat(all_doc_logits, dim=0).numpy()
            vit_p = viterbi_decoder.decode_document(logits)
            test_predictions[doc_id] = [id2label[p] for p in vit_p]

    pred_file = os.path.join(SUB_DIR, "prediction.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2)

    zip_file = os.path.join(SUB_DIR, "submission.zip")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, "prediction.json")
    print(f"Saved: {zip_file} (total sentences: {total_test_sents})", flush=True)

    # 10. Extract 3232-d Features for Downstream Sequence Modeling
    print("\nExtracting 3232-d Attentive Token Features for all splits...", flush=True)
    splits_dict = {"train": train_docs, "dev": dev_docs, "test": test_docs}
    datasets_dict = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    cached_splits = {}

    for split_name, doc_list in splits_dict.items():
        ds = datasets_dict[split_name]
        d_exs = defaultdict(list)
        for ex in ds.examples:
            d_exs[ex["doc_id"]].append(ex)

        doc_feature_list = []
        with torch.no_grad():
            for doc in doc_list:
                doc_id = doc["id"]
                exs = d_exs[doc_id]
                n_sents = len(exs)

                all_embs = []
                for k in range(0, n_sents, 32):
                    chunk = exs[k: k + 32]
                    input_ids   = torch.stack([e["input_ids"] for e in chunk]).to(device)
                    attn_mask   = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                    target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
                    rel_pos     = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                    domain_id   = torch.stack([e["domain_id"] for e in chunk]).to(device)
                    sw_mask     = get_stopword_mask(input_ids, sw_ids).to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, fused_rep, _ = model.extract_token_features(
                            input_ids, attention_mask=attn_mask,
                            target_mask=target_mask, rel_pos=rel_pos,
                            domain_id=domain_id, stopword_mask=sw_mask
                        )
                    all_embs.append(fused_rep.float().cpu())

                doc_embs = torch.cat(all_embs, dim=0)
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
        print(f"  Processed {len(doc_feature_list)} {split_name} documents.", flush=True)

    cached_splits["label2id"] = label2id
    cached_splits["id2label"] = id2label
    torch.save(cached_splits, CACHE_OUT)
    print(f"Saved Attentive Token features to: {CACHE_OUT} ({os.path.getsize(CACHE_OUT)/1e6:.1f} MB)", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
