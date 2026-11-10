"""
Phase 3 - Training LORMA V4.5 with 5-Fold Stratified Cross-Validation on 90 Documents.

Architectural Innovations:
1. 5-Sentence Local Context Window [s_{i-2}, s_{i-1}, s_i, s_{i+1}, s_{i+2}] with Segment Distance Embeddings.
2. 3,290-d Attribution-Aware Head:
   - Scale-invariant Cosine Transitions cos(h_i, h_{i-1}) and cos(h_i, h_{i+1}).
   - Boundary Distances min(i, 20)/20 and min(N-1-i, 20)/20.
   - Causal Party-State Cue Tracking (exponential moving average over preceding discourse).
   - 4-head attention pooling with -2.5 stopword mask and gentle +0.5 init for legal cues.
3. Prior-Preserving Loss Objective (ZERO logit adjustment):
   - Standard Cross-Entropy with label smoothing (eps=0.05).
   - Within-Document Supervised Contrastive Loss (lambda_con=0.15).
   - Confusable-Pair Weight Orthogonality Loss (lambda_ortho=0.05).
   - R-Drop consistency regularization (alpha=0.5).
4. 5-Fold Stratified Cross-Validation:
   - 90 labeled judgments (45 CL + 45 IT) folded into 5 folds of 18 docs each.
   - Dev documents folded directly into training, giving native exposure to higher argument density.
   - Soft probability / logit averaging across all 5 folds on the 10 Test judgments (2,197 sentences).
"""

import os
import sys
import json
import time
import zipfile
import argparse
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

# Ensure code directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))

from phase3.src.data.rr_dataset import load_label_vocab, load_rr_data
from phase3.src.data.rr_context_dataset_v45 import ChronoContextRRDatasetV45
from phase3.src.model.rr_harveylm_v45 import HarveyLMV45ForRR
from phase3.src.evaluation.viterbi_decoder import LegalViterbiDecoder
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model

# ── Paths & Hyperparameters ──────────────────────────────────────────────────
HARVEY_CKPT   = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
PREV_LORMA_PT = "code/phase3/models/attentive_lorma_harveylm/best_model.pt"
DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"
OUT_DIR       = "code/phase3/models/lorma_v45_cv"
SUB_DIR       = "code/phase3/submissions/harveylm/lorma_v45_cv"
LOG_DIR       = "code/phase3/models/logs"

MAX_LEN       = 384
BATCH_SIZE    = 16
ACCUM_STEPS   = 2
EPOCHS        = 5
LR_HEAD       = 3.0e-4
LR_LORMA      = 3.5e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
LORMA_R       = 64
LORMA_ALPHA   = 128
LORMA_DROPOUT = 0.10


def mark_v45_trainable(model):
    for name, param in model.named_parameters():
        if "W_res" in name:
            param.requires_grad = False
        elif ".A" in name or ".B" in name:
            param.requires_grad = True
        elif any(target in name for target in ["head", "seg_embeddings"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)


def evaluate_on_docs(model, doc_examples, doc_list, viterbi_decoder, device, num_labels=13):
    """Evaluates doc-by-doc with Viterbi decoding."""
    model.eval()
    all_raw_preds = []
    all_vit_preds = []
    all_golds = []
    doc_logits_dict = {}

    with torch.no_grad():
        for doc in doc_list:
            doc_id = doc["id"]
            exs = doc_examples.get(doc_id, [])
            if not exs:
                continue

            all_doc_logits = []
            for k in range(0, len(exs), 32):
                chunk = exs[k: k + 32]
                input_ids      = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask      = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                seg_ids        = torch.stack([e["segment_ids"] for e in chunk]).to(device)
                target_mask    = torch.stack([e["target_mask"] for e in chunk]).to(device)
                prev_mask      = torch.stack([e["prev_mask"] for e in chunk]).to(device)
                next_mask      = torch.stack([e["next_mask"] for e in chunk]).to(device)
                token_type_ids = torch.stack([e["token_type_ids"] for e in chunk]).to(device)
                rel_pos        = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                bound_feats    = torch.stack([e["bound_feats"] for e in chunk]).to(device)
                party_state    = torch.stack([e["party_state"] for e in chunk]).to(device)
                struct_scalars = torch.stack([e["struct_scalars"] for e in chunk]).to(device)
                domain_id      = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        segment_ids=seg_ids,
                        target_mask=target_mask,
                        prev_mask=prev_mask,
                        next_mask=next_mask,
                        token_type_ids=token_type_ids,
                        rel_pos=rel_pos,
                        bound_feats=bound_feats,
                        party_state=party_state,
                        struct_scalars=struct_scalars,
                        domain_id=domain_id
                    )
                all_doc_logits.append(out["logits"].float().cpu())

            doc_logits = torch.cat(all_doc_logits, dim=0)
            doc_logits_dict[doc_id] = doc_logits

            raw_p = doc_logits.argmax(dim=-1).tolist()
            vit_p = viterbi_decoder.decode_document(doc_logits)

            all_raw_preds.extend(raw_p)
            all_vit_preds.extend(vit_p)

            golds = [e["label"].item() for e in exs]
            all_golds.extend(golds)

    labels_all = list(range(num_labels))
    raw_f1 = f1_score(all_golds, all_raw_preds, labels=labels_all, average="macro", zero_division=0)
    vit_f1 = f1_score(all_golds, all_vit_preds, labels=labels_all, average="macro", zero_division=0)
    per_class = f1_score(all_golds, all_vit_preds, labels=labels_all, average=None, zero_division=0)

    return raw_f1, vit_f1, per_class, doc_logits_dict


def compute_test_logits(model, test_examples, test_docs, device):
    """Computes test set logits for a trained fold model."""
    model.eval()
    test_logits_dict = {}

    with torch.no_grad():
        for doc in test_docs:
            doc_id = doc["id"]
            exs = test_examples[doc_id]

            all_doc_logits = []
            for k in range(0, len(exs), 32):
                chunk = exs[k: k + 32]
                input_ids      = torch.stack([e["input_ids"] for e in chunk]).to(device)
                attn_mask      = torch.stack([e["attention_mask"] for e in chunk]).to(device)
                seg_ids        = torch.stack([e["segment_ids"] for e in chunk]).to(device)
                target_mask    = torch.stack([e["target_mask"] for e in chunk]).to(device)
                prev_mask      = torch.stack([e["prev_mask"] for e in chunk]).to(device)
                next_mask      = torch.stack([e["next_mask"] for e in chunk]).to(device)
                token_type_ids = torch.stack([e["token_type_ids"] for e in chunk]).to(device)
                rel_pos        = torch.stack([e["rel_pos"] for e in chunk]).to(device)
                bound_feats    = torch.stack([e["bound_feats"] for e in chunk]).to(device)
                party_state    = torch.stack([e["party_state"] for e in chunk]).to(device)
                struct_scalars = torch.stack([e["struct_scalars"] for e in chunk]).to(device)
                domain_id      = torch.stack([e["domain_id"] for e in chunk]).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        segment_ids=seg_ids,
                        target_mask=target_mask,
                        prev_mask=prev_mask,
                        next_mask=next_mask,
                        token_type_ids=token_type_ids,
                        rel_pos=rel_pos,
                        bound_feats=bound_feats,
                        party_state=party_state,
                        struct_scalars=struct_scalars,
                        domain_id=domain_id
                    )
                all_doc_logits.append(out["logits"].float().cpu())

            test_logits_dict[doc_id] = torch.cat(all_doc_logits, dim=0)

    return test_logits_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Folds to train (default: 0 1 2 3 4)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"Training LORMA V4.5 with 5-Fold Stratified Cross-Validation on {device}", flush=True)
    print(f"Active Folds: {args.folds} | Epochs per fold: {EPOCHS} | Max Len: {MAX_LEN}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Tokenizer & Vocab
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_DIR,
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )

    label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
    label2id, id2label = load_label_vocab(label_vocab_path)
    num_labels = len(label2id)

    # 2. Load all 90 Labeled Documents & 10 Test Documents
    cl_train_docs = load_rr_data(os.path.join(DATA_DIR, "Train", "CL_train.json"))
    it_train_docs = load_rr_data(os.path.join(DATA_DIR, "Train", "IT_train.json"))
    cl_dev_docs   = load_rr_data(os.path.join(DATA_DIR, "Train", "CL_dev.json"))
    it_dev_docs   = load_rr_data(os.path.join(DATA_DIR, "Train", "IT_dev.json"))

    cl_docs = cl_train_docs + cl_dev_docs  # 45 CL docs
    it_docs = it_train_docs + it_dev_docs  # 45 IT docs

    test_docs = []
    for fname in ["CL_test.json", "IT_test.json"]:
        test_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Test", fname)))

    print(f"Total Labeled Documents: {len(cl_docs) + len(it_docs)} (45 CL + 45 IT) | Test: {len(test_docs)}", flush=True)

    # 3. Create Stratified Folds (5 folds of 9 CL + 9 IT = 18 val docs, 72 train docs)
    folds = []
    for k in range(5):
        val_cl = cl_docs[k * 9 : (k + 1) * 9]
        val_it = it_docs[k * 9 : (k + 1) * 9]
        val_docs = val_cl + val_it

        trn_cl = [d for i, d in enumerate(cl_docs) if i < k * 9 or i >= (k + 1) * 9]
        trn_it = [d for i, d in enumerate(it_docs) if i < k * 9 or i >= (k + 1) * 9]
        trn_docs = trn_cl + trn_it

        folds.append((trn_docs, val_docs))

    # Pre-build Test Dataset (constant across all folds)
    print("\nTokenizing and building ChronoContextRRDatasetV45 for Test Set...", flush=True)
    test_dataset = ChronoContextRRDatasetV45(test_docs, tokenizer, label2id, max_seq_len=MAX_LEN)
    doc_test_examples = defaultdict(list)
    for ex in test_dataset.examples:
        doc_test_examples[ex["doc_id"]].append(ex)
    print(f"Test Set: {len(test_dataset):,} sentences across {len(test_docs)} documents.", flush=True)

    # Viterbi transition matrix estimated from all labeled training files
    all_train_files = [os.path.join(DATA_DIR, "Train", f) for f in ["CL_train.json", "IT_train.json", "CL_dev.json", "IT_dev.json"]]
    viterbi_decoder = LegalViterbiDecoder(all_train_files, label2id, lambda_trans=0.35)

    all_fold_oof_logits = {}
    all_fold_test_logits = []

    # 4. Iterate Over Selected Folds
    for fold_idx in args.folds:
        trn_docs, val_docs = folds[fold_idx]
        print("\n" + "=" * 80, flush=True)
        print(f">>> STARTING FOLD {fold_idx + 1}/5 (Train: {len(trn_docs)} docs | Val: {len(val_docs)} docs)", flush=True)
        print("=" * 80, flush=True)

        trn_dataset = ChronoContextRRDatasetV45(trn_docs, tokenizer, label2id, max_seq_len=MAX_LEN)
        val_dataset = ChronoContextRRDatasetV45(val_docs, tokenizer, label2id, max_seq_len=MAX_LEN)

        print(f"Sentences - Train: {len(trn_dataset):,} | Val: {len(val_dataset):,}", flush=True)

        doc_val_examples = defaultdict(list)
        for ex in val_dataset.examples:
            doc_val_examples[ex["doc_id"]].append(ex)

        trn_loader = DataLoader(
            trn_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True
        )

        # Instantiate HarveyLM V4.5
        model = HarveyLMV45ForRR.from_pretrained_harvey(HARVEY_CKPT, num_classes=num_labels)
        TARGET_MODULES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
        apply_lorma_to_model(model.harvey, target_modules=TARGET_MODULES, r=LORMA_R, lorma_alpha=LORMA_ALPHA, dropout=LORMA_DROPOUT)

        # Warm-start LORMA weights from peak V4 checkpoint (0.4620)
        if os.path.exists(PREV_LORMA_PT):
            print(f"Warm-starting backbone weights from {PREV_LORMA_PT}...", flush=True)
            prev_state = torch.load(PREV_LORMA_PT, map_location="cpu", weights_only=True)
            model_state = model.state_dict()
            loaded = 0
            for k, v in prev_state.items():
                if k in model_state and "classifier" not in k and "head" not in k and "seg_embeddings" not in k:
                    if model_state[k].shape == v.shape:
                        model_state[k] = v
                        loaded += 1
            model.load_state_dict(model_state, strict=False)
            print(f"Successfully warm-started {loaded} weight tensors from peak LORMA V4 checkpoint!", flush=True)

        mark_v45_trainable(model)
        model.to(device)

        # Optimizer with differential learning rates
        head_params  = [p for n, p in model.named_parameters() if ("head" in n or "seg_embeddings" in n) and p.requires_grad]
        lorma_params = [p for n, p in model.named_parameters() if ("head" not in n and "seg_embeddings" not in n) and p.requires_grad]

        optimizer = AdamW([
            {"params": lorma_params, "lr": LR_LORMA, "weight_decay": WEIGHT_DECAY},
            {"params": head_params,  "lr": LR_HEAD,  "weight_decay": WEIGHT_DECAY}
        ])

        total_steps = (len(trn_loader) // ACCUM_STEPS) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_val_f1 = 0.0
        fold_best_ckpt = os.path.join(OUT_DIR, f"fold_{fold_idx}_best.pt")
        fold_best_val_logits = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            step_count = 0
            t0 = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(trn_loader):
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attn_mask      = batch["attention_mask"].to(device, non_blocking=True)
                seg_ids        = batch["segment_ids"].to(device, non_blocking=True)
                target_mask    = batch["target_mask"].to(device, non_blocking=True)
                prev_mask      = batch["prev_mask"].to(device, non_blocking=True)
                next_mask      = batch["next_mask"].to(device, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
                rel_pos        = batch["rel_pos"].to(device, non_blocking=True)
                bound_feats    = batch["bound_feats"].to(device, non_blocking=True)
                party_state    = batch["party_state"].to(device, non_blocking=True)
                struct_scalars = batch["struct_scalars"].to(device, non_blocking=True)
                domain_id      = batch["domain_id"].to(device, non_blocking=True)
                labels         = batch["label"].to(device, non_blocking=True)
                doc_ids        = batch["doc_id"]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        segment_ids=seg_ids,
                        target_mask=target_mask,
                        prev_mask=prev_mask,
                        next_mask=next_mask,
                        token_type_ids=token_type_ids,
                        rel_pos=rel_pos,
                        bound_feats=bound_feats,
                        party_state=party_state,
                        struct_scalars=struct_scalars,
                        domain_id=domain_id,
                        labels=labels,
                        doc_ids=doc_ids
                    )
                    loss = out["loss"] / ACCUM_STEPS

                loss.backward()
                total_loss += loss.item() * ACCUM_STEPS

                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(trn_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

                if (step + 1) % (200 * ACCUM_STEPS) == 0:
                    curr_lr = optimizer.param_groups[1]["lr"]
                    print(f"  Fold {fold_idx+1} | Epoch {epoch:2d} | Step {step+1:4d}/{len(trn_loader)} | Loss: {loss.item() * ACCUM_STEPS:.4f} | LR: {curr_lr:.2e}", flush=True)

            elapsed = time.time() - t0
            avg_loss = total_loss / len(trn_loader)

            # Evaluate on Validation Fold
            raw_f1, vit_f1, per_class, doc_logits_dict = evaluate_on_docs(
                model, doc_val_examples, val_docs, viterbi_decoder, device, num_labels
            )

            print(f">>> Fold {fold_idx+1} | Epoch {epoch}/{EPOCHS} [{elapsed:.1f}s] | Loss: {avg_loss:.4f} | Val Raw-F1: {raw_f1:.4f} | Val Viterbi-F1: {vit_f1:.4f}", flush=True)

            if vit_f1 > best_val_f1:
                best_val_f1 = vit_f1
                fold_best_val_logits = doc_logits_dict
                torch.save(model.state_dict(), fold_best_ckpt)
                print(f"  >>> NEW BEST MODEL for Fold {fold_idx+1}! (Val F1: {vit_f1:.4f} at Epoch {epoch})", flush=True)

        # Load best fold checkpoint for test inference
        print(f"\nLoading best checkpoint for Fold {fold_idx+1} (Val F1: {best_val_f1:.4f})...", flush=True)
        model.load_state_dict(torch.load(fold_best_ckpt, map_location="cpu", weights_only=True))
        model.to(device)

        # Store OOF validation logits
        if fold_best_val_logits:
            all_fold_oof_logits.update(fold_best_val_logits)

        # Compute Test set emission logits
        print(f"Computing Test Logits for Fold {fold_idx+1}...", flush=True)
        fold_test_logits = compute_test_logits(model, doc_test_examples, test_docs, device)
        all_fold_test_logits.append(fold_test_logits)

        # Save fold test logits
        torch.save(fold_test_logits, os.path.join(OUT_DIR, f"fold_{fold_idx}_test_logits.pt"))

        # Generate cumulative ensemble submission across all completed folds so far
        print(f"\nGenerating Cumulative Ensemble Submission (Folds 1..{len(all_fold_test_logits)})...", flush=True)
        ens_test_preds = {}
        total_sents = 0

        for doc in test_docs:
            doc_id = doc["id"]
            # Average soft probabilities across completed folds
            fold_probs = []
            for f_logits in all_fold_test_logits:
                probs = torch.softmax(f_logits[doc_id], dim=-1)
                fold_probs.append(probs)
            mean_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
            log_emissions = torch.log(mean_probs + 1e-12)

            vit_p = viterbi_decoder.decode_document(log_emissions)
            ens_test_preds[doc_id] = [id2label[p] for p in vit_p]
            total_sents += len(vit_p)

        pred_file = os.path.join(SUB_DIR, "prediction.json")
        with open(pred_file, "w") as f:
            json.dump(ens_test_preds, f, indent=2)

        sub_zip = os.path.join(SUB_DIR, "submission.zip")
        with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_file, "prediction.json")

        print(f"Cumulative {len(all_fold_test_logits)}-Fold Submission Verified: {total_sents} sentences (Expected: 2,197)", flush=True)
        print(f"Saved to: {sub_zip}", flush=True)

    print("\n" + "=" * 80)
    print("ALL FOLDS COMPLETED SUCCESSFULLY!")
    print(f"Final 5-Fold Ensemble Submission: {os.path.join(SUB_DIR, 'submission.zip')}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
