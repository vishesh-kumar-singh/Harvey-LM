"""
Extract and cache contextual sentence representations from fine-tuned LoRA V3.
Outputs: code/phase3/data/cache/harveylm_context_features.pt
Also generates official test submission in code/phase3/submissions/harveylm/lora_v2/submission.zip.
"""
import os
import sys


import json
import time
import zipfile
from collections import defaultdict

import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))

from phase3.src.data.rr_dataset import load_label_vocab, load_rr_data
from phase3.src.data.rr_context_dataset_v3 import ChronoContextRRDataset
from phase3.src.model.rr_harveylm_v3 import HarveyForRhetoricalRoleV3
from phase3.src.evaluation.viterbi_decoder import LegalViterbiDecoder
from phase2.src.model.lora_from_scratch import apply_lora_to_model

HARVEY_CKPT   = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
BEST_MODEL    = "code/phase3/models/lora_v2_harveylm/best_model.pt"
SUB_DIR       = "code/phase3/submissions/harveylm/lora_v2"
CACHE_OUT     = "code/phase3/data/cache/harveylm_context_features.pt"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"

MAX_LEN       = 256
POS_DIM       = 128
DOM_DIM       = 32
LORA_R        = 64
LORA_ALPHA    = 128
LORA_DROPOUT  = 0.10
CHUNK_SIZE    = 32

os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs("code/phase3/data/cache", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"Extracting Contextual Features from LoRA V3 Best Model on {device}")
print("=" * 70)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_DIR,
    bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
)
label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
label2id, id2label = load_label_vocab(label_vocab_path)
num_labels = len(label2id)

print(f"Loading base model from {HARVEY_CKPT}...")
model = HarveyForRhetoricalRoleV3.from_pretrained_harvey(
    HARVEY_CKPT, num_labels=num_labels, pos_dim=POS_DIM, domain_dim=DOM_DIM
)
target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
model = apply_lora_to_model(model, target_modules=target_modules, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

print(f"Loading trained weights from {BEST_MODEL}...")
state = torch.load(BEST_MODEL, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# Load all splits
train_docs = []
for fname in ["CL_train.json", "IT_train.json"]:
    train_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Train", fname)))

dev_docs = []
for fname in ["CL_dev.json", "IT_dev.json"]:
    dev_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Train", fname)))

test_docs = []
for fname in ["CL_test.json", "IT_test.json"]:
    test_docs.extend(load_rr_data(os.path.join(DATA_DIR, "Test", fname)))

print(f"Train docs: {len(train_docs)} | Dev docs: {len(dev_docs)} | Test docs: {len(test_docs)}")

train_files = [os.path.join(DATA_DIR, "Train", f) for f in ["CL_train.json", "IT_train.json"]]
viterbi_decoder = LegalViterbiDecoder(train_files, label2id, lambda_trans=0.35)

# 1. Generate Test Predictions with Chunking
print("\nGenerating Test Predictions with chunk size 32...")
test_dataset = ChronoContextRRDataset(test_docs, tokenizer, label2id, max_length=MAX_LEN)
test_examples = defaultdict(list)
for ex in test_dataset.examples:
    test_examples[ex["doc_id"]].append(ex)

test_predictions = {}
with torch.no_grad():
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

# 2. Extract and Cache Features for Train, Dev, Test
print(f"\nExtracting Contextual Sentence Features for all splits...")
splits_data = [
    ("train", train_docs, ChronoContextRRDataset(train_docs, tokenizer, label2id, max_length=MAX_LEN)),
    ("dev",   dev_docs,   ChronoContextRRDataset(dev_docs, tokenizer, label2id, max_length=MAX_LEN)),
    ("test",  test_docs,  test_dataset)
]

cached_splits = {}
with torch.no_grad():
    for split_name, s_docs, s_dataset in splits_data:
        split_exs = defaultdict(list)
        for ex in s_dataset.examples:
            split_exs[ex["doc_id"]].append(ex)

        doc_feature_list = []
        for doc in s_docs:
            doc_id = doc["id"]
            exs = split_exs[doc_id]
            n_sents = len(exs)

            all_embs = []
            torch.cuda.empty_cache()
            for s_idx in range(0, n_sents, CHUNK_SIZE):
                chunk = exs[s_idx: s_idx + CHUNK_SIZE]
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
print(f"\nFeatures successfully saved to: {CACHE_OUT} (size: {os.path.getsize(CACHE_OUT)/1e6:.1f} MB)")
print("Done!")
