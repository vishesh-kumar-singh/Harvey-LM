"""
Phase 3 - Teacher Soft Target Extraction for Knowledge Distillation.

Combines:
1. Peak HarveyLM LORMA V3 Model representations
2. Grand Super-Consensus (0.4785 LB Macro-F1) peak labels
3. Stratified Cross-Validation ensemble test emissions

Outputs:
code/phase3/data/cache/teacher_soft_targets.pt
Mapping { doc_id: Tensor(N, 13) } containing calibrated probability distributions.
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))
from phase3.src.data.rr_dataset import load_label_vocab, load_rr_data
from phase3.src.data.rr_context_dataset_v3 import ChronoContextRRDataset
from phase3.src.model.rr_harveylm_v3 import HarveyForRhetoricalRoleV3
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model

HARVEY_CKPT   = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
DATA_DIR      = "code/phase3/data/raw/Phase_3_RR"
BEST_LORMA    = "code/phase3/models/harveylm/lorma_v3_harveylm/best_model.pt"
TOKENIZER_DIR = "code/phase1/models/tokenizer/legal_bpe_tokenizer.json"
GRAND_PRED    = "code/phase3/submissions/ensemble/grand_consensus_v45/prediction.json"
CV_LOGITS_DIR = "code/phase3/models/harveylm/lorma_v45_cv"
OUTPUT_CACHE  = "code/phase3/data/cache/teacher_soft_targets.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"Generating Grand Teacher Soft Targets on {device}...")
print("=" * 70)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_DIR,
    bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
)
label_vocab_path = os.path.join(DATA_DIR, "Train", "label_vocab.json")
label2id, id2label = load_label_vocab(label_vocab_path)
num_labels = len(label2id)

# 1. Load Raw Documents
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

all_docs = train_docs + dev_docs + test_docs
print(f"Total documents to extract soft targets: {len(all_docs)} (Train: {len(train_docs)}, Dev: {len(dev_docs)}, Test: {len(test_docs)})")

# 2. Build Dataset
dataset = ChronoContextRRDataset(all_docs, tokenizer, label2id, max_length=256)
from collections import defaultdict
doc_examples = defaultdict(list)
for ex in dataset.examples:
    doc_examples[ex["doc_id"]].append(ex)

# 3. Load Teacher Model
print(f"Loading Base HarveyLM from {HARVEY_CKPT}...")
model = HarveyForRhetoricalRoleV3.from_pretrained_harvey(
    HARVEY_CKPT, num_labels=num_labels, pos_dim=128, domain_dim=32,
    hidden_dropout=0.15, class_weights=None, gamma=2.0
)
model = apply_lorma_to_model(
    model, target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    r=64, lorma_alpha=128, dropout=0.10
)
print(f"Loading fine-tuned LORMA V3 checkpoint from {BEST_LORMA}...")
sd = torch.load(BEST_LORMA, map_location="cpu")
model.load_state_dict(sd, strict=False)
model.to(device).bfloat16().eval()

# 4. Load Grand Consensus (0.4785 LB) predictions
grand_preds = {}
if os.path.exists(GRAND_PRED):
    with open(GRAND_PRED) as f:
        grand_preds = json.load(f)
    print(f"Loaded Grand Consensus peak labels for {len(grand_preds)} test docs.")

# 5. Load CV Ensemble Test Logits
cv_test_logits = defaultdict(list)
for fold_idx in range(5):
    lp = os.path.join(CV_LOGITS_DIR, f"fold_{fold_idx}_test_logits.pt")
    if os.path.exists(lp):
        flogits = torch.load(lp, map_location="cpu")
        for did, ltensor in flogits.items():
            cv_test_logits[did].append(ltensor.float())

mean_cv_test_logits = {}
for did, llist in cv_test_logits.items():
    if llist:
        mean_cv_test_logits[did] = torch.stack(llist).mean(0)
print(f"Loaded 5-fold CV test logits for {len(mean_cv_test_logits)} test docs.")

teacher_targets = {}

with torch.no_grad():
    for doc in all_docs:
        doc_id = doc["id"]
        exs = doc_examples.get(doc_id, [])
        if not exs:
            continue

        doc_logits_list = []
        for k in range(0, len(exs), 32):
            chunk = exs[k: k + 32]
            input_ids = torch.stack([e["input_ids"] for e in chunk]).to(device)
            attn_mask = torch.stack([e["attention_mask"] for e in chunk]).to(device)
            target_mask = torch.stack([e["target_mask"] for e in chunk]).to(device)
            rel_pos = torch.stack([e["rel_pos"] for e in chunk]).to(device)
            domain_id = torch.stack([e["domain_id"] for e in chunk]).to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                target_mask=target_mask,
                rel_pos=rel_pos,
                domain_id=domain_id
            )
            doc_logits_list.append(out["logits"].float().cpu())

        base_logits = torch.cat(doc_logits_list, dim=0)  # (N, 13)

        # For test documents: blend with CV ensemble and Grand Consensus
        if doc_id in grand_preds:
            # 1. Base probabilities from model
            p_base = F.softmax(base_logits, dim=-1)
            # 2. CV ensemble probabilities if available
            if doc_id in mean_cv_test_logits and mean_cv_test_logits[doc_id].shape[0] == p_base.shape[0]:
                p_cv = F.softmax(mean_cv_test_logits[doc_id], dim=-1)
                p_blend = 0.5 * p_base + 0.5 * p_cv
            else:
                p_blend = p_base

            # 3. Incorporate 0.4785 Grand Consensus (sharp teacher signals with label smoothing eps=0.10)
            g_labels = grand_preds[doc_id]
            one_hot = torch.zeros((len(g_labels), num_labels))
            eps = 0.10
            one_hot.fill_(eps / (num_labels - 1))
            for i, l_name in enumerate(g_labels):
                lid = label2id[l_name]
                one_hot[i, lid] = 1.0 - eps

            # Weighted teacher blend: 60% Grand Consensus + 40% soft ensemble
            final_p = 0.60 * one_hot + 0.40 * p_blend
            teacher_targets[doc_id] = final_p
        else:
            # For train/dev documents: temperature softened probabilities
            p = F.softmax(base_logits / 1.5, dim=-1)
            teacher_targets[doc_id] = p

print(f"\nExtracted teacher soft targets for all {len(teacher_targets)} documents.")
torch.save(teacher_targets, OUTPUT_CACHE)
print(f"Saved cached teacher soft targets to {OUTPUT_CACHE} ({os.path.getsize(OUTPUT_CACHE) / 1024 / 1024:.2f} MB).")
