import os
import sys
import json
import re
import zipfile
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file
from torchcrf import CRF

# Add root paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../phase1')))

from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from phase2.src.data.ner_dataset_split import NERDataset, get_label_maps

def get_entities(seq, id2label):
    entities = []
    entity = None
    for i, label_id in enumerate(seq):
        label = id2label.get(label_id, 'O')
        if label.startswith('B-'):
            if entity:
                entities.append(entity)
            entity = {'type': label[2:], 'start': i, 'end': i}
        elif label.startswith('I-'):
            if entity and entity['type'] == label[2:] and entity['end'] == i - 1:
                entity['end'] = i
            else:
                if entity:
                    entities.append(entity)
                entity = {'type': label[2:], 'start': i, 'end': i}
        else:
            if entity:
                entities.append(entity)
                entity = None
    if entity:
        entities.append(entity)
    return entities

def snap_to_word_boundaries(text, start, end):
    while start > 0 and re.match(r'[A-Za-z0-9]', text[start - 1]):
        start -= 1
    while end < len(text) and re.match(r'[A-Za-z0-9]', text[end]):
        end += 1
    while start < end and not re.match(r'[A-Za-z0-9]', text[start]):
        start += 1
    while end > start and not re.match(r'[A-Za-z0-9]', text[end - 1]):
        end -= 1
    return start, end, text[start:end]

class LegalEncDecForTokenClassificationCRF(nn.Module):
    def __init__(self, config, num_labels=25):
        super().__init__()
        self.num_labels = num_labels
        self.enc_dec = LegalEncoderDecoder(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.d_model, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seqlen = input_ids.shape
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        emb = self.enc_dec.embedding(input_ids) + self.enc_dec.pos_encoder(positions)

        enc_mask = None
        if attention_mask is not None:
            enc_mask = (1.0 - attention_mask.float()) * -10000.0
            enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)

        h = emb
        for layer in self.enc_dec.encoders:
            h = layer(h, enc_mask)

        h = self.enc_dec.norm(h)
        logits = self.classifier(self.dropout(h))

        if labels is not None:
            valid_mask = (labels != -100)
            labels_clamped = labels.clone()
            labels_clamped[~valid_mask] = 0
            crf_mask = attention_mask.bool() & valid_mask if attention_mask is not None else valid_mask
            loss = -self.crf(logits, labels_clamped, mask=crf_mask, reduction='mean')
            preds = self.crf.decode(logits, mask=crf_mask)
            return loss, logits, preds

        crf_mask = attention_mask.bool() if attention_mask is not None else None
        preds = self.crf.decode(logits, mask=crf_mask)
        return logits, preds

def evaluate(model, val_texts, val_golds, tokenizer, id2label, device, max_length=512):
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    encoded_docs = {}
    for doc_id, text in val_texts.items():
        encoded_docs[doc_id] = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)

    for doc_id, text in val_texts.items():
        encoded = encoded_docs[doc_id]
        input_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']

        preds_for_doc = []
        for i in range(0, len(input_ids), max_length):
            chunk_ids = input_ids[i:i + max_length]
            chunk_offsets = offsets[i:i + max_length]

            pad_len = max_length - len(chunk_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            attention_mask = [1] * len(chunk_ids) + [0] * pad_len
            chunk_ids = chunk_ids + [pad_id] * pad_len

            chunk_tensor = torch.tensor([chunk_ids], dtype=torch.long).to(device)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.bool).to(device)

            with torch.no_grad():
                _, preds = model(chunk_tensor, attention_mask=mask_tensor)
                chunk_preds = preds[0]

            chunk_preds = chunk_preds[:len(chunk_offsets)]
            pred_entities = get_entities(chunk_preds, id2label)

            for ent in pred_entities:
                start_tok = ent['start']
                end_tok = ent['end']
                if chunk_offsets[start_tok][0] < 0 or chunk_offsets[end_tok][1] <= 0:
                    continue
                start_char = chunk_offsets[start_tok][0]
                end_char = chunk_offsets[end_tok][1]
                preds_for_doc.append((start_char, end_char, ent['type'], text[start_char:end_char]))

        gold_annotations = val_golds.get(doc_id, [])
        gold_set = set([(ann[0], ann[1], ann[2]) for ann in gold_annotations])
        pred_set = set([(ann[0], ann[1], ann[2]) for ann in preds_for_doc])

        for p in pred_set:
            if p in gold_set:
                true_positives[p[2]] += 1
            else:
                false_positives[p[2]] += 1

        for g in gold_set:
            if g not in pred_set:
                false_negatives[g[2]] += 1

    all_classes = set(list(true_positives.keys()) + list(false_positives.keys()) + list(false_negatives.keys()))
    total_f1 = 0
    num_valid = 0
    for cls in sorted(list(all_classes)):
        tp = true_positives[cls]
        fp = false_positives[cls]
        fn = false_negatives[cls]
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        total_f1 += f1
        num_valid += 1
    return total_f1 / num_valid if num_valid > 0 else 0

def run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"=== Legal Encoder-Decoder NER Training on {device} ===")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels_file = "code/phase2/data/Phase_2_LNER/Train/labels.txt"
    label2id, id2label = get_label_maps(labels_file)
    num_labels = len(label2id)

    print("Loading datasets...")
    train_texts, train_golds = {}, {}
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-gold.json") as f:
        train_golds.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-gold.json") as f:
        train_golds.update(json.load(f))

    val_keys = list(train_texts.keys())[:6]
    val_texts = {k: train_texts[k] for k in val_keys}
    val_golds = {k: train_golds[k] for k in val_keys}

    train_dataset = NERDataset(train_texts, train_golds, tokenizer, label2id, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    print("Initializing LegalEncoderDecoder backbone...")
    ed_dir = "code/phase1/models/final_architectures/4_EncoderDecoder/epoch_10"
    config = EncDecConfig.from_pretrained(ed_dir)
    model = LegalEncDecForTokenClassificationCRF(config, num_labels=num_labels)
    state_dict = load_file(f"{ed_dir}/model.safetensors")
    model.enc_dec.load_state_dict(state_dict, strict=False)

    # Unfreeze entire bidirectional encoder + classifier + CRF
    for name, param in model.named_parameters():
        if "encoders" in name or "pos_encoder" in name or "enc_dec.norm" in name or "classifier" in name or "crf" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_p:,} / {total_p:,} ({100*trainable_p/total_p:.2f}%)")

    model.to(device)

    crf_params = [p for n, p in model.named_parameters() if 'crf' in n and p.requires_grad]
    cls_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    enc_params = [p for n, p in model.named_parameters() if 'crf' not in n and 'classifier' not in n and p.requires_grad]
    optimizer = AdamW([
        {'params': crf_params, 'lr': 1e-3, 'weight_decay': 0.01},
        {'params': cls_params, 'lr': 2e-4, 'weight_decay': 0.01},
        {'params': enc_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])

    out_model_dir = "code/phase2/models/lner_enc_dec_crf"
    os.makedirs(out_model_dir, exist_ok=True)

    epochs = 5
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, _, _ = model(ids, attention_mask=mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if b_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {b_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

        model.eval()
        val_f1 = evaluate(model, val_texts, val_golds, tokenizer, id2label, device)
        print(f"Epoch {epoch+1} Tracking Validation Macro-F1: {val_f1:.4f}")

        if val_f1 >= best_f1 or epoch == 0:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{out_model_dir}/best_model.pt")
            print(f"*** New best LegalEncoderDecoder model saved! (Val Macro-F1: {val_f1:.4f}) ***")

    # Generate Test Predictions
    print("\nLoading best model for test prediction generation...")
    model.load_state_dict(torch.load(f"{out_model_dir}/best_model.pt", map_location=device))
    model.eval()

    with open("code/phase2/data/Phase_2_LNER/Test/fold3-text.json") as f:
        test_texts = json.load(f)

    max_length = 512
    raw_preds = {}
    snapped_preds = {}

    for doc_id, text in test_texts.items():
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        input_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']

        preds_for_doc = []
        for i in range(0, len(input_ids), max_length):
            chunk_ids = input_ids[i:i + max_length]
            chunk_offsets = offsets[i:i + max_length]

            pad_len = max_length - len(chunk_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            att_mask = [1] * len(chunk_ids) + [0] * pad_len
            chunk_ids = chunk_ids + [pad_id] * pad_len

            chunk_t = torch.tensor([chunk_ids], dtype=torch.long).to(device)
            mask_t = torch.tensor([att_mask], dtype=torch.bool).to(device)

            with torch.no_grad():
                _, preds = model(chunk_t, attention_mask=mask_t)
                chunk_preds = preds[0]

            chunk_preds = chunk_preds[:len(chunk_offsets)]
            pred_entities = get_entities(chunk_preds, id2label)

            for ent in pred_entities:
                st = ent['start']
                ed = ent['end']
                if chunk_offsets[st][0] < 0 or chunk_offsets[ed][1] <= 0:
                    continue
                start_c = chunk_offsets[st][0]
                end_c = chunk_offsets[ed][1]
                preds_for_doc.append([start_c, end_c, ent['type'], text[start_c:end_c]])

        # Deduplicate
        seen_raw = set()
        dedup_raw = []
        for p in preds_for_doc:
            k = (p[0], p[1], p[2])
            if k not in seen_raw:
                seen_raw.add(k)
                dedup_raw.append(p)
        raw_preds[doc_id] = dedup_raw

        # Snap
        seen_snap = set()
        dedup_snap = []
        for start_c, end_c, ltype, _ in dedup_raw:
            ns, ne, nt = snap_to_word_boundaries(text, start_c, end_c)
            if ns < ne:
                k = (ns, ne, ltype)
                if k not in seen_snap:
                    seen_snap.add(k)
                    dedup_snap.append([ns, ne, ltype, nt])
        snapped_preds[doc_id] = dedup_snap

    # 1. Raw Submission Export
    raw_sub_dir = "code/phase2/submissions/enc_dec/4_crf"
    os.makedirs(raw_sub_dir, exist_ok=True)
    raw_json = os.path.join(raw_sub_dir, "prediction.json")
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(raw_preds, f, ensure_ascii=False, indent=2)
    for zname in ["prediction.zip", "submission.zip"]:
        zpath = os.path.join(raw_sub_dir, zname)
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(raw_json, arcname="prediction.json")
    print(f"Exported Raw LegalEncoderDecoder submission: {raw_sub_dir}")

    # 2. Snapped Submission Export
    snap_sub_dir = "code/phase2/submissions/enc_dec/5_crf_boundary_snapping"
    os.makedirs(snap_sub_dir, exist_ok=True)
    snap_json = os.path.join(snap_sub_dir, "prediction.json")
    with open(snap_json, "w", encoding="utf-8") as f:
        json.dump(snapped_preds, f, ensure_ascii=False, indent=2)
    for zname in ["prediction.zip", "submission.zip"]:
        zpath = os.path.join(snap_sub_dir, zname)
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(snap_json, arcname="prediction.json")
    print(f"Exported Snapped LegalEncoderDecoder submission: {snap_sub_dir}")
    print(f"=== LegalEncoderDecoder Finished! Best Val F1: {best_f1:.4f} ===")

if __name__ == "__main__":
    run()
