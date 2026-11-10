import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phase2.src.model.ner_harveylm import HarveyForTokenClassification
from phase2.src.model.architecture import HarveyTransformerConfig
from phase2.src.data.ner_dataset_split import NERDataset, get_label_maps
from phase2.src.model.lorma_from_scratch import apply_lorma_to_model

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
from torchcrf import CRF

class HarveyForTokenClassificationCRF(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        
        from phase2.src.model.ner_harveylm import HarveyLM
        self.harvey = HarveyLM(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        kwargs["output_hidden_states"] = True
        
        outputs = self.harvey(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[0]
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output) # [batch, seq_len, num_labels]
        
        loss = None
        preds = None
        
        if labels is not None:
            mask = (labels != -100)
            
            labels_clamped = labels.clone()
            labels_clamped[~mask] = 0
            
            if attention_mask is not None:
                crf_mask = attention_mask.bool() & mask
            else:
                crf_mask = mask
                
            loss = -self.crf(logits, labels_clamped, mask=crf_mask, reduction='mean')
            preds = self.crf.decode(logits, mask=crf_mask)
            
            class LossOutput:
                pass
            out = LossOutput()
            out.loss = loss
            out.logits = logits
            out.preds = preds
            return out
            
        preds = self.crf.decode(logits)
        class PredOutput:
            pass
        out = PredOutput()
        out.logits = logits
        out.preds = preds
        return out

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
                outputs = model(chunk_tensor, attention_mask=mask_tensor)
                preds = outputs.preds[0] # CRF decode returns list of lists
                
            chunk_preds = preds[:len(chunk_offsets)]
            pred_entities = get_entities(chunk_preds, id2label)
            
            for ent in pred_entities:
                start_tok = ent['start']
                end_tok = ent['end']
                
                if chunk_offsets[start_tok][0] < 0 or chunk_offsets[end_tok][1] <= 0:
                    continue
                    
                start_char = chunk_offsets[start_tok][0]
                end_char = chunk_offsets[end_tok][1]
                label_type = ent['type']
                preds_for_doc.append((start_char, end_char, label_type, text[start_char:end_char]))
                
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
    num_valid_classes = 0
    
    for cls in sorted(list(all_classes)):
        tp = true_positives[cls]
        fp = false_positives[cls]
        fn = false_negatives[cls]
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        total_f1 += f1
        num_valid_classes += 1
        
    macro_f1 = total_f1 / num_valid_classes if num_valid_classes > 0 else 0
    return macro_f1

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    labels_file = "code/phase2/data/Phase_2_LNER/Train/labels.txt"
    label2id, id2label = get_label_maps(labels_file)
    num_labels = len(label2id)
    
    print("Loading data...")
    train_texts, train_golds = {}, {}
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-gold.json", 'r', encoding='utf-8') as f:
        train_golds.update(json.load(f))
        
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-gold.json", 'r', encoding='utf-8') as f:
        train_golds.update(json.load(f))
        
    # Validation split - exactly same 6 docs to monitor, but let's ALSO train on everything except those 6 first?
    # Actually to CRUSH leaderboard we should train on ALL data (100%).
    # Let's still use the 6 docs for validation evaluation, but include them in training too!
    
    val_keys = list(train_texts.keys())[:6] # just grab first 6 for monitoring
    val_texts_dict = {k: train_texts[k] for k in val_keys}
    val_golds_dict = {k: train_golds[k] for k in val_keys}
    
    # Calculate class weights
    counts = {'RESP': 381, 'DATE': 1575, 'COURT': 1013, 'WIT': 187, 'APP': 407, 'STAT': 1260, 'A.COUNSEL': 195, 'PREC': 1183, 'JUDGE': 245, 'R.COUNSEL': 184, 'CASENO': 760, 'AUTH': 1003}
    weights = []
    for i in range(num_labels):
        label = id2label[i]
        if label == "O":
            weights.append(0.5) # Downweight O
        else:
            base = label[2:]
            count = counts.get(base, 1)
            weights.append(2000.0 / count)
            
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"Training on ALL {len(train_texts)} documents (100% of data). Validating on {len(val_texts_dict)} overlapping documents for tracking.")
            
    train_dataset = NERDataset(train_texts, train_golds, tokenizer, label2id, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print("Initializing LORMA-CRF model with FROZEN BACKBONE...")
    checkpoint_dir = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
    config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    model = HarveyForTokenClassificationCRF(config, num_labels=num_labels)
    model.harvey.load_state_dict(torch.load(f"{checkpoint_dir}/model.safetensors", weights_only=True), strict=False)
    
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    # LORMA r=64
    model = apply_lorma_to_model(model, target_modules=target_modules, r=64, lorma_alpha=128, dropout=0.1)
    
    # Load already trained LORMA & classifier weights to warm-start emission layer!
    lorma_checkpoint = "code/phase2/models/lner_custom_lorma_merged_weighted_full/best_model.pt"
    if os.path.exists(lorma_checkpoint):
        print(f"Loading warm-start LORMA & classifier weights from {lorma_checkpoint}...")
        pretrained_dict = torch.load(lorma_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(pretrained_dict, strict=False)
        print("Warm-start weights loaded successfully.")
    
    # CRITICAL FIX: Freeze everything except LORMA, classifier, and crf
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name or "classifier" in name or "crf" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
            
    model.to(device)
    
    crf_params = [p for n, p in model.named_parameters() if 'crf' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'crf' not in n and p.requires_grad]
    optimizer = AdamW([
        {'params': crf_params, 'lr': 1e-3, 'weight_decay': 0.01},
        {'params': other_params, 'lr': 5e-5, 'weight_decay': 0.01}
    ])
    epochs = 5
    out_dir = "code/phase2/models/lner_custom_lorma_crf_full"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Starting training...")
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
        
        model.eval()
        val_f1 = evaluate(model, val_texts_dict, val_golds_dict, tokenizer, id2label, device)
        print(f"Epoch {epoch+1} Tracking Validation Strict Macro-F1: {val_f1:.4f}")
        
        trainable_state_dict = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}
        torch.save(trainable_state_dict, f"{out_dir}/epoch_{epoch+1}.pt")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(trainable_state_dict, f"{out_dir}/best_model.pt")
            print(f"*** New best model saved! (F1: {val_f1:.4f}) ***")

if __name__ == "__main__":
    train()
