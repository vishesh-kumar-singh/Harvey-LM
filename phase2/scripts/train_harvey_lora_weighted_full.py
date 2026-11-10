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
from phase2.src.model.lora_from_scratch import apply_lora_to_model

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

def evaluate(model, val_texts, val_golds, tokenizer, id2label, device, max_length=2048):
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
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)[0].cpu().numpy()
                
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
    f1_texts, f2_texts = {}, {}
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-text.json") as f:
        f1_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-gold.json", 'r', encoding='utf-8') as f:
        f1_golds = json.load(f)
        
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-text.json") as f:
        f2_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-gold.json", 'r', encoding='utf-8') as f:
        f2_golds = json.load(f)
        
    f1_keys = list(f1_texts.keys())
    f2_keys = list(f2_texts.keys())
    
    # 3 from fold1, 3 from fold2
    val_keys = f1_keys[:3] + f2_keys[:3]
    
    train_texts = {}
    train_golds = {}
    val_texts_dict = {}
    val_golds_dict = {}
    
    for k in f1_keys:
        train_texts[k] = f1_texts[k]
        train_golds[k] = f1_golds.get(k, [])
        if k in val_keys:
            val_texts_dict[k] = f1_texts[k]
            val_golds_dict[k] = f1_golds.get(k, [])
            
    for k in f2_keys:
        train_texts[k] = f2_texts[k]
        train_golds[k] = f2_golds.get(k, [])
        if k in val_keys:
            val_texts_dict[k] = f2_texts[k]
            val_golds_dict[k] = f2_golds.get(k, [])
            
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
    
    print(f"Training on {len(train_texts)} documents, Validating on {len(val_texts_dict)} documents.")
            
    train_dataset = NERDataset(train_texts, train_golds, tokenizer, label2id, max_length=2048)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("Initializing LoRA model with FROZEN BACKBONE and CLASS WEIGHTS...")
    checkpoint_dir = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
    config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    model = HarveyForTokenClassification(config, num_labels=num_labels)
    model.harvey.load_state_dict(torch.load(f"{checkpoint_dir}/model.safetensors", weights_only=True), strict=False)
    
    # Inject class weights
    model.class_weights = class_weights
    
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    # LoRA r=64
    model = apply_lora_to_model(model, target_modules=target_modules, r=64, lora_alpha=128, lora_dropout=0.1)
    
    # CRITICAL FIX: Freeze everything except LoRA and classifier
    for name, param in model.named_parameters():
        if "lora" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
            
    model.to(device)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
    epochs = 20
    out_dir = "code/phase2/models/lner_custom_lora_merged_weighted_full"
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
        print(f"Epoch {epoch+1} Validation Strict Macro-F1: {val_f1:.4f}")
        
        trainable_state_dict = {k: v for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad}
        torch.save(trainable_state_dict, f"{out_dir}/epoch_{epoch+1}.pt")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(trainable_state_dict, f"{out_dir}/best_model.pt")
            print(f"*** New best model saved! (F1: {val_f1:.4f}) ***")

if __name__ == "__main__":
    train()
