import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
import os
import sys
import copy
import zipfile
import re
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phase2.src.model.ner_harveylm import HarveyForTokenClassification, FocalLoss
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
    f1_scores = []
    for c in all_classes:
        tp = true_positives[c]
        fp = false_positives[c]
        fn = false_negatives[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
        
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return macro_f1

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

def generate_predictions(model, tokenizer, id2label, device, out_dir, snapped_dir, max_length=512):
    print("Generating test predictions for fold3...")
    with open("code/phase2/data/Phase_2_LNER/Test/fold3-text.json") as f:
        test_texts = json.load(f)
        
    model.eval()
    predictions = {}
    snapped_predictions = {}
    
    for doc_id, text in tqdm(test_texts.items()):
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
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
                entity_text = text[start_char:end_char]
                preds_for_doc.append([start_char, end_char, label_type, entity_text])
                
        pred_set = []
        seen = set()
        for p in preds_for_doc:
            key = (p[0], p[1], p[2])
            if key not in seen:
                seen.add(key)
                pred_set.append(p)
        predictions[doc_id] = pred_set
        
        # Also compute snapped version
        final_ents = []
        seen_spans = set()
        for start, end, label, _ in pred_set:
            new_start, new_end, new_text = snap_to_word_boundaries(text, start, end)
            if new_start < new_end:
                span_key = (new_start, new_end, label)
                if span_key not in seen_spans:
                    final_ents.append([new_start, new_end, label, new_text])
                    seen_spans.add(span_key)
        snapped_predictions[doc_id] = final_ents
        
    # Save standard predictions
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "prediction.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    for zname in ["prediction.zip", "submission.zip"]:
        zip_path = os.path.join(out_dir, zname)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(out_file, arcname="prediction.json")
    print(f"Standard KD predictions saved to {out_dir}")
    
    # Save snapped predictions
    os.makedirs(snapped_dir, exist_ok=True)
    snapped_file = os.path.join(snapped_dir, "prediction.json")
    with open(snapped_file, 'w', encoding='utf-8') as f:
        json.dump(snapped_predictions, f, ensure_ascii=False, indent=2)
    for zname in ["prediction.zip", "submission.zip"]:
        zip_path = os.path.join(snapped_dir, zname)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(snapped_file, arcname="prediction.json")
    print(f"Snapped KD predictions saved to {snapped_dir}")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    labels_file = "code/phase2/data/Phase_2_LNER/Train/labels.txt"
    label2id, id2label = get_label_maps(labels_file)
    num_labels = len(label2id)
    
    print("Loading dataset...")
    train_texts, train_golds = {}, {}
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold1-gold.json", 'r', encoding='utf-8') as f:
        train_golds.update(json.load(f))
        
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-text.json") as f:
        train_texts.update(json.load(f))
    with open("code/phase2/data/Phase_2_LNER/Train/fold2-gold.json", 'r', encoding='utf-8') as f:
        train_golds.update(json.load(f))
        
    val_keys = list(train_texts.keys())[:6]
    val_texts_dict = {k: train_texts[k] for k in val_keys}
    val_golds_dict = {k: train_golds[k] for k in val_keys}
    
    print(f"Training on ALL {len(train_texts)} documents (100% of data). Validating on {len(val_texts_dict)} tracking documents.")
    
    train_dataset = NERDataset(train_texts, train_golds, tokenizer, label2id, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 1. Load TEACHER MODEL (16 layers, 228M params + LORMA)
    print("Loading TEACHER model (16 layers, HarveyLM + LORMA)...")
    checkpoint_dir = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
    config_teacher = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    teacher = HarveyForTokenClassification(config_teacher, num_labels=num_labels)
    teacher_base_weights = torch.load(f"{checkpoint_dir}/model.safetensors", weights_only=True)
    teacher.harvey.load_state_dict(teacher_base_weights, strict=False)
    
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    teacher = apply_lorma_to_model(teacher, target_modules=target_modules, r=64, lorma_alpha=128, dropout=0.1)
    
    teacher_best_weights = torch.load("code/phase2/models/lner_custom_lorma_merged_weighted_full/best_model.pt", map_location='cpu', weights_only=True)
    teacher.load_state_dict(teacher_best_weights, strict=False)
    
    # Freeze teacher completely
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    teacher.to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher initialized and frozen. Total parameters: {teacher_params:,}")
    
    # 2. Build STUDENT MODEL (8 layers, 121M params - 47% parameter reduction)
    print("Building STUDENT model (8 layers, compact architecture)...")
    config_student = copy.deepcopy(config_teacher)
    config_student.n_layers = 8
    student = HarveyForTokenClassification(config_student, num_labels=num_labels)
    
    # Initialize student layers from teacher pre-trained weights (every 2nd layer)
    harvey_student_state = student.harvey.state_dict()
    for k in harvey_student_state.keys():
        if k in ['tok_embeddings.weight', 'norm.weight']:
            harvey_student_state[k] = teacher_base_weights[k]
        elif k.startswith('layers.'):
            parts = k.split('.')
            s_layer = int(parts[1])
            t_layer = s_layer * 2
            t_key = k.replace(f'layers.{s_layer}.', f'layers.{t_layer}.')
            if t_key in teacher_base_weights:
                harvey_student_state[k] = teacher_base_weights[t_key]
                
    student.harvey.load_state_dict(harvey_student_state)
    # Warm-start student classifier head with teacher's trained classifier
    if hasattr(teacher, 'classifier'):
        student.classifier.weight.data.copy_(teacher.classifier.weight.data)
        if teacher.classifier.bias is not None and student.classifier.bias is not None:
            student.classifier.bias.data.copy_(teacher.classifier.bias.data)
            
    student.to(device)
    student_params = sum(p.numel() for p in student.parameters())
    reduction_pct = 100 * (1 - student_params / teacher_params)
    print(f"Student initialized! Total parameters: {student_params:,} (Compression: {reduction_pct:.1f}% reduction)")
    
    # Loss functions: Hard CE + Soft KL Divergence (Hinton KD)
    temperature = 2.0
    alpha_kd = 0.5 # 0.5 Hard CE + 0.5 Soft KD
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 6
    out_dir = "code/phase2/models/lner_student_distillation_full"
    os.makedirs(out_dir, exist_ok=True)
    
    best_f1 = 0
    print("Starting Knowledge Distillation training...")
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0
        total_hard_loss = 0
        total_kd_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass teacher (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits # [batch, seq_len, num_labels]
                
            # Forward pass student
            student_outputs = student(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits # [batch, seq_len, num_labels]
            
            # Mask active non-padding tokens
            active_mask = (labels != -100) & (attention_mask.bool())
            
            active_student_logits = student_logits[active_mask]
            active_teacher_logits = teacher_logits[active_mask]
            active_labels = labels[active_mask]
            
            # Hard Ground-Truth CE Loss
            loss_hard = ce_loss_fn(active_student_logits, active_labels)
            
            # Soft KD Loss (Hinton KL Divergence)
            p_teacher = F.softmax(active_teacher_logits / temperature, dim=-1)
            log_p_student = F.log_softmax(active_student_logits / temperature, dim=-1)
            loss_kd = (temperature ** 2) * kl_loss_fn(log_p_student, p_teacher)
            
            loss = alpha_kd * loss_hard + (1.0 - alpha_kd) * loss_kd
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_hard_loss += loss_hard.item()
            total_kd_loss += loss_kd.item()
            
            if batch_idx % 25 == 0:
                print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Total Loss: {loss.item():.4f} | Hard CE: {loss_hard.item():.4f} | Soft KD: {loss_kd.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        student.eval()
        val_f1 = evaluate(student, val_texts_dict, val_golds_dict, tokenizer, id2label, device, max_length=512)
        print(f"Epoch {epoch} Tracking Validation Strict Macro-F1: {val_f1:.4f}")
        
        torch.save(student.state_dict(), os.path.join(out_dir, f"epoch_{epoch}.pt"))
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(student.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"*** New best student model saved! (F1: {val_f1:.4f}) ***")
            
    print(f"\nDistillation training complete! Best Tracking Macro-F1: {best_f1:.4f}")
    
    # Load best model and generate test predictions
    best_weights = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device, weights_only=True)
    student.load_state_dict(best_weights)
    
    sub_dir = "code/phase2/submissions/harveylm/6_kd_student"
    snapped_dir = "code/phase2/submissions/harveylm/7_kd_student_boundary_snapping"
    generate_predictions(student, tokenizer, id2label, device, sub_dir, snapped_dir, max_length=512)

if __name__ == "__main__":
    train()
