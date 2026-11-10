import json
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import os
import sys
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phase2.src.model.ner_harveylm import HarveyForTokenClassification
from phase2.src.model.architecture import HarveyTransformerConfig
from phase2.src.data.ner_dataset_split import get_label_maps
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
        
        preds = self.crf.decode(logits)
        class PredOutput:
            pass
        out = PredOutput()
        out.logits = logits
        out.preds = preds
        return out

def generate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    labels_file = "code/phase2/data/Phase_2_LNER/Train/labels.txt"
    label2id, id2label = get_label_maps(labels_file)
    num_labels = len(label2id)
    
    print("Loading test data...")
    test_texts = {}
    with open("code/phase2/data/Phase_2_LNER/Test/fold3-text.json") as f:
        test_texts = json.load(f)
            
    checkpoint_dir = "code/phase1/models/final_architectures/2_HarveyLM/epoch_99"
    config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    model = HarveyForTokenClassificationCRF(config, num_labels=num_labels)
    model.harvey.load_state_dict(torch.load(f"{checkpoint_dir}/model.safetensors", weights_only=True), strict=False)
    model.to(device)
    
    target_modules = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    model = apply_lorma_to_model(model, target_modules=target_modules, r=64, lorma_alpha=128, dropout=0.1)
    
    # Load best weights
    best_weights = torch.load("code/phase2/models/lner_custom_lorma_crf_full/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_weights, strict=False)
    
    model.eval()
    
    max_length = 512
    predictions = {}
    
    print("Generating predictions for fold3...")
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
                preds = outputs.preds[0]
                
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
                # Format: [start, end, label, text]
                preds_for_doc.append([start_char, end_char, label_type, entity_text])
                
        # Deduplicate predictions exactly like they might be evaluated
        pred_set = []
        seen = set()
        for p in preds_for_doc:
            key = (p[0], p[1], p[2])
            if key not in seen:
                seen.add(key)
                pred_set.append(p)
                
        predictions[doc_id] = pred_set
        
    out_dir = "code/phase2/submissions/harveylm/4_lorma_crf"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "prediction.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
    print(f"Predictions saved to {out_file}")
    
    # Create zip files (prediction.zip and submission.zip)
    import zipfile
    for zname in ["prediction.zip", "submission.zip"]:
        zip_path = os.path.join(out_dir, zname)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(out_file, arcname="prediction.json")
        print(f"Zip created at {zip_path}")

if __name__ == "__main__":
    generate()
