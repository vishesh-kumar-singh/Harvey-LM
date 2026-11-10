import json
import re
import os
import zipfile

def snap_to_word_boundaries(text, start, end):
    # Expand left to include word characters
    while start > 0 and re.match(r'[A-Za-z0-9]', text[start - 1]):
        start -= 1
        
    # Expand right to include word characters
    while end < len(text) and re.match(r'[A-Za-z0-9]', text[end]):
        end += 1
        
    # Shrink left to remove leading non-word characters
    while start < end and not re.match(r'[A-Za-z0-9]', text[start]):
        start += 1
        
    # Shrink right to remove trailing non-word characters
    while end > start and not re.match(r'[A-Za-z0-9]', text[end - 1]):
        end -= 1
        
    return start, end, text[start:end]

print("Loading test texts...")
with open("code/phase2/data/Phase_2_LNER/Test/fold3-text.json") as f:
    texts = json.load(f)
    
print("Loading CRF predictions...")
with open("code/phase2/submissions/harveylm/4_lorma_crf/prediction.json") as f:
    preds = json.load(f)

snapped_preds = {}
for doc_id, entities in preds.items():
    text = texts[doc_id]
    final_ents = []
    seen_spans = set()
    
    for start, end, label, _ in entities:
        new_start, new_end, new_text = snap_to_word_boundaries(text, start, end)
        if new_start < new_end:
            span_key = (new_start, new_end, label)
            if span_key not in seen_spans:
                final_ents.append([new_start, new_end, label, new_text])
                seen_spans.add(span_key)
            
    snapped_preds[doc_id] = final_ents
    
out_dir = "code/phase2/submissions/harveylm/5_lorma_crf_boundary_snapping"
os.makedirs(out_dir, exist_ok=True)
pred_path = os.path.join(out_dir, "prediction.json")
with open(pred_path, "w", encoding="utf-8") as f:
    json.dump(snapped_preds, f, ensure_ascii=False, indent=2)

for zname in ["prediction.zip", "submission.zip"]:
    zip_path = os.path.join(out_dir, zname)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_path, arcname="prediction.json")
    print(f"Created {zip_path}")

print("Saved snapped CRF predictions!")
