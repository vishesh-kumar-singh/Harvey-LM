#!/usr/bin/env python3
"""
Create reproducible Train / Held-out Validation splits for Phase 6.
Extracts dev_truth.json in the exact format expected by evaluate_phase6.py:
  {"doc_id": "reference summary", ...}
"""

import os
import json
import random

RAW_TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/Phase_6_Summ/Train/train.json"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/eval_truth"))

def create_splits(seed=42, val_ratio=0.10):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    print(f"Loading raw train data from: {RAW_TRAIN_PATH}")
    with open(RAW_TRAIN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Raw record count: {len(data)}")
    
    # Filter empty summaries
    clean_data = []
    dropped = 0
    for r in data:
        doc = r.get("document", "").strip()
        summ = r.get("summary", "").strip()
        rid = str(r.get("id"))
        if not doc or not summ:
            dropped += 1
            continue
        clean_data.append({"id": rid, "document": doc, "summary": summ})
        
    print(f"Dropped {dropped} invalid/empty records. Remaining clean records: {len(clean_data)}")
    
    # Set seed for deterministic split
    random.seed(seed)
    random.shuffle(clean_data)
    
    val_size = int(len(clean_data) * val_ratio)
    dev_split = clean_data[:val_size]
    train_split = clean_data[val_size:]
    
    print(f"\nCreated Splits (seed={seed}, val_ratio={val_ratio:.2f}):")
    print(f"  Train split size: {len(train_split)} cases")
    print(f"  Dev split size:   {len(dev_split)} cases")
    
    train_out_path = os.path.join(PROCESSED_DIR, "train.json")
    dev_out_path = os.path.join(PROCESSED_DIR, "dev.json")
    dev_truth_path = os.path.join(EVAL_DIR, "dev_truth.json")
    
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(dev_out_path, "w", encoding="utf-8") as f:
        json.dump(dev_split, f, indent=2)
        
    # Construct dev_truth dict for evaluate_phase6.py: {"doc_id": "reference summary", ...}
    dev_truth = {r["id"]: r["summary"] for r in dev_split}
    with open(dev_truth_path, "w", encoding="utf-8") as f:
        json.dump(dev_truth, f, indent=2)
        
    print(f"\nSaved processed files:")
    print(f"  - {train_out_path} ({len(train_split)} cases)")
    print(f"  - {dev_out_path} ({len(dev_split)} cases)")
    print(f"  - {dev_truth_path} ({len(dev_truth)} truth entries for evaluate_phase6.py)")

if __name__ == "__main__":
    create_splits(seed=42, val_ratio=0.10)
