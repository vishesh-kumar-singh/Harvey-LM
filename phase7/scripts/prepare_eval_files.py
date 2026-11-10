#!/usr/bin/env python3
"""
Prepare evaluation files and candidate ID lists for Phase 7 PCR.
Generates:
1. code/phase7/data/eval_truth/dev_truth.json (Ground truth mapping for Dev split)
2. code/phase7/data/eval_truth/dev_candidate_ids.json (List of all valid Dev candidate IDs)
3. code/phase7/data/eval_truth/test_candidate_ids.json (List of all valid Test candidate IDs)
4. code/phase7/data/eval_truth/train_candidate_ids.json (List of all valid Train candidate IDs)
"""

import os
import json

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/Phase_7_PCR"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/eval_truth"))
os.makedirs(EVAL_DIR, exist_ok=True)

def main():
    # 1. Dev truth
    dev_path = os.path.join(DATA_DIR, "Dev/dev.json")
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
        
    dev_truth = {}
    for q_id, q_val in dev_data.items():
        dev_truth[q_id] = q_val.get("relevant_candidates", [])
        
    dev_truth_path = os.path.join(EVAL_DIR, "dev_truth.json")
    with open(dev_truth_path, "w", encoding="utf-8") as f:
        json.dump(dev_truth, f, indent=2)
    print(f"Wrote dev_truth.json ({len(dev_truth)} queries) to {dev_truth_path}")
    
    # 2. Candidate ID lists
    for split, filename in [
        ("dev", "Dev/dev_candidates.json"),
        ("test", "Test/test_candidates.json"),
        ("train", "Train/train_candidates.json")
    ]:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        cand_ids = sorted(list(c_data.keys()))
        out_path = os.path.join(EVAL_DIR, f"{split}_candidate_ids.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cand_ids, f, indent=2)
        print(f"Wrote {split}_candidate_ids.json ({len(cand_ids)} candidates) to {out_path}")

if __name__ == "__main__":
    main()
