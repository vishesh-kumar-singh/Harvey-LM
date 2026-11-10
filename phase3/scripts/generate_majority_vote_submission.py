"""
Generate Majority Voting Submissions for Phase 3 Test Set.
Supports:
  1. 4-Model Weighted Majority Vote (Optimal Ensemble, LoRA Standalone, Fused CRF, Full FT)
  2. 5-Model Consensus Vote (Optimal Ensemble, LoRA Standalone, LORMA Standalone, Fused CRF, Full FT)
"""
import os
import json
import zipfile
from collections import Counter

# Input predictions
MODELS = {
    "Optimal_Ensemble": "code/phase3/submissions/ensemble/optimal_calibrated/prediction.json",
    "LoRA_Standalone": "code/phase3/submissions/harveylm/lora_v2/prediction.json",
    "CRF_Fused": "code/phase3/submissions/harveylm/hierarchical_fused/prediction.json",
    "Full_FT": "code/phase3/submissions/harveylm/full_finetuning/prediction.json",
    "LORMA_Standalone": "code/phase3/submissions/harveylm/lorma_v3/prediction.json"
}

WEIGHTS_4 = {
    "Optimal_Ensemble": 0.4452,
    "LoRA_Standalone":  0.4436,
    "CRF_Fused":        0.4305,
    "Full_FT":          0.3412
}

def create_submission(pred_dict, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "prediction.json")
    zip_path = os.path.join(output_dir, "submission.zip")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pred_dict, f, indent=2)
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "prediction.json")
        
    total_docs = len(pred_dict)
    total_sents = sum(len(v) for v in pred_dict.values())
    print(f"[{name}]")
    print(f"  Docs: {total_docs} | Sentences: {total_sents}")
    print(f"  JSON: {json_path}")
    print(f"  ZIP:  {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)\n")

def main():
    print("Loading test predictions from all models...")
    preds = {}
    for name, path in MODELS.items():
        assert os.path.exists(path), f"Missing prediction file: {path}"
        with open(path, "r", encoding="utf-8") as f:
            preds[name] = json.load(f)
            
    doc_keys = list(preds["Optimal_Ensemble"].keys())
    
    # 1. 4-Model Weighted Majority Vote
    w4_preds = {}
    # 2. 5-Model Consensus Vote
    five_preds = {}
    
    for d in doc_keys:
        n = len(preds["Optimal_Ensemble"][d])
        w4_list, five_list = [], []
        
        for i in range(n):
            # 4-Model Weighted
            score_dict = Counter()
            for m, w in WEIGHTS_4.items():
                label = preds[m][d][i]
                score_dict[label] += w
            best_w4 = score_dict.most_common(1)[0][0]
            w4_list.append(best_w4)
            
            # 5-Model Plurality (with priority to top models on ties)
            c5 = Counter([preds[m][d][i] for m in MODELS.keys()])
            top_count = c5.most_common(1)[0][1]
            top_candidates = [k for k, v in c5.items() if v == top_count]
            if len(top_candidates) == 1:
                best_5m = top_candidates[0]
            else:
                # Break tie using highest performing model
                best_5m = preds["Optimal_Ensemble"][d][i] if preds["Optimal_Ensemble"][d][i] in top_candidates else preds["LoRA_Standalone"][d][i]
            five_list.append(best_5m)
            
        w4_preds[d] = w4_list
        five_preds[d] = five_list
        
    create_submission(
        w4_preds,
        "code/phase3/submissions/ensemble/majority_vote_4m",
        "4-Model Weighted Majority Vote"
    )
    
    create_submission(
        five_preds,
        "code/phase3/submissions/ensemble/majority_vote_5m",
        "5-Model Consensus Vote"
    )

if __name__ == "__main__":
    main()
