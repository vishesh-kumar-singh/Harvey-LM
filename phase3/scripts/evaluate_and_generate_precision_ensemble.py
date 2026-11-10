"""
Phase 3 - Precision-Routed Specialist Consensus Ensemble.

Idea:
Instead of democratic majority voting where every model has an equal vote across all classes,
weight each model's prediction on sentence i by its demonstrated class-specific precision:
    Weight(m, c) = BaseWeight[m] * (max(Precision_m(c), 0.10) ** alpha)

When Model A is the specialized expert on Class A (e.g., V5 on None/RulingByPresentCourt,
CRF on RulingByLowerCourt/Issue, V4 on Statute/Ratio), its vote dominates.
"""

import os
import sys
import json
import zipfile
import subprocess
from collections import Counter
from sklearn.metrics import classification_report, f1_score

# 1. Load Ground Truth
with open("code/phase3/data/eval_truth/cl_dev_truth.json") as f:
    cl_truth = json.load(f)
with open("code/phase3/data/eval_truth/it_dev_truth.json") as f:
    it_truth = json.load(f)

truth = {}
truth.update(cl_truth)
truth.update(it_truth)
dev_docs = list(truth.keys())
golds = [lbl for d in dev_docs for lbl in truth[d]]

with open("code/phase3/data/eval_truth/labels.json") as f:
    labels_dict = json.load(f)
labels = list(labels_dict.keys())

# 2. Define Candidate Models & Base Weights
models_config = {
    "v4": {
        "dev": "code/phase3/models/attentive_lorma_harveylm/dev_prediction.json",
        "test": "code/phase3/submissions/harveylm/attentive_lorma/prediction.json",
        "base_weight": 1.20,
        "lb_score": 0.4620
    },
    "v3": {
        "dev": "code/phase3/models/lorma_v3_harveylm/dev_prediction.json",
        "test": "code/phase3/submissions/harveylm/lorma_v3/prediction.json",
        "base_weight": 1.05,
        "lb_score": 0.4599
    },
    "crf_lorma": {
        "dev": "code/phase3/models/hierarchical_crf_lorma/dev_prediction.json",
        "test": "code/phase3/submissions/harveylm/hierarchical_lorma/prediction.json",
        "base_weight": 1.00,
        "lb_score": 0.4583
    },
    "crf_attn": {
        "dev": "code/phase3/models/hierarchical_crf_attentive_lorma/dev_prediction.json",
        "test": "code/phase3/submissions/harveylm/hierarchical_attentive_lorma/prediction.json",
        "base_weight": 0.95,
        "lb_score": None
    },
    "v5": {
        "dev": "code/phase3/models/lorma_v5_harveylm/dev_prediction.json",
        "test": "code/phase3/submissions/harveylm/lorma_v5/prediction.json",
        "base_weight": 0.90,
        "lb_score": 0.4369
    }
}

# 3. Load Predictions & Compute Empirical Precision on Dev
dev_preds = {}
precisions = {}

for m_name, cfg in models_config.items():
    with open(cfg["dev"]) as f:
        dev_preds[m_name] = json.load(f)
    plist = [p for d in dev_docs for p in dev_preds[m_name][d]]
    rep = classification_report(golds, plist, labels=labels, output_dict=True, zero_division=0)
    precisions[m_name] = {lbl: rep[lbl]["precision"] for lbl in labels}

print("=" * 80)
print("CLASS PRECISION CHAMPIONS (SPECIALIST ROUTING MAP)")
print("=" * 80)
for lbl in labels:
    best_m = max(models_config.keys(), key=lambda m: precisions[m][lbl])
    best_p = precisions[best_m][lbl]
    print(f"  {lbl:24s} -> Champion: {best_m:10s} (Precision: {best_p:.4f})")

# 4. Evaluate Precision-Weighted Consensus on Dev
alpha = 2.5
dev_out = {}
total_dev_sents = 0

for d in dev_docs:
    d_preds = []
    n = len(dev_preds["v4"][d])
    total_dev_sents += n
    for i in range(n):
        tallies = Counter()
        for m_name, cfg in models_config.items():
            p_val = dev_preds[m_name][d][i]
            p = precisions[m_name].get(p_val, 0.50)
            w = cfg["base_weight"] * (max(p, 0.10) ** alpha)
            tallies[p_val] += w
        d_preds.append(tallies.most_common(1)[0][0])
    dev_out[d] = d_preds

dev_out_file = "code/phase3/scratch/precision_routed_dev.json"
with open(dev_out_file, "w") as f:
    json.dump(dev_out, f, indent=2)

print("\n" + "=" * 80)
print("OFFICIAL EVALUATION ON DEV SET:")
print("=" * 80)
subprocess.run([
    sys.executable, "code/Evaluation Script/evaluate_phase3.py",
    "--truth-cl", "code/phase3/data/eval_truth/cl_dev_truth.json",
    "--truth-it", "code/phase3/data/eval_truth/it_dev_truth.json",
    "--prediction", dev_out_file,
    "--labels", "code/phase3/data/eval_truth/labels.json"
])

# 5. Generate Test Set Predictions with the Same Precision Routing
SUB_DIR = "code/phase3/submissions/ensemble/precision_specialist_consensus"
os.makedirs(SUB_DIR, exist_ok=True)

test_preds = {}
for m_name, cfg in models_config.items():
    with open(cfg["test"]) as f:
        test_preds[m_name] = json.load(f)

test_docs = list(test_preds["v4"].keys())
final_test_preds = {}
total_test_sents = 0
vote_breakdown = Counter()

for d in test_docs:
    d_out = []
    n = len(test_preds["v4"][d])
    total_test_sents += n
    for i in range(n):
        votes = {m: test_preds[m][d][i] for m in models_config}
        tallies = Counter()
        for m_name, cfg in models_config.items():
            p_val = votes[m_name]
            p = precisions[m_name].get(p_val, 0.50)
            w = cfg["base_weight"] * (max(p, 0.10) ** alpha)
            tallies[p_val] += w

        winner = tallies.most_common(1)[0][0]
        d_out.append(winner)

        # Check consensus
        unique_votes = set(votes.values())
        if len(unique_votes) == 1:
            vote_breakdown["5_unanimous"] += 1
        elif len(unique_votes) == 2:
            vote_breakdown["majority_4_or_3"] += 1
        else:
            vote_breakdown["split_routed_by_precision"] += 1

    final_test_preds[d] = d_out

test_pred_file = os.path.join(SUB_DIR, "prediction.json")
with open(test_pred_file, "w") as f:
    json.dump(final_test_preds, f, indent=2)

zip_file = os.path.join(SUB_DIR, "submission.zip")
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(test_pred_file, "prediction.json")

print("\n" + "=" * 80)
print("TEST SUBMISSION VERIFICATION:")
print("=" * 80)
print(f"Total test documents: {len(final_test_preds)} (Expected: 10)")
print(f"Total test sentences: {total_test_sents} (Expected: 2,197)")
print("Breakdown of decision types:")
for k, v in vote_breakdown.items():
    print(f"  {k:30s}: {v:4d} ({v/total_test_sents*100:5.1f}%)")
print(f"\nSubmission zip saved: {zip_file}")
