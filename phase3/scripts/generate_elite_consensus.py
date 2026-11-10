"""
Generate the Elite 3 Consensus Ensemble submission:
1. Attentive Token Head LORMA V4 (LB: 0.4620, weight: 1.20)
2. LORMA V3 Standalone (LB: 0.4599, weight: 1.00)
3. Hierarchical BiLSTM-Transformer-CRF on LORMA (LB: 0.4583, weight: 1.00)
"""
import os
import json
import zipfile
from collections import Counter

SUB_DIR = "code/phase3/submissions/ensemble/elite_consensus"
os.makedirs(SUB_DIR, exist_ok=True)

models = {
    "lorma_v4":  {"path": "code/phase3/submissions/harveylm/attentive_lorma/prediction.json", "weight": 1.20},
    "lorma_v3":  {"path": "code/phase3/submissions/harveylm/lorma_v3/prediction.json", "weight": 1.00},
    "crf_lorma": {"path": "code/phase3/submissions/harveylm/hierarchical_lorma/prediction.json", "weight": 1.00},
}

preds = {name: json.load(open(cfg["path"])) for name, cfg in models.items()}
doc_ids = list(preds["lorma_v4"].keys())

final_preds = {}
total_sents = 0
vote_breakdown = Counter()

for doc_id in doc_ids:
    p4 = preds["lorma_v4"][doc_id]
    p3 = preds["lorma_v3"][doc_id]
    pc = preds["crf_lorma"][doc_id]
    n = len(p4)
    total_sents += n
    doc_out = []

    for i in range(n):
        v4, v3, vc = p4[i], p3[i], pc[i]
        tallies = Counter()
        tallies[v4] += models["lorma_v4"]["weight"]
        tallies[v3] += models["lorma_v3"]["weight"]
        tallies[vc] += models["crf_lorma"]["weight"]

        winner = tallies.most_common(1)[0][0]
        doc_out.append(winner)

        if v4 == v3 == vc:
            vote_breakdown["unanimous"] += 1
        elif v4 == v3:
            vote_breakdown["v4_v3"] += 1
        elif v4 == vc:
            vote_breakdown["v4_crf"] += 1
        elif v3 == vc:
            vote_breakdown["v3_crf_override"] += 1
        else:
            vote_breakdown["three_way_tie_v4_wins"] += 1

    final_preds[doc_id] = doc_out

print(f"Total documents: {len(final_preds)}")
print(f"Total sentences: {total_sents}")
print(f"Breakdown of consensus:")
for k, v in vote_breakdown.items():
    print(f"  {k:26s}: {v:4d} ({v/total_sents*100:5.1f}%)")

pred_file = os.path.join(SUB_DIR, "prediction.json")
with open(pred_file, "w") as f:
    json.dump(final_preds, f, indent=2)

zip_file = os.path.join(SUB_DIR, "submission.zip")
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(pred_file, "prediction.json")

print(f"\nSaved submission to: {zip_file}")
