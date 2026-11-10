"""
Export the best calibrated sequence ensemble (0.4681 Dev Macro-F1).
"""
import os
import sys
import json
import zipfile
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from phase3.src.model.hierarchical_crf import HierarchicalLegalDocCRF

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SUB_DIR = "code/phase3/submissions/ensemble/optimal_v2"
os.makedirs(SUB_DIR, exist_ok=True)

def load_model(ckpt_path, embed_dim, num_labels=13):
    m = HierarchicalLegalDocCRF(embed_dim=embed_dim, pos_dim=128, hidden_dim=512, num_labels=num_labels)
    s = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(s, strict=False)
    m.to(DEVICE)
    m.eval()
    return m

def get_emiss(model, docs):
    res = []
    with torch.no_grad():
        for d in docs:
            emb = d["embeddings"].unsqueeze(0).to(DEVICE)
            pos = d["positions"].unsqueeze(0).to(DEVICE)
            pos_emb = model.pos_enc(pos)
            x = model.in_proj(torch.cat([emb, pos_emb], dim=-1))
            lstm_out, _ = model.bilstm(x)
            trans_out = model.doc_transformer(x)
            h = model.norm(lstm_out + trans_out)
            res.append(model.emission(h).squeeze(0).cpu())
    return res

fused_data = torch.load("code/phase3/data/cache/harveylm_fused_features.pt", map_location="cpu", weights_only=True)
full_data  = torch.load("code/phase3/data/cache/harveylm_full_features.pt",  map_location="cpu", weights_only=True)
lorma_data = torch.load("code/phase3/data/cache/harveylm_lorma_features.pt", map_location="cpu", weights_only=True)
lora_data  = torch.load("code/phase3/data/cache/harveylm_context_features.pt", map_location="cpu", weights_only=True)
trans_data = torch.load("code/phase3/data/cache/empirical_transitions.pt", map_location="cpu", weights_only=True)

id2label = fused_data["id2label"]
label2id = fused_data["label2id"]
num_labels = len(id2label)

dev_fused = fused_data["dev"]
test_fused = fused_data["test"]

m_fused = load_model("code/phase3/models/hierarchical_crf_fused/best_model.pt", 2048)
m_full  = load_model("code/phase3/models/hierarchical_crf_full/best_model.pt", 1024)
m_lorma = load_model("code/phase3/models/hierarchical_crf_lorma/best_model.pt", 1024)
m_lora  = load_model("code/phase3/models/hierarchical_crf_lora/best_model.pt", 1024)

e_f  = get_emiss(m_fused, dev_fused)
e_u  = get_emiss(m_full, full_data["dev"])
e_l  = get_emiss(m_lorma, lorma_data["dev"])
e_lo = get_emiss(m_lora, lora_data["dev"])

trans = (0.45 * m_fused.crf.transitions.detach().cpu() +
         0.35 * m_full.crf.transitions.detach().cpu() +
         0.20 * m_lorma.crf.transitions.detach().cpu()) * 1.25

start_probs = (trans_data["start_counts"] + 0.1) / (trans_data["start_counts"] + 0.1).sum()
end_probs = (trans_data["end_counts"] + 0.1) / (trans_data["end_counts"] + 0.1).sum()

dev_blended = []
for i in range(len(dev_fused)):
    p = (0.45 * F.softmax(e_f[i], dim=-1) +
         0.30 * F.softmax(e_u[i], dim=-1) +
         0.15 * F.softmax(e_l[i], dim=-1) +
         0.10 * F.softmax(e_lo[i], dim=-1))
    dev_blended.append(torch.log(p.clamp(min=1e-8)))

def viterbi(log_e, b_vec):
    N, K = log_e.shape
    dp = torch.zeros(N, K)
    bp = torch.zeros(N, K, dtype=torch.long)
    dp[0] = log_e[0] + b_vec + torch.log(start_probs.clamp(min=1e-8))
    for t in range(1, N):
        scores = dp[t - 1].unsqueeze(1) + trans
        best_scores, best_tags = scores.max(dim=0)
        dp[t] = log_e[t] + b_vec + best_scores
        bp[t] = best_tags
    dp[-1] += torch.log(end_probs.clamp(min=1e-8))
    path = [int(dp[-1].argmax().item())]
    for t in range(N - 1, 0, -1):
        path.append(int(bp[t, path[-1]].item()))
    path.reverse()
    return path

# Discovered optimal bias vector
b_vec = torch.zeros(num_labels)
b_vec[label2id["Fact"]]                   = -0.40
b_vec[label2id["Issue"]]                  = +1.60
b_vec[label2id["ArgumentPetitioner"]]      = +0.13
b_vec[label2id["ArgumentRespondent"]]      = +1.30
b_vec[label2id["Statute"]]                 = +0.68
b_vec[label2id["PrecedentReliedUpon"]]     = +1.30
b_vec[label2id["PrecedentNotReliedUpon"]]  = -0.17
b_vec[label2id["RulingByLowerCourt"]]      = +0.72
b_vec[label2id["RulingByPresentCourt"]]    = -1.50
b_vec[label2id["None"]]                    = -0.50

dev_preds_dict = {}
for i, d in enumerate(dev_fused):
    doc_id = d["id"]
    path = viterbi(dev_blended[i], b_vec)
    dev_preds_dict[doc_id] = [id2label[p] for p in path]

dev_pred_file = os.path.join(SUB_DIR, "dev_prediction.json")
with open(dev_pred_file, "w") as f:
    json.dump(dev_preds_dict, f, indent=2)

print("=" * 70)
print("OFFICIAL PHASE 3 EVALUATION ON DEV SET:")
print("=" * 70)
import subprocess
subprocess.run([
    sys.executable, "code/Evaluation Script/evaluate_phase3.py",
    "--truth-cl", "code/phase3/data/eval_truth/cl_dev_truth.json",
    "--truth-it", "code/phase3/data/eval_truth/it_dev_truth.json",
    "--prediction", dev_pred_file,
    "--labels", "code/phase3/data/eval_truth/labels.json"
])
print("=" * 70)

# Generate Test Submission
print("Generating test predictions...")
e_test_f  = get_emiss(m_fused, test_fused)
e_test_u  = get_emiss(m_full, full_data["test"])
e_test_l  = get_emiss(m_lorma, lorma_data["test"])
e_test_lo = get_emiss(m_lora, lora_data["test"])

test_preds_dict = {}
total_test = 0
for i, d in enumerate(test_fused):
    doc_id = d["id"]
    n = d["n_sents"]
    total_test += n
    p = (0.45 * F.softmax(e_test_f[i], dim=-1) +
         0.30 * F.softmax(e_test_u[i], dim=-1) +
         0.15 * F.softmax(e_test_l[i], dim=-1) +
         0.10 * F.softmax(e_test_lo[i], dim=-1))
    log_e = torch.log(p.clamp(min=1e-8))
    path = viterbi(log_e, b_vec)
    assert len(path) == n
    test_preds_dict[doc_id] = [id2label[p] for p in path]

pred_file = os.path.join(SUB_DIR, "prediction.json")
with open(pred_file, "w") as f:
    json.dump(test_preds_dict, f, indent=2)

zip_file = os.path.join(SUB_DIR, "submission.zip")
with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(pred_file, "prediction.json")
print(f"Saved: {zip_file} (total sentences: {total_test})")
print("Done!")
