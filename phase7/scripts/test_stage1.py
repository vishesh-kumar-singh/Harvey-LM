#!/usr/bin/env python3
"""
Test Stage 1: CSLS Anti-Hubness + Statutory Regex Prior on existing HarveyLM LORMA checkpoint.
"""
import os
import sys
import json
import time
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase7.src.model.pcr_retriever import UniversalLegalRetriever
from phase7.src.data.pcr_dataset import PCRDocDataset, collate_docs
from phase7.src.evaluation.eval_utils import compute_pcr_metrics

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
EVAL_DIR = os.path.join(ROOT_CODE, "phase7/data/eval_truth")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
CKPT_PATH = os.path.join(ROOT_CODE, "phase7/models/harveylm/lorma/best_model.pt")

STATUTE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|article|art\.?|order|rule|clause)\s*"
    r"(\d+[a-zA-Z]*(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)"
    r"(?:\s+of\s+the\s+([a-zA-Z\s]{3,35}?act|[a-zA-Z\s]{3,25}?code|constitution))?",
    re.IGNORECASE,
)

def extract_statutory_signatures(text: str):
    signatures = set()
    for match in STATUTE_PATTERN.finditer(text.lower()):
        num_part = re.sub(r"\s+", "", match.group(1))
        act_part = match.group(2)
        if act_part:
            act_tokens = [w[:4] for w in act_part.split() if w not in {"the", "of", "and", "act", "code"}]
            act_slug = "_".join(act_tokens[:3])
            signatures.add(f"stat_{num_part}_{act_slug}" if act_slug else f"stat_{num_part}")
        else:
            signatures.add(f"stat_{num_part}")
    return signatures

def build_statutory_tversky_matrix(queries_dict, candidates_dict, query_ids, cand_ids):
    print("Building Statutory Tversky overlap matrix...")
    cand_sigs = {}
    for cid in cand_ids:
        txt = candidates_dict[cid].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        cand_sigs[cid] = extract_statutory_signatures(txt)

    query_sigs = {}
    for qid in query_ids:
        txt = queries_dict[qid].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        query_sigs[qid] = extract_statutory_signatures(txt)

    df = Counter()
    for sigs in cand_sigs.values():
        for s in sigs:
            df[s] += 1
    num_c = len(cand_ids)
    idf = {s: math.log(1.0 + num_c / (1.0 + count)) for s, count in df.items()}

    sim_matrix = np.zeros((len(query_ids), len(cand_ids)), dtype=np.float32)
    for i, qid in enumerate(query_ids):
        q_set = query_sigs[qid]
        if not q_set:
            continue
        q_weights = {s: idf.get(s, 1.0) for s in q_set}
        q_total = sum(q_weights.values())

        for j, cid in enumerate(cand_ids):
            c_set = cand_sigs[cid]
            if not c_set:
                continue
            inter = q_set.intersection(c_set)
            if not inter:
                continue
            inter_w = sum(q_weights[s] for s in inter)
            c_only_w = sum(idf.get(s, 1.0) for s in (c_set - q_set))
            q_only_w = q_total - inter_w
            sim_matrix[i, j] = inter_w / (inter_w + 0.7 * q_only_w + 0.3 * c_only_w + 1e-6)
    return sim_matrix

def apply_csls_anti_hubness(sim_matrix: np.ndarray, k_neighbors: int = 10, beta_hub: float = 0.35) -> np.ndarray:
    k_eff = min(k_neighbors, sim_matrix.shape[0])
    topk_query_sims = np.partition(sim_matrix, -k_eff, axis=0)[-k_eff:, :]
    hubness_penalty = np.mean(topk_query_sims, axis=0, keepdims=True)
    calibrated = sim_matrix - beta_hub * hubness_penalty
    mean_q = np.mean(calibrated, axis=1, keepdims=True)
    std_q = np.std(calibrated, axis=1, keepdims=True) + 1e-6
    return (calibrated - mean_q) / std_q

@torch.no_grad()
def encode_docs(model, doc_dict, tokenizer, max_seq_len, batch_size, device):
    model.eval()
    dataset = PCRDocDataset(doc_dict, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_docs, num_workers=2)
    doc_ids_all = []
    embeds_list = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.encode(ids, mask)
        embeds_list.append(embeds.float().cpu())
        doc_ids_all.extend(batch["doc_ids"])
    return doc_ids_all, torch.cat(embeds_list, dim=0)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(os.path.join(DATA_DIR, "Dev/dev.json")) as f: dev_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev_candidates.json")) as f: dev_c = json.load(f)
    with open(os.path.join(EVAL_DIR, "dev_truth.json")) as f: dev_truth = json.load(f)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print("Loading HarveyLM LORMA model from:", CKPT_PATH)
    state = torch.load(CKPT_PATH, map_location=device)
    model = UniversalLegalRetriever(arch_type="harveylm", adaptation="lorma", r=32, alpha=64).to(device)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)

    print("Encoding Dev split...")
    dev_c_ids, dev_c_embeds = encode_docs(model, dev_c, tokenizer, max_seq_len=768, batch_size=32, device=device)
    dev_q_ids, dev_q_embeds = encode_docs(model, dev_q, tokenizer, max_seq_len=768, batch_size=32, device=device)

    # Raw Cosine Similarity
    sim_raw = torch.matmul(dev_q_embeds, dev_c_embeds.T).numpy()

    def get_pred(sim_mat):
        top7_idx = torch.topk(torch.from_numpy(sim_mat), k=7, dim=-1).indices.numpy()
        pred = {}
        for i, qid in enumerate(dev_q_ids):
            pred[qid] = [dev_c_ids[idx] for idx in top7_idx[i]]
        return pred

    # Baseline Raw
    m_raw = compute_pcr_metrics(dev_truth, get_pred(sim_raw))
    print(f"\n[Raw Baseline] Dev Micro-F1@7: {m_raw['micro_f1_7']:.6f} | Prec: {m_raw['micro_precision_7']:.6f} | Rec: {m_raw['micro_recall_7']:.6f} | TP: {m_raw['tp']}")

    # Build Statutory Tversky Overlap Matrix
    t0 = time.time()
    s_stat = build_statutory_tversky_matrix(dev_q, dev_c, dev_q_ids, dev_c_ids)
    print(f"Statutory matrix built in {time.time() - t0:.2f}s | Non-zero entries: {np.count_nonzero(s_stat)}")

    # Test CSLS Anti-Hubness alone
    for beta in [0.15, 0.25, 0.35, 0.45]:
        sim_csls = apply_csls_anti_hubness(sim_raw, k_neighbors=10, beta_hub=beta)
        m_csls = compute_pcr_metrics(dev_truth, get_pred(sim_csls))
        print(f"[CSLS beta={beta}] Dev Micro-F1@7: {m_csls['micro_f1_7']:.6f} | TP: {m_csls['tp']}")

    # Normalize Statutory matrix
    s_stat_z = (s_stat - np.mean(s_stat, axis=1, keepdims=True)) / (np.std(s_stat, axis=1, keepdims=True) + 1e-6)

    # Grid search blend: CSLS + Statutory Prior
    best_f1, best_w, best_tp = 0.0, None, 0
    sim_csls = apply_csls_anti_hubness(sim_raw, k_neighbors=10, beta_hub=0.35)

    for w_stat in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        blended = sim_csls + w_stat * s_stat_z
        m_blend = compute_pcr_metrics(dev_truth, get_pred(blended))
        if m_blend['micro_f1_7'] > best_f1:
            best_f1 = m_blend['micro_f1_7']
            best_w = w_stat
            best_tp = m_blend['tp']
        print(f"[Blend w_stat={w_stat:.2f}] Dev Micro-F1@7: {m_blend['micro_f1_7']:.6f} | TP: {m_blend['tp']}")

    print(f"\n>>> BEST STAGE 1 BLEND: Dev Micro-F1@7 = {best_f1:.6f} (TP={best_tp}, gained +{best_tp - m_raw['tp']} hits over baseline!) with w_stat={best_w}")

if __name__ == "__main__":
    main()
