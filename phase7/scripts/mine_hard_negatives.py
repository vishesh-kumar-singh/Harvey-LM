#!/usr/bin/env python3
"""
Mine Two-Tier Hard Negatives for Phase 7 Training:
1. Statutory-Lexical Hard Negatives (top non-relevant candidates with highest statutory clause overlap)
2. Self-Adversarial Dense Hard Negatives (top non-relevant candidates from HarveyLM LORMA checkpoint)
"""
import os
import sys
import json
import time
import math
import re
from collections import Counter
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

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
CKPT_PATH = os.path.join(ROOT_CODE, "phase7/models/harveylm/lorma/best_model.pt")
OUT_PATH = os.path.join(ROOT_CODE, "phase7/data/train_hard_negatives.json")

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

    print("Loading training data...")
    with open(os.path.join(DATA_DIR, "Train/train.json")) as f:
        train_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Train/train_candidates.json")) as f:
        train_c = json.load(f)

    cand_ids = list(train_c.keys())
    query_ids = list(train_q.keys())

    # 1. Statutory Signatures Overlap
    print("Extracting statutory signatures for 4,320 candidates and 827 queries...")
    t0 = time.time()
    cand_sigs = {}
    for cid in cand_ids:
        txt = train_c[cid].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        cand_sigs[cid] = extract_statutory_signatures(txt)

    query_sigs = {}
    for qid in query_ids:
        txt = train_q[qid].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        query_sigs[qid] = extract_statutory_signatures(txt)

    df = Counter()
    for sigs in cand_sigs.values():
        for s in sigs:
            df[s] += 1
    num_c = len(cand_ids)
    idf = {s: math.log(1.0 + num_c / (1.0 + count)) for s, count in df.items()}
    print(f"Statutory signatures extracted in {time.time() - t0:.2f}s | Vocabulary: {len(idf)} statutory features")

    # 2. Dense Embeddings from HarveyLM LORMA
    print(f"Loading HarveyLM LORMA from {CKPT_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    state = torch.load(CKPT_PATH, map_location=device)
    model = UniversalLegalRetriever(arch_type="harveylm", adaptation="lorma", r=32, alpha=64).to(device)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)

    print("Encoding 4,320 training candidates...")
    t0 = time.time()
    c_ids_enc, c_embeds = encode_docs(model, train_c, tokenizer, max_seq_len=768, batch_size=32, device=device)
    print(f"Encoded candidates in {time.time() - t0:.2f}s")

    print("Encoding 827 training queries...")
    t0 = time.time()
    q_ids_enc, q_embeds = encode_docs(model, train_q, tokenizer, max_seq_len=768, batch_size=32, device=device)
    print(f"Encoded queries in {time.time() - t0:.2f}s")

    print("Computing cosine similarity matrix [827 x 4,320]...")
    sim_matrix = torch.matmul(q_embeds, c_embeds.T).numpy()
    cand_id_to_idx = {cid: idx for idx, cid in enumerate(c_ids_enc)}

    print("Mining hard negatives for each training query...")
    stat_hard_neg_map = {}
    dense_hard_neg_map = {}

    for i, qid in enumerate(query_ids):
        gold_set = set(train_q[qid].get("relevant_candidates", []))
        q_set = query_sigs[qid]

        # 1. Statutory overlap scoring
        stat_scores = []
        if q_set:
            q_weights = {s: idf.get(s, 1.0) for s in q_set}
            q_tot = sum(q_weights.values())
            for cid in cand_ids:
                if cid in gold_set:
                    continue
                c_set = cand_sigs[cid]
                inter = q_set.intersection(c_set)
                if not inter:
                    continue
                inter_w = sum(q_weights[s] for s in inter)
                score = inter_w / (inter_w + 0.7 * (q_tot - inter_w) + 1e-6)
                stat_scores.append((score, cid))
            stat_scores.sort(key=lambda x: x[0], reverse=True)
            stat_hard_neg_map[qid] = [cid for _, cid in stat_scores[:25]]
        else:
            stat_hard_neg_map[qid] = []

        # 2. Dense Hard Negatives (top non-gold candidates with sim <= 0.88 to avoid unlabelled true precedents)
        row_sims = sim_matrix[i]
        top_dense_indices = np.argsort(-row_sims)[:60]
        dense_negs = []
        for idx in top_dense_indices:
            cid = c_ids_enc[idx]
            if cid in gold_set:
                continue
            if row_sims[idx] > 0.88:
                continue  # safety filter
            dense_negs.append(cid)
            if len(dense_negs) >= 30:
                break
        dense_hard_neg_map[qid] = dense_negs

    out_data = {
        "stat_hard_neg_map": stat_hard_neg_map,
        "dense_hard_neg_map": dense_hard_neg_map,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved mined hard negatives to {OUT_PATH}")

if __name__ == "__main__":
    main()
