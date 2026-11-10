#!/usr/bin/env python3
"""
Explore Phase 7 Prior Case Retrieval (PCR) Dataset.
Computes comprehensive empirical statistics:
- Split sizes (queries, candidate pools, positive relevance pairs)
- Relevance density and label distributions (mean, median, IQR, max)
- Candidate citation hubness (in-degree distributions)
- Length profiles (words and BPE tokens for queries and candidates)
- Lexical overlap (Jaccard similarity of positive vs. negative pairs)
- Legal terminology and citation pattern analysis
"""

import os
import sys
import json
import numpy as np
from tokenizers import Tokenizer

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/Phase_7_PCR"))
TOKENIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../phase1/models/tokenizer/legal_bpe_tokenizer.json"))

def load_json(rel_path):
    path = os.path.join(DATA_DIR, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_analysis():
    print("==========================================================================")
    print("PHASE 7 PRIOR CASE RETRIEVAL (PCR) EMPIRICAL DATASET ANALYSIS")
    print("==========================================================================")
    
    train_q = load_json("Train/train.json")
    train_c = load_json("Train/train_candidates.json")
    dev_q = load_json("Dev/dev.json")
    dev_c = load_json("Dev/dev_candidates.json")
    test_q = load_json("Test/test.json")
    test_c = load_json("Test/test_candidates.json")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    print("\n--- 1. SPLIT SUMMARY & ISOLATION CHECK ---")
    print(f"Train: {len(train_q):>5} queries | {len(train_c):>6} candidates in pool")
    print(f"Dev:   {len(dev_q):>5} queries | {len(dev_c):>6} candidates in pool")
    print(f"Test:  {len(test_q):>5} queries | {len(test_c):>6} candidates in pool")
    
    # Check candidate pool overlaps
    s_tr_c = set(train_c.keys())
    s_dv_c = set(dev_c.keys())
    s_ts_c = set(test_c.keys())
    print(f"Train & Dev candidate ID overlap: {len(s_tr_c & s_dv_c)}")
    print(f"Dev & Test candidate ID overlap:  {len(s_dv_c & s_ts_c)}")
    print(f"Train & Test candidate ID overlap: {len(s_tr_c & s_ts_c)}")

    # Check query ID overlaps
    s_tr_q = set(train_q.keys())
    s_dv_q = set(dev_q.keys())
    s_ts_q = set(test_q.keys())
    print(f"Train & Dev query ID overlap: {len(s_tr_q & s_dv_q)}")
    print(f"Dev & Test query ID overlap:  {len(s_dv_q & s_ts_q)}")
    print(f"Train & Test query ID overlap: {len(s_tr_q & s_ts_q)}")

    print("\n--- 2. RELEVANCE LABEL DENSITY (K=7 EVALUATION ALIGNMENT) ---")
    train_rel_counts = [len(v.get("relevant_candidates", [])) for v in train_q.values()]
    dev_rel_counts = [len(v.get("relevant_candidates", [])) for v in dev_q.values()]
    total_train_pos = sum(train_rel_counts)
    total_dev_pos = sum(dev_rel_counts)

    print(f"Train Total Positive Pairs: {total_train_pos:,}")
    print(f"  Mean: {np.mean(train_rel_counts):.2f} | Std: {np.std(train_rel_counts):.2f} | Median: {np.median(train_rel_counts):.1f}")
    print(f"  Min: {np.min(train_rel_counts)} | Max: {np.max(train_rel_counts)} | IQR: [{np.percentile(train_rel_counts, 25):.1f}, {np.percentile(train_rel_counts, 75):.1f}]")
    print(f"  90th %ile: {np.percentile(train_rel_counts, 90):.1f} | 95th %ile: {np.percentile(train_rel_counts, 95):.1f}")

    print(f"\nDev Total Positive Pairs: {total_dev_pos:,}")
    print(f"  Mean: {np.mean(dev_rel_counts):.2f} | Std: {np.std(dev_rel_counts):.2f} | Median: {np.median(dev_rel_counts):.1f}")
    print(f"  Min: {np.min(dev_rel_counts)} | Max: {np.max(dev_rel_counts)} | IQR: [{np.percentile(dev_rel_counts, 25):.1f}, {np.percentile(dev_rel_counts, 75):.1f}]")
    print(f"  90th %ile: {np.percentile(dev_rel_counts, 90):.1f} | 95th %ile: {np.percentile(dev_rel_counts, 95):.1f}")

    # Queries with <= 7 vs > 7 relevant candidates
    tr_le7 = sum(1 for c in train_rel_counts if c <= 7)
    dv_le7 = sum(1 for c in dev_rel_counts if c <= 7)
    print(f"Train queries with <= 7 relevant candidates: {tr_le7} ({tr_le7/len(train_q)*100:.1f}%)")
    print(f"Dev queries with <= 7 relevant candidates:   {dv_le7} ({dv_le7/len(dev_q)*100:.1f}%)")

    print("\n--- 3. CANDIDATE HUBNESS / CITATION DEGREE DISTRIBUTION ---")
    tr_cand_cit = {}
    for v in train_q.values():
        for c in v.get("relevant_candidates", []):
            tr_cand_cit[c] = tr_cand_cit.get(c, 0) + 1
    cit_vals = list(tr_cand_cit.values())
    unseen_cands = len(train_c) - len(tr_cand_cit)
    print(f"Train candidates cited >= 1 time: {len(tr_cand_cit)} / {len(train_c)} ({len(tr_cand_cit)/len(train_c)*100:.1f}%)")
    print(f"Train candidates never cited in training: {unseen_cands} ({unseen_cands/len(train_c)*100:.1f}%)")
    print(f"Citation count per cited candidate - Mean: {np.mean(cit_vals):.2f}, Median: {np.median(cit_vals):.1f}, Max: {np.max(cit_vals)}")
    print(f"Top 5 most cited candidates in Train: {sorted(tr_cand_cit.items(), key=lambda x: x[1], reverse=True)[:5]}")

    print("\n--- 4. SEQUENCE LENGTH DISTRIBUTION (WORDS & BPE SUBWORDS) ---")
    def get_lengths(doc_dict):
        words = []
        tokens = []
        for v in doc_dict.values():
            txt = v.get("text", "")
            if isinstance(txt, list):
                txt = " ".join(txt)
            w = len(txt.split())
            tok_len = len(tokenizer.encode(txt, add_special_tokens=False).ids)
            words.append(w)
            tokens.append(tok_len)
        return words, tokens

    tr_q_words, tr_q_tokens = get_lengths(train_q)
    tr_c_words, tr_c_tokens = get_lengths(train_c)
    dv_q_words, dv_q_tokens = get_lengths(dev_q)
    dv_c_words, dv_c_tokens = get_lengths(dev_c)
    ts_q_words, ts_q_tokens = get_lengths(test_q)
    ts_c_words, ts_c_tokens = get_lengths(test_c)

    print(f"Train Queries (Words):     Mean: {np.mean(tr_q_words):.1f} | Med: {np.median(tr_q_words):.1f} | 90%: {np.percentile(tr_q_words, 90):.1f} | Max: {np.max(tr_q_words)}")
    print(f"Train Queries (Tokens):    Mean: {np.mean(tr_q_tokens):.1f} | Med: {np.median(tr_q_tokens):.1f} | 90%: {np.percentile(tr_q_tokens, 90):.1f} | Max: {np.max(tr_q_tokens)}")
    print(f"Train Candidates (Words):  Mean: {np.mean(tr_c_words):.1f} | Med: {np.median(tr_c_words):.1f} | 90%: {np.percentile(tr_c_words, 90):.1f} | Max: {np.max(tr_c_words)}")
    print(f"Train Candidates (Tokens): Mean: {np.mean(tr_c_tokens):.1f} | Med: {np.median(tr_c_tokens):.1f} | 90%: {np.percentile(tr_c_tokens, 90):.1f} | Max: {np.max(tr_c_tokens)}")
    
    print(f"\nDev Queries (Tokens):      Mean: {np.mean(dv_q_tokens):.1f} | Med: {np.median(dv_q_tokens):.1f} | 90%: {np.percentile(dv_q_tokens, 90):.1f} | Max: {np.max(dv_q_tokens)}")
    print(f"Dev Candidates (Tokens):   Mean: {np.mean(dv_c_tokens):.1f} | Med: {np.median(dv_c_tokens):.1f} | 90%: {np.percentile(dv_c_tokens, 90):.1f} | Max: {np.max(dv_c_tokens)}")

    print(f"\nTest Queries (Tokens):     Mean: {np.mean(ts_q_tokens):.1f} | Med: {np.median(ts_q_tokens):.1f} | 90%: {np.percentile(ts_q_tokens, 90):.1f} | Max: {np.max(ts_q_tokens)}")
    print(f"Test Candidates (Tokens):  Mean: {np.mean(ts_c_tokens):.1f} | Med: {np.median(ts_c_tokens):.1f} | 90%: {np.percentile(ts_c_tokens, 90):.1f} | Max: {np.max(ts_c_tokens)}")

    print("\n--- 5. LEXICAL OVERLAP: POSITIVE VS. NEGATIVE PAIRS (JACCARD) ---")
    # Sample 1000 positive pairs and 1000 random negative pairs from Dev
    np.random.seed(42)
    dev_cand_keys = list(dev_c.keys())
    
    pos_jaccard_words = []
    pos_jaccard_tokens = []
    neg_jaccard_words = []
    neg_jaccard_tokens = []

    dev_c_token_sets = {}
    dev_c_word_sets = {}
    for cid, cval in dev_c.items():
        txt = cval.get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        w_set = set(txt.lower().split())
        t_set = set(tokenizer.encode(txt, add_special_tokens=False).ids)
        dev_c_word_sets[cid] = w_set
        dev_c_token_sets[cid] = t_set

    for qid, qval in dev_q.items():
        txt = qval.get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        qw_set = set(txt.lower().split())
        qt_set = set(tokenizer.encode(txt, add_special_tokens=False).ids)
        
        pos_cands = [c for c in qval.get("relevant_candidates", []) if c in dev_c]
        for pc in pos_cands:
            cw_set = dev_c_word_sets[pc]
            ct_set = dev_c_token_sets[pc]
            jw = len(qw_set & cw_set) / max(1, len(qw_set | cw_set))
            jt = len(qt_set & ct_set) / max(1, len(qt_set | ct_set))
            pos_jaccard_words.append(jw)
            pos_jaccard_tokens.append(jt)
            
            # Sample a negative candidate
            rand_c = np.random.choice(dev_cand_keys)
            while rand_c in pos_cands:
                rand_c = np.random.choice(dev_cand_keys)
            rw_set = dev_c_word_sets[rand_c]
            rt_set = dev_c_token_sets[rand_c]
            njw = len(qw_set & rw_set) / max(1, len(qw_set | rw_set))
            njt = len(qt_set & rt_set) / max(1, len(qt_set | rt_set))
            neg_jaccard_words.append(njw)
            neg_jaccard_tokens.append(njt)

    print(f"Positive Pairs Word Jaccard: Mean={np.mean(pos_jaccard_words):.4f}, Median={np.median(pos_jaccard_words):.4f}")
    print(f"Negative Pairs Word Jaccard: Mean={np.mean(neg_jaccard_words):.4f}, Median={np.median(neg_jaccard_words):.4f}")
    print(f"Word Jaccard Ratio (Pos / Neg): {np.mean(pos_jaccard_words) / max(1e-9, np.mean(neg_jaccard_words)):.2f}x")

    print(f"\nPositive Pairs Token Jaccard: Mean={np.mean(pos_jaccard_tokens):.4f}, Median={np.median(pos_jaccard_tokens):.4f}")
    print(f"Negative Pairs Token Jaccard: Mean={np.mean(neg_jaccard_tokens):.4f}, Median={np.median(neg_jaccard_tokens):.4f}")
    print(f"Token Jaccard Ratio (Pos / Neg): {np.mean(pos_jaccard_tokens) / max(1e-9, np.mean(neg_jaccard_tokens)):.2f}x")

    print("\n--- 6. CITATION & STATUTORY MARKER FREQUENCY ---")
    markers = ["section", "act", "court", "appellant", "respondent", "article", "order", "held", "judgment", "v."]
    print(f"{'Marker':<15} {'Train Q Freq':<15} {'Train C Freq':<15} {'Dev Q Freq':<15} {'Dev C Freq':<15}")
    for m in markers:
        def marker_freq(doc_dict, mark):
            cnt = sum(1 for v in doc_dict.values() if mark in (v.get("text") if isinstance(v.get("text"), str) else " ".join(v.get("text", []))).lower())
            return f"{cnt / len(doc_dict) * 100:.1f}%"
        print(f"{m:<15} {marker_freq(train_q, m):<15} {marker_freq(train_c, m):<15} {marker_freq(dev_q, m):<15} {marker_freq(dev_c, m):<15}")

    print("\n==========================================================================")
    print("EMPIRICAL DATASET ANALYSIS COMPLETE")
    print("==========================================================================")

if __name__ == "__main__":
    run_analysis()
