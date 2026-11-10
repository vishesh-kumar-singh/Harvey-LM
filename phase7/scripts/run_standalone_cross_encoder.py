#!/usr/bin/env python3
"""
Phase 7: Standalone Causal Cross-Encoder on HarveyLM (ALiBi Transformer).
A pure, single-model architecture eliminating dual-tower vector bottlenecks.

Architecture Highlights:
1. Backbone: Custom Phase 1 HarveyLM (16 layers, 210M params, ALiBi attention, RMSNorm, SwiGLU).
2. Adaptation: LORMA (Truncated SVD Low-Rank Multi-Head Adaptation, r=32, alpha=64).
3. Cross-Attention Geometry:
   Input: [Candidate Salient Tokens (512)] + [SEP] + [Query Salient Tokens (512)] (Total 1024 tokens).
   Under HarveyLM's causal lower-triangular attention + ALiBi biases:
   Every single Query token performs multi-head cross-attention over ALL Candidate tokens
   across all 16 transformer layers.
4. Relevance Projection Head:
   Extracts joint representation [ MeanPool(Q|C) || EOS_State ] in R^(2048) and projects
   via a 2-layer SwiGLU relevance network to a scalar matching logit s(Q, C).
5. Objective:
   Groupwise Plackett-Luce Listwise Cross-Entropy + Pairwise Margin RankNet Loss
   over groups of 1 positive + 5 mined hard negatives.
6. Cascade:
   Retrieves Top-35 candidate cases per query using fine-tuned HarveyLM LORMA,
   then executes full cross-attention token-level reranking to emit the Top-7 unique precedents.
"""

import os
import sys
import json
import time
import math
import random
import argparse
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase7.src.evaluation.eval_utils import compute_pcr_metrics, export_pcr_submission

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
EVAL_DIR = os.path.join(ROOT_CODE, "phase7/data/eval_truth")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
EVAL_SCRIPT_PATH = os.path.join(ROOT_CODE, "Evaluation Script/evaluate_phase7.py")
HARVEY_BASE_CKPT = os.path.join(ROOT_CODE, "phase1/models/final_architectures/2_HarveyLM/epoch_99")
BIENCODER_CKPT = os.path.join(ROOT_CODE, "phase7/models/ensemble/peak_solution/best_model.pt")
SUB_DIR = os.path.join(ROOT_CODE, "phase7/submissions/harveylm/cross_encoder")
MODEL_DIR = os.path.join(ROOT_CODE, "phase7/models/harveylm/cross_encoder")

TARGET_MODULES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]

# Regex and Saliency Markers
import re
STATUTE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|article|art\.?|order|rule|clause)\s*"
    r"(\d+[a-zA-Z]*(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)"
    r"(?:\s+of\s+the\s+([a-zA-Z\s]{3,35}?act|[a-zA-Z\s]{3,25}?code|constitution))?",
    re.IGNORECASE,
)
RATIO_MARKERS = {
    "ratio", "decidendi", "settled", "principle", "constitution bench",
    "relying", "distinguished", "overruled", "interpreting", "mandates",
    "precedent", "doctrine", "jurisprudence", "impermissible", "ultra vires", "held that"
}
BOILERPLATE_MARKERS = {
    "counsel", "learned", "appearing", "adjourned", "affidavit",
    "vakalatnama", "registry", "listed", "coram", "ia no"
}


def extract_salient_tokens_512(token_ids: List[int], raw_text: str, tokenizer, max_len=512) -> List[int]:
    """Extracts 512 tokens: 192 Head + 192 Salient Ratio/Statutory + 128 Tail."""
    total_len = len(token_ids)
    if total_len <= max_len:
        return token_ids

    head_len = 192
    tail_len = 128
    middle_budget = max_len - head_len - tail_len
    window_size = 64

    head_tokens = token_ids[:head_len]
    tail_tokens = token_ids[total_len - tail_len:]
    middle_pool = token_ids[head_len: total_len - tail_len]

    num_windows = len(middle_pool) // window_size
    if num_windows == 0:
        return (head_tokens + middle_pool + tail_tokens)[:max_len]

    target_windows = middle_budget // window_size
    if num_windows <= target_windows:
        return (head_tokens + middle_pool[:middle_budget] + tail_tokens)[:max_len]

    window_scores = []
    for w_idx in range(num_windows):
        chunk_ids = middle_pool[w_idx * window_size : (w_idx + 1) * window_size]
        chunk_str = tokenizer.decode(chunk_ids).lower()

        stat_count = len(STATUTE_PATTERN.findall(chunk_str))
        ratio_count = sum(1 for m in RATIO_MARKERS if m in chunk_str)
        boiler_count = sum(1 for b in BOILERPLATE_MARKERS if b in chunk_str)

        rel_pos = w_idx / max(1, num_windows - 1)
        pos_bonus = 0.5 if 0.35 <= rel_pos <= 0.85 else 0.0

        score = 3.0 * stat_count + 2.0 * ratio_count - 1.5 * boiler_count + pos_bonus
        window_scores.append((score, w_idx))

    top_indices = sorted([idx for _, idx in sorted(window_scores, reverse=True)[:target_windows]])
    salient_middle = []
    for idx in top_indices:
        salient_middle.extend(middle_pool[idx * window_size : (idx + 1) * window_size])

    return head_tokens + salient_middle[:middle_budget] + tail_tokens


class StandaloneHarveyCrossEncoder(nn.Module):
    """
    Standalone Causal Cross-Encoder wrapping HarveyLM with LORMA and a SwiGLU relevance head.
    Input layout: [Candidate (512)] + [SEP] + [Query (512)] (Total sequence <= 1024).
    """
    def __init__(self, ckpt_dir: str = HARVEY_BASE_CKPT, r: int = 32, alpha: int = 64, dropout: float = 0.1):
        super().__init__()
        self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
        self.backbone = HarveyLM(self.config)
        self.dim = self.config.dim

        # Load pretrained weights
        from safetensors.torch import load_file
        st_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(st_path):
            st = load_file(st_path)
            self.backbone.load_state_dict(st, strict=False)
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(st, strict=False)

        # Inject LORMA
        apply_lorma_to_model(self.backbone, r=r, lorma_alpha=alpha, target_modules=TARGET_MODULES)
        mark_only_lorma_as_trainable(self.backbone)

        # 2-layer SwiGLU Relevance Head over [MeanPool(Q|C) || EOS_State] in R^(2*dim)
        self.w_gate = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.w_up = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.norm = nn.RMSNorm(self.dim)
        self.dropout = nn.Dropout(dropout)
        self.score_proj = nn.Linear(self.dim, 1, bias=False)
        nn.init.normal_(self.score_proj.weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        query_mask: torch.Tensor,      # [B, L] 1.0 on query tokens, 0.0 on candidate/pad
        attention_mask: torch.Tensor,  # [B, L] 1.0 on valid tokens, 0.0 on pad
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        h = self.backbone.tok_embeddings(input_ids)

        alibi = -torch.abs(self.backbone.alibi[:, :seqlen, :seqlen].to(device=h.device, dtype=h.dtype))
        # Mask padded tokens
        mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()

        for layer in self.backbone.layers:
            h, _ = layer(h, None, mask, alibi, None, False)
        h = self.backbone.norm(h)  # [B, L, D]

        # 1. Mean pool over Query tokens (each attended to the entire Candidate prefix)
        q_weights = query_mask.unsqueeze(-1).to(h.dtype)
        h_q = (h * q_weights).sum(dim=1) / (q_weights.sum(dim=1) + 1e-6)

        # 2. Extract final valid token representation
        lengths = attention_mask.sum(dim=1).long() - 1
        h_eos = h[torch.arange(bsz, device=h.device), lengths]

        # 3. SwiGLU relevance projection
        joint = torch.cat([h_q, h_eos], dim=-1)  # [B, 2*D]
        gate = F.silu(self.w_gate(joint))
        up = self.w_up(joint)
        act = self.dropout(self.norm(gate * up))
        logits = self.score_proj(act).squeeze(-1)  # [B]
        return logits


class CrossEncoderPairDataset(Dataset):
    """
    Dataset of (Query, Candidate) pairs with tokenized concatenation:
    [Candidate_Tokens (512)] + [SEP] + [Query_Tokens (512)]
    """
    def __init__(self, pairs: List[Tuple[str, str, int]], q_cache: Dict, c_cache: Dict, max_len=1024, sep_id=2):
        self.pairs = pairs
        self.q_cache = q_cache
        self.c_cache = c_cache
        self.max_len = max_len
        self.sep_id = sep_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, cid, label = self.pairs[idx]
        q_tokens = self.q_cache[qid]
        c_tokens = self.c_cache[cid]

        # Build joint sequence: Cand + [SEP] + Query
        seq = c_tokens + [self.sep_id] + q_tokens
        seq = seq[:self.max_len]

        cand_len = min(len(c_tokens), self.max_len)
        total_len = len(seq)

        pad_len = self.max_len - total_len
        ids = torch.tensor(seq + [0] * pad_len, dtype=torch.long)

        attn_mask = torch.zeros(self.max_len, dtype=torch.float32)
        attn_mask[:total_len] = 1.0

        q_mask = torch.zeros(self.max_len, dtype=torch.float32)
        if total_len > (cand_len + 1):
            q_mask[cand_len + 1 : total_len] = 1.0

        return {
            "input_ids": ids,
            "attention_mask": attn_mask,
            "query_mask": q_mask,
            "label": torch.tensor(label, dtype=torch.float32),
            "qid": qid,
            "cid": cid,
        }


def collate_pairs(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "query_mask": torch.stack([item["query_mask"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "qids": [item["qid"] for item in batch],
        "cids": [item["cid"] for item in batch],
    }


def run_standalone_cross_encoder():
    print("==========================================================================")
    print("PHASE 7: STANDALONE CAUSAL CROSS-ENCODER ON HARVEYLM")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    print("\n--- 1. Loading Datasets & Ground Truth ---")
    with open(os.path.join(DATA_DIR, "Train/train.json")) as f: train_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Train/train_candidates.json")) as f: train_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev.json")) as f: dev_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev_candidates.json")) as f: dev_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test.json")) as f: test_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test_candidates.json")) as f: test_c = json.load(f)

    with open(os.path.join(EVAL_DIR, "dev_truth.json")) as f: dev_truth = json.load(f)
    with open(os.path.join(EVAL_DIR, "dev_candidate_ids.json")) as f: dev_cand_ids = json.load(f)
    with open(os.path.join(EVAL_DIR, "test_candidate_ids.json")) as f: test_cand_ids = json.load(f)

    # 2. Stage-1 Candidate Screening: Retrieve Top-35 Pools
    print("\n--- 2. Screening Top-35 Candidates per Query using Fine-Tuned Bi-Encoder ---")
    from run_peak_solution import UniversalLegalRetriever, encode_tri_docs, apply_csls_anti_hubness
    bi_model = UniversalLegalRetriever(arch_type="harveylm", adaptation="lorma", r=32, alpha=64).to(device)
    bi_st = torch.load(BIENCODER_CKPT, map_location=device)
    bi_model.load_state_dict(bi_st.get("model_state_dict", bi_st), strict=False)
    bi_model.eval()

    print("Encoding Dev split for candidate screening...")
    dev_c_ids, dev_c_emb = encode_tri_docs(bi_model, dev_c, tokenizer, max_len=1024, batch_size=32, device=device)
    dev_q_ids, dev_q_emb = encode_tri_docs(bi_model, dev_q, tokenizer, max_len=1024, batch_size=32, device=device)
    sim_dev = torch.matmul(dev_q_emb, dev_c_emb.T).numpy()
    z_dev = apply_csls_anti_hubness(sim_dev, beta_hub=0.35)

    print("Encoding Test split for candidate screening...")
    test_c_ids, test_c_emb = encode_tri_docs(bi_model, test_c, tokenizer, max_len=1024, batch_size=32, device=device)
    test_q_ids, test_q_emb = encode_tri_docs(bi_model, test_q, tokenizer, max_len=1024, batch_size=32, device=device)
    sim_test = torch.matmul(test_q_emb, test_c_emb.T).numpy()
    z_test = apply_csls_anti_hubness(sim_test, beta_hub=0.35)

    # Extract Top-35 candidate pools
    TOP_K_SCREEN = 35
    top35_dev_indices = np.argpartition(-z_dev, kth=TOP_K_SCREEN, axis=1)[:, :TOP_K_SCREEN]
    dev_pool_map = {}
    dev_recall_at_35 = 0
    total_dev_gold = sum(len(dev_truth[qid]) for qid in dev_q_ids)

    for i, qid in enumerate(dev_q_ids):
        pool = [dev_c_ids[idx] for idx in top35_dev_indices[i]]
        dev_pool_map[qid] = pool
        gold_set = set(dev_truth[qid])
        dev_recall_at_35 += len(gold_set.intersection(set(pool)))

    print(f"Top-{TOP_K_SCREEN} Dev Recall Ceiling: {dev_recall_at_35}/{total_dev_gold} ({dev_recall_at_35/total_dev_gold*100:.2f}%)")

    top35_test_indices = np.argpartition(-z_test, kth=TOP_K_SCREEN, axis=1)[:, :TOP_K_SCREEN]
    test_pool_map = {}
    for i, qid in enumerate(test_q_ids):
        test_pool_map[qid] = [test_c_ids[idx] for idx in top35_test_indices[i]]

    del bi_model, bi_st, dev_c_emb, dev_q_emb, test_c_emb, test_q_emb
    torch.cuda.empty_cache()

    # 3. Pre-tokenize Salient 512-Token Chunks
    print("\n--- 3. Pre-tokenizing Salient Contexts (512 Tokens each) ---")
    t0 = time.time()
    q_cache = {}
    c_cache = {}

    all_q_dict = {**train_q, **dev_q, **test_q}
    for qid, qdata in all_q_dict.items():
        txt = qdata.get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        raw_ids = tokenizer.encode(txt, add_special_tokens=False).ids
        q_cache[qid] = extract_salient_tokens_512(raw_ids, txt, tokenizer, max_len=512)

    all_c_dict = {**train_c, **dev_c, **test_c}
    for cid, cdata in all_c_dict.items():
        txt = cdata.get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        raw_ids = tokenizer.encode(txt, add_special_tokens=False).ids
        c_cache[cid] = extract_salient_tokens_512(raw_ids, txt, tokenizer, max_len=512)
    print(f"Tokenized {len(q_cache)} queries and {len(c_cache)} candidates in {time.time() - t0:.2f}s")

    # 4. Build Training Groups: 1 Positive + 5 Mined Hard Negatives
    print("\n--- 4. Assembling Listwise Training Groups for Cross-Encoder ---")
    with open(os.path.join(ROOT_CODE, "phase7/data/train_hard_negatives.json")) as f:
        hard_negs_data = json.load(f)
    stat_negs = hard_negs_data["stat_hard_neg_map"]
    dense_negs = hard_negs_data["dense_hard_neg_map"]

    train_groups = []  # List of [ (qid, pos_cid), (qid, neg_1), ... (qid, neg_5) ]
    train_q_valid = [q for q in train_q.keys() if len(train_q[q].get("relevant_candidates", [])) > 0]

    for qid in train_q_valid:
        golds = train_q[qid]["relevant_candidates"]
        pos_cid = random.choice(golds)
        gold_set = set(golds)

        stat_pool = [c for c in stat_negs.get(qid, []) if c not in gold_set]
        dense_pool = [c for c in dense_negs.get(qid, []) if c not in gold_set]

        negs = []
        if len(dense_pool) >= 3:
            negs.extend(random.sample(dense_pool[:20], 3))
        if len(stat_pool) >= 2:
            negs.extend(random.sample(stat_pool[:20], 2))
        while len(negs) < 5:
            rc = random.choice(list(train_c.keys()))
            if rc not in gold_set and rc not in negs:
                negs.append(rc)

        group_pairs = [(qid, pos_cid, 1)] + [(qid, n, 0) for n in negs[:5]]
        train_groups.append(group_pairs)

    print(f"Constructed {len(train_groups)} listwise training groups (Group size G=6)")

    # Flatten for dataloader
    flat_train_pairs = [p for g in train_groups for p in g]
    train_dataset = CrossEncoderPairDataset(flat_train_pairs, q_cache, c_cache, max_len=1024)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, collate_fn=collate_pairs, num_workers=2)

    # 5. Initialize Standalone HarveyLM Cross-Encoder
    print("\n--- 5. Initializing Standalone HarveyLM Cross-Encoder ---")
    ce_model = StandaloneHarveyCrossEncoder(ckpt_dir=HARVEY_BASE_CKPT, r=32, alpha=64, dropout=0.1).to(device)

    # Warm-start LORMA weights from our best Bi-Encoder to accelerate convergence
    if os.path.exists(BIENCODER_CKPT):
        print(f"Warm-starting LORMA adapters from {BIENCODER_CKPT}...")
        st = torch.load(BIENCODER_CKPT, map_location=device)
        ce_model.backbone.load_state_dict(st.get("model_state_dict", st), strict=False)

    trainable_params = sum(p.numel() for p in ce_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in ce_model.parameters())
    print(f"Cross-Encoder Trainable Params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in ce_model.parameters() if p.requires_grad],
        lr=1.0e-4,
        weight_decay=0.01
    )
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # 6. Fine-Tuning Loop
    print("\n--- 6. Beginning Cross-Encoder Fine-Tuning ---")
    ce_model.train()
    t_train_start = time.time()
    G = 6  # Group size

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            query_mask = batch["query_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = ce_model(input_ids, query_mask, attention_mask)  # [B]
                B_groups = logits.size(0) // G
                grouped_logits = logits[:B_groups * G].view(B_groups, G)

                # Column 0 is the positive candidate
                target_zeros = torch.zeros(B_groups, dtype=torch.long, device=device)
                loss_ce = F.cross_entropy(grouped_logits, target_zeros)

                # Pairwise margin ranknet penalty
                pos_l = grouped_logits[:, 0:1]
                neg_l = grouped_logits[:, 1:]
                loss_pairwise = F.softplus(-(pos_l - neg_l)).mean()

                loss = 0.6 * loss_ce + 0.4 * loss_pairwise

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs} | Step {step:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(train_loader)
        print(f"--> Epoch {epoch} completed in {time.time() - t_epoch_start:.1f}s | Avg Loss: {avg_loss:.4f}")

    print(f"Cross-Encoder Training Finished in {time.time() - t_train_start:.1f}s")
    ce_ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    torch.save({"model_state_dict": ce_model.state_dict()}, ce_ckpt_path)
    print(f"Saved Standalone Cross-Encoder to {ce_ckpt_path}")

    # 7. Reranking Inference Function
    @torch.no_grad()
    def rerank_candidate_pools(pool_map: Dict[str, List[str]], batch_size=32) -> Dict[str, List[str]]:
        ce_model.eval()
        pairs = []
        for qid, cands in pool_map.items():
            for cid in cands:
                pairs.append((qid, cid, 0))

        ds = CrossEncoderPairDataset(pairs, q_cache, c_cache, max_len=1024)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs, num_workers=2)

        scores_by_query = {qid: {} for qid in pool_map.keys()}
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            query_mask = batch["query_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = ce_model(input_ids, query_mask, attention_mask)
            logits_np = logits.float().cpu().numpy()

            for qid, cid, s in zip(batch["qids"], batch["cids"], logits_np):
                scores_by_query[qid][cid] = s

        # Extract strictly ordered Top-7
        final_preds = {}
        for qid, c_scores in scores_by_query.items():
            ranked = sorted(c_scores.items(), key=lambda x: x[1], reverse=True)
            top7 = [c for c, _ in ranked[:7]]
            final_preds[qid] = top7
        return final_preds

    # 8. Dev Reranking & Evaluation
    print("\n--- 8. Executing Dev Split Reranking ---")
    t0 = time.time()
    dev_final_preds = rerank_candidate_pools(dev_pool_map, batch_size=32)
    print(f"Dev Reranking finished in {time.time() - t0:.2f}s")

    dev_metrics = compute_pcr_metrics(dev_truth, dev_final_preds)
    print("==========================================================================")
    print("STANDALONE HARVEYLM CROSS-ENCODER DEV PERFORMANCE:")
    print(f"Dev Micro-F1@7:        {dev_metrics['micro_f1_7']:.6f}")
    print(f"Dev Micro-Precision@7: {dev_metrics['micro_precision_7']:.6f}")
    print(f"Dev Micro-Recall@7:    {dev_metrics['micro_recall_7']:.6f}")
    print(f"True Positives: {dev_metrics['tp']} (Baseline: 252 | Gain: +{dev_metrics['tp'] - 252})")
    print(f"False Positives: {dev_metrics['fp']} | False Negatives: {dev_metrics['fn']}")
    print("==========================================================================")

    dev_pred_path = os.path.join(SUB_DIR, "dev_prediction.json")
    with open(dev_pred_path, "w", encoding="utf-8") as f:
        json.dump(dev_final_preds, f, indent=2)

    # 9. Test Reranking & Packaging Official Submission
    print("\n--- 9. Executing Test Split Reranking & Packaging Submission ---")
    t0 = time.time()
    test_final_preds = rerank_candidate_pools(test_pool_map, batch_size=32)
    print(f"Test Reranking finished in {time.time() - t0:.2f}s")

    sub_zip_path = os.path.join(SUB_DIR, "submission.zip")
    export_pcr_submission(
        pred_dict=test_final_preds,
        output_zip_path=sub_zip_path,
        allowed_candidates=set(test_cand_ids),
        expected_queries=set(test_q.keys())
    )

    # 10. Official Evaluation Script Check
    cmd = [
        sys.executable,
        EVAL_SCRIPT_PATH,
        "--truth", os.path.join(EVAL_DIR, "dev_truth.json"),
        "--candidate-ids", os.path.join(EVAL_DIR, "dev_candidate_ids.json"),
        "--prediction", dev_pred_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n[Official Evaluator Verification on Dev]:")
    print(res.stdout.strip())

    # 11. Log to Results Tracker
    tracker_path = os.path.join(ROOT_CODE, "phase7/results.txt")
    log_line = f"STANDALONE_CE    HARVEYLM_LORMA       {dev_metrics['micro_f1_7']:<14.6f} {dev_metrics['micro_precision_7']:<11.6f} {dev_metrics['micro_recall_7']:<12.6f} [READY]        {sub_zip_path}\n"
    with open(tracker_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\nLogged results to {tracker_path}")

    print("\n==========================================================================")
    print("STANDALONE CROSS-ENCODER COMPLETE & VERIFIED!")
    print(f"Official Submission Zip: {sub_zip_path}")
    print("==========================================================================")


if __name__ == "__main__":
    run_standalone_cross_encoder()
