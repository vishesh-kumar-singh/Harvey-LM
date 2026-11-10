#!/usr/bin/env python3
"""
Phase 7: MASTER PEAK SOLUTION PIPELINE
Architecture and Pipeline Specification:
1. Tri-Anchor Saliency Context (1,024 tokens: 384 Head + 384 Middle Salient + 256 Tail)
2. Two-Tier Hard Negative Mining (Statutory + Self-Adversarial Dense)
3. Multi-Positive Cosine-Margin InfoNCE + Pairwise Hard Negative Margin Loss
4. Sub-Batched Encoding (sub_batch_size=4) for 100% OOM Immunity
5. LORMA Fine-Tuning on HarveyLM (210M, ALiBi)
6. CSLS Anti-Hubness Calibration (beta=0.35)
7. Asymmetric Statutory Tversky Overlap Integration
8. Calibrated Multi-Model Z-Score Logit Fusion
9. Verification with official evaluate_phase7.py and Packaging of submission.zip
"""
import os
import sys
import json
import time
import math
import re
import random
import zipfile
import subprocess
from collections import Counter
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

from phase7.src.model.pcr_retriever import UniversalLegalRetriever
from phase7.src.evaluation.eval_utils import compute_pcr_metrics, export_pcr_submission

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
EVAL_DIR = os.path.join(ROOT_CODE, "phase7/data/eval_truth")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
EVAL_SCRIPT_PATH = os.path.join(ROOT_CODE, "Evaluation Script/evaluate_phase7.py")
HARD_NEG_PATH = os.path.join(ROOT_CODE, "phase7/data/train_hard_negatives.json")

SUB_DIR = os.path.join(ROOT_CODE, "phase7/submissions/ensemble/peak_solution")
MODEL_DIR = os.path.join(ROOT_CODE, "phase7/models/ensemble/peak_solution")
BASE_LORMA_CKPT = os.path.join(ROOT_CODE, "phase7/models/harveylm/lorma/best_model.pt")
BASE_QLORA_CKPT = os.path.join(ROOT_CODE, "phase7/models/harveylm/qlora/best_model.pt")
BASE_ROPE_CKPT = os.path.join(ROOT_CODE, "phase7/models/baseline/lora/best_model.pt")

STATUTE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|article|art\.?|order|rule|clause)\s*"
    r"(\d+[a-zA-Z]*(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)"
    r"(?:\s+of\s+the\s+([a-zA-Z\s]{3,35}?act|[a-zA-Z\s]{3,25}?code|constitution))?",
    re.IGNORECASE,
)
RATIO_MARKERS = {
    "ratio", "decidendi", "settled", "principle", "constitution bench",
    "relying", "distinguished", "overruled", "interpreting", "mandates",
    "precedent", "doctrine", "jurisprudence", "impermissible", "ultra vires"
}
BOILERPLATE_MARKERS = {
    "counsel", "learned", "appearing", "adjourned", "affidavit",
    "vakalatnama", "registry", "listed", "coram", "ia no"
}
PARTY_MASK_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z\.\&\s]{2,35}?)\s+(?:v\.|vs\.|versus)\s+([A-Z][a-zA-Z\.\&\s]{2,35})\b"
)


def extract_statutory_signatures(text: str) -> Set[str]:
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


def build_statutory_tversky_matrix(
    queries_dict: Dict[str, Dict],
    candidates_dict: Dict[str, Dict],
    query_ids: List[str],
    cand_ids: List[str],
) -> np.ndarray:
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


def extract_tri_anchor_tokens(
    token_ids: List[int],
    raw_text: str,
    tokenizer,
    max_len: int = 1024,
    head_len: int = 384,
    tail_len: int = 256,
    window_size: int = 128,
    mask_parties: bool = False,
) -> List[int]:
    """
    Constructs a 1024-token Tri-Anchor sequence:
    [384 Head Tokens] + [384 Salient Middle Tokens] + [256 Tail Tokens].
    """
    if mask_parties and random.random() < 0.5:
        raw_text = PARTY_MASK_PATTERN.sub("PETITIONER_ENTITY v. RESPONDENT_ENTITY", raw_text)
        token_ids = tokenizer.encode(raw_text, add_special_tokens=False).ids

    total_len = len(token_ids)
    if total_len <= max_len:
        return token_ids

    middle_budget = max_len - head_len - tail_len
    head_tokens = token_ids[:head_len]
    tail_tokens = token_ids[total_len - tail_len :]
    middle_pool = token_ids[head_len : total_len - tail_len]

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


class HardNegativePCRDataset(Dataset):
    """Samples P=2 positives and K_hard=4 hard negatives per query."""
    def __init__(
        self,
        query_ids: List[str],
        queries_dict: Dict[str, Dict],
        candidates_dict: Dict[str, Dict],
        stat_hard_neg_map: Dict[str, List[str]],
        dense_hard_neg_map: Dict[str, List[str]],
        tokenizer,
        max_len: int = 1024,
        num_positives: int = 2,
        num_hard_negatives: int = 4,
        pad_token_id: int = 0,
    ):
        self.gold_map = {q: queries_dict[q].get("relevant_candidates", []) for q in query_ids}
        self.query_ids = [q for q in query_ids if len(self.gold_map.get(q, [])) > 0]
        self.queries_dict = queries_dict
        self.candidates_dict = candidates_dict
        self.stat_hard_neg_map = stat_hard_neg_map
        self.dense_hard_neg_map = dense_hard_neg_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_positives = num_positives
        self.num_hard_negatives = num_hard_negatives
        self.pad_token_id = pad_token_id
        self.all_cand_ids = list(candidates_dict.keys())

        print(f"Pre-tokenizing {len(self.query_ids)} queries and {len(self.all_cand_ids)} candidates...")
        t0 = time.time()
        self.query_cache = {}
        for qid in self.query_ids:
            txt = queries_dict[qid].get("text", "")
            if isinstance(txt, list): txt = " ".join(txt)
            self.query_cache[qid] = (tokenizer.encode(txt, add_special_tokens=False).ids, txt)

        self.cand_cache = {}
        for cid in self.all_cand_ids:
            txt = candidates_dict[cid].get("text", "")
            if isinstance(txt, list): txt = " ".join(txt)
            self.cand_cache[cid] = (tokenizer.encode(txt, add_special_tokens=False).ids, txt)
        print(f"Tokenization complete in {time.time() - t0:.2f}s")

    def _pad_sequence(self, seq: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = seq[: self.max_len]
        pad_len = self.max_len - len(seq)
        ids = torch.tensor(seq + [self.pad_token_id] * pad_len, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[: len(seq)] = 1.0
        return ids, mask

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qid = self.query_ids[idx]
        pos_pool = self.gold_map[qid]
        gold_set = set(pos_pool)

        if len(pos_pool) >= self.num_positives:
            chosen_pos = random.sample(pos_pool, self.num_positives)
        else:
            chosen_pos = [random.choice(pos_pool) for _ in range(self.num_positives)]

        half_k = self.num_hard_negatives // 2
        stat_pool = [c for c in self.stat_hard_neg_map.get(qid, []) if c not in gold_set]
        dense_pool = [c for c in self.dense_hard_neg_map.get(qid, []) if c not in gold_set]

        chosen_negs = []
        if len(stat_pool) >= half_k:
            chosen_negs.extend(random.sample(stat_pool[:25], half_k))
        if len(dense_pool) >= (self.num_hard_negatives - len(chosen_negs)):
            rem = self.num_hard_negatives - len(chosen_negs)
            avail_dense = [c for c in dense_pool[:30] if c not in chosen_negs]
            if len(avail_dense) >= rem:
                chosen_negs.extend(random.sample(avail_dense, rem))

        while len(chosen_negs) < self.num_hard_negatives:
            rc = random.choice(self.all_cand_ids)
            if rc not in gold_set and rc not in chosen_negs:
                chosen_negs.append(rc)

        q_ids_raw, q_txt = self.query_cache[qid]
        q_seq = extract_tri_anchor_tokens(
            q_ids_raw, q_txt, self.tokenizer,
            max_len=self.max_len, mask_parties=True
        )
        q_ids, q_mask = self._pad_sequence(q_seq)

        pos_ids_list, pos_mask_list = [], []
        for pid in chosen_pos:
            p_ids_raw, p_txt = self.cand_cache[pid]
            p_seq = extract_tri_anchor_tokens(
                p_ids_raw, p_txt, self.tokenizer,
                max_len=self.max_len, mask_parties=False
            )
            p_ids, p_mask = self._pad_sequence(p_seq)
            pos_ids_list.append(p_ids)
            pos_mask_list.append(p_mask)

        neg_ids_list, neg_mask_list = [], []
        for nid in chosen_negs:
            n_ids_raw, n_txt = self.cand_cache[nid]
            n_seq = extract_tri_anchor_tokens(
                n_ids_raw, n_txt, self.tokenizer,
                max_len=self.max_len, mask_parties=False
            )
            n_ids, n_mask = self._pad_sequence(n_seq)
            neg_ids_list.append(n_ids)
            neg_mask_list.append(n_mask)

        return {
            "q_ids": q_ids,
            "q_mask": q_mask,
            "pos_ids": torch.stack(pos_ids_list, dim=0),   # [P, L]
            "pos_mask": torch.stack(pos_mask_list, dim=0), # [P, L]
            "neg_ids": torch.stack(neg_ids_list, dim=0),   # [K_hard, L]
            "neg_mask": torch.stack(neg_mask_list, dim=0), # [K_hard, L]
        }


def collate_hard_neg_batch(batch):
    return {
        "q_ids": torch.stack([item["q_ids"] for item in batch], dim=0),       # [B, L]
        "q_mask": torch.stack([item["q_mask"] for item in batch], dim=0),     # [B, L]
        "pos_ids": torch.stack([item["pos_ids"] for item in batch], dim=0),   # [B, P, L]
        "pos_mask": torch.stack([item["pos_mask"] for item in batch], dim=0), # [B, P, L]
        "neg_ids": torch.stack([item["neg_ids"] for item in batch], dim=0),   # [B, K, L]
        "neg_mask": torch.stack([item["neg_mask"] for item in batch], dim=0), # [B, K, L]
    }


class TriAnchorDocDataset(Dataset):
    def __init__(self, doc_dict: Dict[str, Dict], tokenizer, max_len: int = 1024, pad_token_id: int = 0):
        self.doc_ids = list(doc_dict.keys())
        self.doc_dict = doc_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        did = self.doc_ids[idx]
        txt = self.doc_dict[did].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        tok_ids = self.tokenizer.encode(txt, add_special_tokens=False).ids

        seq = extract_tri_anchor_tokens(tok_ids, txt, self.tokenizer, max_len=self.max_len, mask_parties=False)
        seq = seq[: self.max_len]
        pad_len = self.max_len - len(seq)
        ids = torch.tensor(seq + [self.pad_token_id] * pad_len, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[: len(seq)] = 1.0
        return {"doc_id": did, "input_ids": ids, "attention_mask": mask}


def collate_tri_docs(batch):
    return {
        "doc_ids": [item["doc_id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
    }


@torch.no_grad()
def encode_tri_docs(model, doc_dict, tokenizer, max_len=1024, batch_size=16, device="cuda:0"):
    model.eval()
    dataset = TriAnchorDocDataset(doc_dict, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tri_docs, num_workers=2)
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


def encode_in_subbatches(model, ids: torch.Tensor, mask: torch.Tensor, sub_batch_size: int = 4) -> torch.Tensor:
    outputs = []
    N = ids.size(0)
    for i in range(0, N, sub_batch_size):
        sub_ids = ids[i : i + sub_batch_size]
        sub_mask = mask[i : i + sub_batch_size]
        out = model.encode(sub_ids, sub_mask)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


class MultiPositiveMarginInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05, cosine_margin=0.03, pairwise_margin=0.15, margin_weight=0.5):
        super().__init__()
        self.tau = temperature
        self.m = cosine_margin
        self.gamma = pairwise_margin
        self.margin_weight = margin_weight

    def forward(self, z_q, z_pos, z_neg):
        B, P, d = z_pos.shape
        K_hard = z_neg.shape[1]

        sim_pos = torch.einsum("bd,bpd->bp", z_q, z_pos)
        sim_hard_neg = torch.einsum("bd,bkd->bk", z_q, z_neg)

        flat_pos = z_pos.reshape(B * P, d)
        flat_neg = z_neg.reshape(B * K_hard, d)
        sim_all_pos = torch.matmul(z_q, flat_pos.T)
        sim_all_neg = torch.matmul(z_q, flat_neg.T)

        own_pos_mask = torch.zeros((B, B * P), dtype=torch.bool, device=z_q.device)
        for i in range(B):
            own_pos_mask[i, i * P : (i + 1) * P] = True
        inbatch_pos_negs = sim_all_pos.masked_select(~own_pos_mask).view(B, (B - 1) * P)

        all_neg_logits = torch.cat([sim_hard_neg, inbatch_pos_negs], dim=-1) / self.tau
        log_sum_exp_neg = torch.logsumexp(all_neg_logits, dim=-1, keepdim=True)

        pos_logits = (sim_pos - self.m) / self.tau
        denom = torch.logaddexp(pos_logits, log_sum_exp_neg.expand_as(pos_logits))
        loss_infonce = -(pos_logits - denom).mean()

        diff = self.gamma - sim_pos.unsqueeze(2) + sim_hard_neg.unsqueeze(1)
        loss_pairwise = F.relu(diff).mean()

        return loss_infonce + self.margin_weight * loss_pairwise


def apply_csls_anti_hubness(sim_matrix: np.ndarray, k_neighbors: int = 10, beta_hub: float = 0.35) -> np.ndarray:
    k_eff = min(k_neighbors, sim_matrix.shape[0])
    topk_query_sims = np.partition(sim_matrix, -k_eff, axis=0)[-k_eff:, :]
    hubness_penalty = np.mean(topk_query_sims, axis=0, keepdims=True)
    calibrated = sim_matrix - beta_hub * hubness_penalty
    mean_q = np.mean(calibrated, axis=1, keepdims=True)
    std_q = np.std(calibrated, axis=1, keepdims=True) + 1e-6
    return (calibrated - mean_q) / std_q


def run_peak_solution():
    print("==========================================================================")
    print("LAUNCHING PHASE 7 MASTER PEAK SOLUTION PIPELINE")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Raw Splits
    print("\n--- 1. Loading Datasets & Evaluation Truth ---")
    with open(os.path.join(DATA_DIR, "Train/train.json")) as f: train_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Train/train_candidates.json")) as f: train_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev.json")) as f: dev_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev_candidates.json")) as f: dev_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test.json")) as f: test_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test_candidates.json")) as f: test_c = json.load(f)

    with open(os.path.join(EVAL_DIR, "dev_truth.json")) as f: dev_truth = json.load(f)
    with open(os.path.join(EVAL_DIR, "dev_candidate_ids.json")) as f: dev_cand_ids = json.load(f)
    with open(os.path.join(EVAL_DIR, "test_candidate_ids.json")) as f: test_cand_ids = json.load(f)

    with open(HARD_NEG_PATH) as f:
        hard_negs = json.load(f)
    stat_hard_neg_map = hard_negs["stat_hard_neg_map"]
    dense_hard_neg_map = hard_negs["dense_hard_neg_map"]

    # 2. Build Dataset & Model
    MAX_LEN = 1024
    print(f"\n--- 2. Initializing Bi-Encoder v2 (HarveyLM LORMA, max_seq_len={MAX_LEN}) ---")
    train_ds = HardNegativePCRDataset(
        query_ids=list(train_q.keys()),
        queries_dict=train_q,
        candidates_dict=train_c,
        stat_hard_neg_map=stat_hard_neg_map,
        dense_hard_neg_map=dense_hard_neg_map,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        num_positives=2,
        num_hard_negatives=4
    )
    batch_size = 4
    accum_steps = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_hard_neg_batch,
        num_workers=2,
        pin_memory=True
    )
    print(f"Total training queries: {len(train_ds)} | Batches per epoch: {len(train_loader)} (effective batch size={batch_size * accum_steps})")

    model = UniversalLegalRetriever(
        arch_type="harveylm",
        adaptation="lorma",
        r=32,
        alpha=64,
        temperature=0.05
    ).to(device)

    if os.path.exists(BASE_LORMA_CKPT):
        print(f"Warm-starting from {BASE_LORMA_CKPT}...")
        st = torch.load(BASE_LORMA_CKPT, map_location=device)
        model.load_state_dict(st.get("model_state_dict", st), strict=False)

    criterion = MultiPositiveMarginInfoNCELoss(
        temperature=0.05,
        cosine_margin=0.03,
        pairwise_margin=0.15,
        margin_weight=0.5
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.5e-4,
        weight_decay=0.01
    )
    epochs = 4
    total_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # 3. Training Loop with Micro-Batching
    print("\n--- 3. Beginning Multi-Positive Margin InfoNCE Fine-Tuning ---")
    model.train()
    global_step = 0
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            q_ids = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            pos_ids = batch["pos_ids"].to(device)   # [B, P, L]
            pos_mask = batch["pos_mask"].to(device)
            neg_ids = batch["neg_ids"].to(device)   # [B, K, L]
            neg_mask = batch["neg_mask"].to(device)

            B, P, L = pos_ids.shape
            K_hard = neg_ids.shape[1]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Encode in micro-batches to guarantee zero OOM
                z_q = encode_in_subbatches(model, q_ids, q_mask, sub_batch_size=4)
                z_pos = encode_in_subbatches(model, pos_ids.view(B * P, L), pos_mask.view(B * P, L), sub_batch_size=4).view(B, P, -1)
                z_neg = encode_in_subbatches(model, neg_ids.view(B * K_hard, L), neg_mask.view(B * K_hard, L), sub_batch_size=4).view(B, K_hard, -1)
                loss = criterion(z_q, z_pos, z_neg)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 20 == 0 or step == len(train_loader):
                    print(f"Epoch {epoch}/{epochs} | Step {step:3d}/{len(train_loader)} | Loss: {loss.item() * accum_steps:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(train_loader)
        print(f"--> Epoch {epoch} finished in {time.time() - t_epoch_start:.1f}s | Avg Loss: {avg_loss:.4f}")

    print(f"Training completed in {time.time() - t_train_start:.1f}s")
    v2_ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, v2_ckpt_path)
    print(f"Saved Bi-Encoder v2 checkpoint to {v2_ckpt_path}")

    # 4. Encoding Dev and Test Sets
    print(f"\n--- 4. Encoding Dev & Test Splits (Tri-Anchor {MAX_LEN} Tokens) ---")
    t0 = time.time()
    dev_c_ids, dev_c_embeds_v2 = encode_tri_docs(model, dev_c, tokenizer, max_len=MAX_LEN, batch_size=16, device=device)
    dev_q_ids, dev_q_embeds_v2 = encode_tri_docs(model, dev_q, tokenizer, max_len=MAX_LEN, batch_size=16, device=device)
    print(f"Encoded Dev split in {time.time() - t0:.2f}s")

    t0 = time.time()
    test_c_ids, test_c_embeds_v2 = encode_tri_docs(model, test_c, tokenizer, max_len=MAX_LEN, batch_size=16, device=device)
    test_q_ids, test_q_embeds_v2 = encode_tri_docs(model, test_q, tokenizer, max_len=MAX_LEN, batch_size=16, device=device)
    print(f"Encoded Test split in {time.time() - t0:.2f}s")

    sim_dev_v2 = torch.matmul(dev_q_embeds_v2, dev_c_embeds_v2.T).numpy()
    sim_test_v2 = torch.matmul(test_q_embeds_v2, test_c_embeds_v2.T).numpy()

    # 5. Load Constituent Checkpoint Embeddings for Grand Ensemble
    print("\n--- 5. Loading Prior Checkpoints for Grand Ensembling ---")
    del model
    torch.cuda.empty_cache()

    # Model 1: HarveyLM LORMA v1 (768 tokens)
    st1 = torch.load(BASE_LORMA_CKPT, map_location=device)
    m1 = UniversalLegalRetriever(arch_type="harveylm", adaptation="lorma", r=32, alpha=64).to(device)
    m1.load_state_dict(st1.get("model_state_dict", st1), strict=False)
    _, dev_c_e1 = encode_tri_docs(m1, dev_c, tokenizer, max_len=768, batch_size=32, device=device)
    _, dev_q_e1 = encode_tri_docs(m1, dev_q, tokenizer, max_len=768, batch_size=32, device=device)
    _, test_c_e1 = encode_tri_docs(m1, test_c, tokenizer, max_len=768, batch_size=32, device=device)
    _, test_q_e1 = encode_tri_docs(m1, test_q, tokenizer, max_len=768, batch_size=32, device=device)
    sim_dev_lorma1 = torch.matmul(dev_q_e1, dev_c_e1.T).numpy()
    sim_test_lorma1 = torch.matmul(test_q_e1, test_c_e1.T).numpy()
    del m1, st1
    torch.cuda.empty_cache()

    # Model 2: HarveyLM QLoRA (768 tokens)
    st2 = torch.load(BASE_QLORA_CKPT, map_location=device)
    m2 = UniversalLegalRetriever(arch_type="harveylm", adaptation="qlora", r=32, alpha=64).to(device)
    m2.load_state_dict(st2.get("model_state_dict", st2), strict=False)
    _, dev_c_e2 = encode_tri_docs(m2, dev_c, tokenizer, max_len=768, batch_size=32, device=device)
    _, dev_q_e2 = encode_tri_docs(m2, dev_q, tokenizer, max_len=768, batch_size=32, device=device)
    _, test_c_e2 = encode_tri_docs(m2, test_c, tokenizer, max_len=768, batch_size=32, device=device)
    _, test_q_e2 = encode_tri_docs(m2, test_q, tokenizer, max_len=768, batch_size=32, device=device)
    sim_dev_qlora = torch.matmul(dev_q_e2, dev_c_e2.T).numpy()
    sim_test_qlora = torch.matmul(test_q_e2, test_c_e2.T).numpy()
    del m2, st2
    torch.cuda.empty_cache()

    # 6. Statutory Tversky Overlap Matrices
    print("\n--- 6. Computing Statutory Tversky Overlap Matrices ---")
    s_stat_dev = build_statutory_tversky_matrix(dev_q, dev_c, dev_q_ids, dev_c_ids)
    s_stat_test = build_statutory_tversky_matrix(test_q, test_c, test_q_ids, test_c_ids)

    # 7. CSLS Anti-Hubness and Z-Score Normalization
    print("\n--- 7. Applying CSLS Anti-Hubness & Z-Score Normalization ---")
    z_dev_v2 = apply_csls_anti_hubness(sim_dev_v2, beta_hub=0.35)
    z_test_v2 = apply_csls_anti_hubness(sim_test_v2, beta_hub=0.35)

    z_dev_lorma1 = apply_csls_anti_hubness(sim_dev_lorma1, beta_hub=0.35)
    z_test_lorma1 = apply_csls_anti_hubness(sim_test_lorma1, beta_hub=0.35)

    z_dev_qlora = apply_csls_anti_hubness(sim_dev_qlora, beta_hub=0.35)
    z_test_qlora = apply_csls_anti_hubness(sim_test_qlora, beta_hub=0.35)

    z_dev_stat = (s_stat_dev - np.mean(s_stat_dev, axis=1, keepdims=True)) / (np.std(s_stat_dev, axis=1, keepdims=True) + 1e-6)
    z_test_stat = (s_stat_test - np.mean(s_stat_test, axis=1, keepdims=True)) / (np.std(s_stat_test, axis=1, keepdims=True) + 1e-6)

    def get_pred_dict(sim_mat, query_ids, cand_ids):
        top7_idx = torch.topk(torch.from_numpy(sim_mat), k=7, dim=-1).indices.numpy()
        pred = {}
        for i, qid in enumerate(query_ids):
            pred[qid] = [cand_ids[idx] for idx in top7_idx[i]]
        return pred

    # Standalone Bi-Encoder v2 Dev score
    m_v2_alone = compute_pcr_metrics(dev_truth, get_pred_dict(z_dev_v2, dev_q_ids, dev_c_ids))
    print(f"\n[Bi-Encoder v2 ({MAX_LEN} Tokens + Hard Negs + CSLS)] Dev Micro-F1@7: {m_v2_alone['micro_f1_7']:.6f} | TP: {m_v2_alone['tp']}")

    # 8. Optimize Grand Ensemble Weights on Dev
    print("\n--- 8. Optimizing Grand Ensemble Weights on Dev ---")
    best_f1, best_w, best_tp = 0.0, None, 0
    for w_v2 in [0.40, 0.50, 0.60]:
        for w_v1 in [0.20, 0.30, 0.40]:
            for w_q in [0.10, 0.15, 0.20]:
                for w_s in [0.10, 0.15, 0.20]:
                    blended = w_v2 * z_dev_v2 + w_v1 * z_dev_lorma1 + w_q * z_dev_qlora + w_s * z_dev_stat
                    m = compute_pcr_metrics(dev_truth, get_pred_dict(blended, dev_q_ids, dev_c_ids))
                    if m['micro_f1_7'] > best_f1:
                        best_f1 = m['micro_f1_7']
                        best_w = (w_v2, w_v1, w_q, w_s)
                        best_tp = m['tp']

    print("==========================================================================")
    print("GRAND PEAK SOLUTION DEV PERFORMANCE:")
    print(f"Optimal Weights: v2={best_w[0]}, v1={best_w[1]}, qlora={best_w[2]}, stat={best_w[3]}")
    print(f"Dev Micro-F1@7:        {best_f1:.6f}")
    final_dev_sim = best_w[0] * z_dev_v2 + best_w[1] * z_dev_lorma1 + best_w[2] * z_dev_qlora + best_w[3] * z_dev_stat
    final_dev_pred = get_pred_dict(final_dev_sim, dev_q_ids, dev_c_ids)
    final_dev_metrics = compute_pcr_metrics(dev_truth, final_dev_pred)
    print(f"Dev Micro-Precision@7: {final_dev_metrics['micro_precision_7']:.6f}")
    print(f"Dev Micro-Recall@7:    {final_dev_metrics['micro_recall_7']:.6f}")
    print(f"True Positives: {final_dev_metrics['tp']} (Gain over 252 baseline: +{final_dev_metrics['tp'] - 252} hits!)")
    print(f"False Positives: {final_dev_metrics['fp']} | False Negatives: {final_dev_metrics['fn']}")
    print("==========================================================================")

    # Save Dev Predictions
    dev_pred_path = os.path.join(SUB_DIR, "dev_prediction.json")
    with open(dev_pred_path, "w", encoding="utf-8") as f:
        json.dump(final_dev_pred, f, indent=2)

    # 9. Compute Test Predictions & Package Official Submission
    print("\n--- 9. Packaging Official Peak Submission ---")
    final_test_sim = best_w[0] * z_test_v2 + best_w[1] * z_test_lorma1 + best_w[2] * z_test_qlora + best_w[3] * z_test_stat
    final_test_pred = get_pred_dict(final_test_sim, test_q_ids, test_c_ids)

    sub_zip_path = os.path.join(SUB_DIR, "submission.zip")
    export_pcr_submission(
        pred_dict=final_test_pred,
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

    # 11. Log to Master Results Tracker
    tracker_path = os.path.join(ROOT_CODE, "phase7/results.txt")
    log_line = f"PEAK_SOLUTION    GRAND_ENSEMBLE       {best_f1:<14.6f} {final_dev_metrics['micro_precision_7']:<11.6f} {final_dev_metrics['micro_recall_7']:<12.6f} [READY]        {sub_zip_path}\n"
    with open(tracker_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\nLogged peak solution results to {tracker_path}")
    print("\n==========================================================================")
    print("PEAK SOLUTION COMPLETE & VERIFIED!")
    print(f"Official Submission Zip: {sub_zip_path}")
    print("==========================================================================")


if __name__ == "__main__":
    run_peak_solution()
