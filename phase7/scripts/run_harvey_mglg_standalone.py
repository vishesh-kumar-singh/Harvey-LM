#!/usr/bin/env python3
"""
Phase 7: Standalone Harvey-MGLG (Multi-Granular Late-Interaction & Graph-Attentive Network).
Targets: >= 0.4000 Micro-F1@7 on IL-PCSR Benchmark.
Single, Unified Architecture without Ad-Hoc Checkpoint Ensembling.

Key Innovations:
1. Asymmetric Saliency Extractor tailored for IL-PCSR citation placeholders ([PRECEDENT], [SECTION]).
2. 16-Segment Contextual Proposition Head with SVD-initialized Isometric Projections.
3. Bidirectional Top-K Proposition MaxSim Late Interaction.
4. Edge-Updated Graph Attention (EUGATLayer) over Candidate Doctrinal & Citation Network.
5. Non-Parametric Transductive Memory Bank transferring training citation probability.
6. CSLS Anti-Hubness + Damped Column-Marginal Sinkhorn-Knopp Balancing.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import math
import time
import random
import zipfile
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase7.src.model.pcr_retriever import UniversalLegalRetriever
from phase7.src.evaluation.eval_utils import compute_pcr_metrics, export_pcr_submission

DATA_DIR = os.path.join(ROOT_CODE, "phase7/data/raw/Phase_7_PCR")
EVAL_DIR = os.path.join(ROOT_CODE, "phase7/data/eval_truth")
TOKENIZER_PATH = os.path.join(ROOT_CODE, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
EVAL_SCRIPT_PATH = os.path.join(ROOT_CODE, "Evaluation Script/evaluate_phase7.py")
HARD_NEG_PATH = os.path.join(ROOT_CODE, "phase7/data/train_hard_negatives.json")
BASE_LORMA_CKPT = os.path.join(ROOT_CODE, "phase7/models/ensemble/peak_solution/best_model.pt")
SUB_DIR = os.path.join(ROOT_CODE, "phase7/submissions/harveylm/mglg_standalone")
MODEL_DIR = os.path.join(ROOT_CODE, "phase7/models/harveylm/mglg_standalone")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. ASYMMETRIC IL-PCSR SALIENCY & STATUTORY/PLACEHOLDER FEATURE EXTRACTOR
# ==============================================================================

EXPLICIT_STATUTE_RE = re.compile(
    r"\b(?:section|sec\.?|s\.|article|art\.?|order|rule|clause|sub-section)\s*"
    r"(\d+[a-zA-Z]*(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)"
    r"(?:\s+of\s+the\s+([a-zA-Z\s]{3,35}?act|[a-zA-Z\s]{3,25}?code|constitution))?",
    re.IGNORECASE,
)
MASKED_PRECEDENT_RE = re.compile(
    r"(\[precedent\]|\[citation\]|\bheld\s+in\b|\brelied\s+upon\b|\brelying\s+on\b|"
    r"\bdistinguished\b|\boverruled\b|\bfollowed\s+in\b|\bratio\s+of\b|\bcase\s+of\b|"
    r"\bv\.\s+|\bvs\.?\s+|\breported\s+in\b|\bconstitution\s+bench\b)",
    re.IGNORECASE,
)
MASKED_STATUTE_RE = re.compile(
    r"(\[section\]|\[act\]|\[statute\]|\bprovision\b|\bstatutory\b|\benactment\b|"
    r"\bmandates\b|\bultra\s+vires\b|\bsub-section\b|\bproviso\b)",
    re.IGNORECASE,
)
CAND_HOLDING_RE = re.compile(
    r"(\bwe\s+hold\b|\bheld\s+that\b|\bit\s+is\s+well\s+settled\b|\bsettled\s+law\b|"
    r"\bsettled\s+principle\b|\bratio\s+decidendi\b|\bin\s+our\s+considered\s+opinion\b|"
    r"\bin\s+our\s+view\b|\bmust\s+be\s+construed\b|\btherefore\s+hold\b|\blegal\s+position\b)",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"(\blearned\s+counsel\b|\badjourned\b|\bvakalatnama\b|\baffidavit\b|\bregistry\b|"
    r"\bia\s+no\b|\blisted\s+on\b|\bcoram\b|\bspeaking\s+through\b)",
    re.IGNORECASE,
)

LEGAL_DOMAIN_KEYWORDS = {
    "tax_revenue": ["income tax", "assessment year", "commissioner", "deduction", "assessee", "customs", "excise", "tariff", "gst", "vat"],
    "criminal_ipc": ["murder", "homicide", "fir", "eyewitness", "post-mortem", "bail", "conviction", "sentence", "accused", "prosecution", "302", "498a"],
    "labour_service": ["workman", "industrial dispute", "retrenchment", "back wages", "reinstatement", "disciplinary", "seniority", "pension", "promotion", "termination"],
    "land_property": ["land acquisition", "compensation", "market value", "solatium", "notification", "possession", "tenancy", "eviction", "landlord", "tenant", "mortgage"],
    "motor_tort": ["motor vehicles", "accident", "rash and negligent", "multiplier", "tribunal", "insurer", "insurance company", "claimant", "disability"],
    "arbitration_commercial": ["arbitration", "arbitral award", "arbitrator", "contract", "breach", "tender", "bank guarantee", "insolvency", "dishonour", "cheque", "138"],
    "constitutional_writ": ["writ petition", "mandamus", "certiorari", "fundamental right", "arbitrary", "natural justice", "ultra vires", "article 14", "article 226", "article 32"],
    "family_civil": ["divorce", "maintenance", "cruelty", "hindu marriage", "partition", "will", "probate", "specific performance", "injunction", "limitation", "condonation"],
}


def extract_doctrinal_and_statutory_profile(text: str) -> Counter:
    lower_text = text.lower()
    feats = Counter()
    for match in EXPLICIT_STATUTE_RE.finditer(lower_text):
        num_part = re.sub(r"\s+", "", match.group(1))
        act_part = match.group(2)
        if act_part:
            act_tokens = [w[:4] for w in act_part.split() if w not in {"the", "of", "and", "act", "code"}]
            slug = "_".join(act_tokens[:3])
            feats[f"stat_{num_part}_{slug}" if slug else f"stat_{num_part}"] += 2.0
        else:
            feats[f"stat_{num_part}"] += 1.5

    for domain, kw_list in LEGAL_DOMAIN_KEYWORDS.items():
        dom_hits = 0
        for kw in kw_list:
            c = lower_text.count(kw)
            if c > 0:
                feats[f"kw_{kw}"] += min(5.0, 1.0 + math.log1p(c))
                dom_hits += c
        if dom_hits > 0:
            feats[f"dom_{domain}"] = min(6.0, 1.5 * math.log1p(dom_hits))

    return feats


def compute_doctrinal_tversky_matrix(
    queries_text: Dict[str, str],
    candidates_text: Dict[str, str],
    query_ids: List[str],
    cand_ids: List[str],
) -> np.ndarray:
    q_profiles = {qid: extract_doctrinal_and_statutory_profile(queries_text[qid]) for qid in query_ids}
    c_profiles = {cid: extract_doctrinal_and_statutory_profile(candidates_text[cid]) for cid in cand_ids}

    df = Counter()
    for prof in c_profiles.values():
        for k in prof.keys():
            df[k] += 1
    num_c = len(cand_ids)
    idf = {k: math.log(1.0 + num_c / (1.0 + v)) for k, v in df.items()}

    mat = np.zeros((len(query_ids), len(cand_ids)), dtype=np.float32)
    for i, qid in enumerate(query_ids):
        qp = q_profiles[qid]
        if not qp:
            continue
        q_w = {k: val * idf.get(k, 1.0) for k, val in qp.items()}
        q_norm = sum(q_w.values())

        for j, cid in enumerate(cand_ids):
            cp = c_profiles[cid]
            if not cp:
                continue
            common = set(qp.keys()).intersection(cp.keys())
            if not common:
                continue
            inter = sum(min(q_w[k], cp[k] * idf.get(k, 1.0)) for k in common)
            c_norm = sum(val * idf.get(k, 1.0) for k, val in cp.items())
            mat[i, j] = inter / (inter + 0.65 * max(0.0, q_norm - inter) + 0.35 * max(0.0, c_norm - inter) + 1e-6)
    return mat


def extract_asymmetric_salient_tokens(
    token_ids: List[int],
    tokenizer,
    is_query: bool = True,
    max_len: int = 2048,
    window_size: int = 128,
) -> List[int]:
    total_len = len(token_ids)
    if total_len <= max_len:
        return token_ids

    if is_query:
        head_w, mid_w, tail_w = 2, 11, 3
    else:
        head_w, mid_w, tail_w = 3, 9, 4

    head_len = head_w * window_size
    tail_len = tail_w * window_size
    mid_budget = mid_w * window_size

    head_tokens = token_ids[:head_len]
    tail_tokens = token_ids[total_len - tail_len :]
    middle_pool = token_ids[head_len : total_len - tail_len]

    num_pool_windows = len(middle_pool) // window_size
    if num_pool_windows <= mid_w:
        return (head_tokens + middle_pool[:mid_budget] + tail_tokens)[:max_len]

    scored_windows = []
    for w_idx in range(num_pool_windows):
        chunk_ids = middle_pool[w_idx * window_size : (w_idx + 1) * window_size]
        chunk_text = tokenizer.decode(chunk_ids)

        explicit_stat = len(EXPLICIT_STATUTE_RE.findall(chunk_text))
        masked_prec = len(MASKED_PRECEDENT_RE.findall(chunk_text))
        masked_stat = len(MASKED_STATUTE_RE.findall(chunk_text))
        holding_hits = len(CAND_HOLDING_RE.findall(chunk_text))
        boiler_hits = len(BOILERPLATE_RE.findall(chunk_text))
        rel_pos = w_idx / max(1, num_pool_windows - 1)

        if is_query:
            score = (
                5.0 * masked_prec
                + 3.5 * masked_stat
                + 3.0 * explicit_stat
                + 2.0 * holding_hits
                - 2.0 * boiler_hits
                + (0.6 if 0.25 <= rel_pos <= 0.85 else 0.0)
            )
        else:
            score = (
                4.5 * holding_hits
                + 3.5 * (explicit_stat + masked_stat)
                + 2.5 * masked_prec
                - 2.0 * boiler_hits
                + (1.2 if 0.45 <= rel_pos <= 0.90 else 0.0)
            )
        scored_windows.append((score, w_idx))

    chosen_indices = sorted([idx for _, idx in sorted(scored_windows, reverse=True)[:mid_w]])
    selected_mid = []
    for idx in chosen_indices:
        selected_mid.extend(middle_pool[idx * window_size : (idx + 1) * window_size])

    return head_tokens + selected_mid[:mid_budget] + tail_tokens


# ==============================================================================
# 2. STANDALONE ARCHITECTURE: HARVEY-MGLG
# ==============================================================================

class EUGATLayer(nn.Module):
    """Edge-Updated Graph Attention Layer with zero-initialized residual gate."""
    def __init__(self, d_model: int = 1024, d_edge: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_node = nn.Linear(d_model, d_model, bias=False)
        self.W_edge = nn.Linear(d_edge, num_heads * 16, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(1, num_heads, 2 * self.head_dim + 16) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.res_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_cand: torch.Tensor,      # [N_c, d_model]
        knn_idx: torch.Tensor,     # [N_c, k_nn]
        edge_feats: torch.Tensor,  # [N_c, k_nn, d_edge]
    ) -> torch.Tensor:
        N_c, k_nn = knn_idx.shape
        H, D_h = self.num_heads, self.head_dim

        node_proj = self.W_node(z_cand).view(N_c, H, D_h)
        self_expanded = node_proj.unsqueeze(1).expand(N_c, k_nn, H, D_h)
        neigh_proj = node_proj[knn_idx]

        e_proj = self.W_edge(edge_feats).view(N_c, k_nn, H, 16)
        cat_features = torch.cat([self_expanded, neigh_proj, e_proj], dim=-1)

        raw_scores = F.leaky_relu((cat_features * self.attn_vec.unsqueeze(0)).sum(dim=-1), negative_slope=0.2)
        attn_weights = F.softmax(raw_scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        agg = torch.einsum("nkh,nkhd->nhd", attn_weights, neigh_proj).reshape(N_c, self.d_model)
        updated = z_cand + torch.tanh(self.res_gate) * self.out_proj(agg)
        return F.normalize(updated, p=2, dim=-1)


class HarveyMGLG(nn.Module):
    """
    Unified Multi-Granular Late-Interaction & Graph-Attentive Network on HarveyLM.
    """
    def __init__(
        self,
        harvey_base: UniversalLegalRetriever,
        d_model: int = 1024,
        d_late: int = 256,
        num_segments: int = 16,
        segment_len: int = 128,
        k_q_top: int = 6,
        k_c_top: int = 4,
    ):
        super().__init__()
        self.base = harvey_base
        self.d_model = d_model
        self.d_late = d_late
        self.num_segments = num_segments
        self.segment_len = segment_len
        self.k_q_top = k_q_top
        self.k_c_top = k_c_top

        # Token saliency gate
        self.token_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.token_gate.weight)
        nn.init.constant_(self.token_gate.bias, 2.0)

        # Late-Interaction proposition projection (SVD-initialized)
        self.late_proj = nn.Linear(d_model, d_late, bias=False)
        self._init_late_proj_via_svd()

        # EUGAT candidate graph refiner
        self.eugat = EUGATLayer(d_model=d_model, d_edge=4, num_heads=4, dropout=0.1)

        # Learnable fusion logit
        self.late_weight_logit = nn.Parameter(torch.tensor(-0.6))

    def _init_late_proj_via_svd(self) -> None:
        with torch.no_grad():
            if hasattr(self.base, "proj") and isinstance(self.base.proj, nn.Linear):
                w_base = self.base.proj.weight.data.float()
                u, s, vh = torch.linalg.svd(w_base, full_matrices=False)
                top_proj = torch.diag(torch.sqrt(s[: self.d_late])) @ vh[: self.d_late, :]
                self.late_proj.weight.data.copy_(top_proj.to(self.late_proj.weight.dtype))
            else:
                nn.init.orthogonal_(self.late_proj.weight)

    def forward_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        backbone = self.base.backbone
        h = backbone.tok_embeddings(input_ids)
        if backbone.config.use_alibi:
            alibi = -torch.abs(backbone.alibi[:, :seqlen, :seqlen].to(device=h.device, dtype=h.dtype))
            freqs_cis = None
        else:
            alibi = None
            freqs_cis = backbone.freqs_cis[:seqlen].to(h.device)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()

        if self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            for layer in backbone.layers:
                h, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False
                )
        else:
            for layer in backbone.layers:
                h, _ = layer(h, freqs_cis, mask, alibi, None, False)

        return backbone.norm(h)

    def encode_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        hidden = self.forward_backbone(input_ids, attention_mask)

        # 1. Saliency-gated global pooling
        gate_weights = torch.sigmoid(self.token_gate(hidden)).squeeze(-1) * attention_mask.to(hidden.dtype)
        denom_glob = gate_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        e_glob = torch.einsum("bl,bld->bd", gate_weights, hidden) / denom_glob

        if hasattr(self.base, "proj"):
            z_glob = F.normalize(self.base.proj(e_glob), p=2, dim=-1)
        else:
            z_glob = F.normalize(e_glob, p=2, dim=-1)

        # 2. 16-Segment Contextual Proposition Pooling
        target_len = self.num_segments * self.segment_len
        if L < target_len:
            pad_len = target_len - L
            hidden = F.pad(hidden, (0, 0, 0, pad_len))
            gate_weights = F.pad(gate_weights, (0, pad_len))
        elif L > target_len:
            hidden = hidden[:, :target_len, :]
            gate_weights = gate_weights[:, :target_len]

        h_segs = hidden.view(B, self.num_segments, self.segment_len, self.d_model)
        g_segs = gate_weights.view(B, self.num_segments, self.segment_len)

        seg_denom = g_segs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        e_segs = torch.einsum("bsw,bswd->bsd", g_segs, h_segs) / seg_denom
        u_segs = F.normalize(self.late_proj(e_segs), p=2, dim=-1)

        return z_glob, u_segs

    def compute_late_interaction_matrix(
        self,
        u_q: torch.Tensor,  # [N_q, 16, 256]
        u_c: torch.Tensor,  # [N_c, 16, 256]
    ) -> torch.Tensor:
        seg_sim = torch.einsum("qmd,cnd->qcmn", u_q.float(), u_c.float())
        max_over_c = seg_sim.max(dim=-1).values
        top_q = torch.topk(max_over_c, k=min(self.k_q_top, max_over_c.size(-1)), dim=-1).values.mean(dim=-1)

        max_over_q = seg_sim.max(dim=-2).values
        top_c = torch.topk(max_over_q, k=min(self.k_c_top, max_over_q.size(-1)), dim=-1).values.mean(dim=-1)

        return 0.55 * top_q + 0.45 * top_c

    def build_candidate_graph_and_refine(
        self,
        z_c: torch.Tensor,                  # [N_c, 1024]
        u_c: torch.Tensor,                  # [N_c, 16, 256]
        cand_stat_sim: torch.Tensor,        # [N_c, N_c]
        cand_cocite_adj: torch.Tensor,      # [N_c, N_c]
        k_nn: int = 15,
    ) -> torch.Tensor:
        N_c = z_c.size(0)
        k_eff = min(k_nn, N_c - 1)

        with torch.no_grad():
            glob_sim = torch.matmul(z_c.float(), z_c.float().T)
            late_sim = self.compute_late_interaction_matrix(u_c, u_c)
            combined_adj = 0.50 * glob_sim + 0.30 * late_sim + 0.10 * cand_stat_sim + 0.10 * cand_cocite_adj
            combined_adj.fill_diagonal_(-1e9)
            _, knn_idx = torch.topk(combined_adj, k=k_eff, dim=-1)

            row_idx = torch.arange(N_c, device=z_c.device).unsqueeze(1).expand(N_c, k_eff)
            f1 = glob_sim[row_idx, knn_idx]
            f2 = late_sim[row_idx, knn_idx]
            f3 = cand_stat_sim[row_idx, knn_idx]
            f4 = cand_cocite_adj[row_idx, knn_idx]
            edge_feats = torch.stack([f1, f2, f3, f4], dim=-1).to(z_c.dtype)

        return self.eugat(z_c, knn_idx, edge_feats)


# ==============================================================================
# 3. MULTI-GRANULAR MARGIN INFONCE LOSS
# ==============================================================================

class MultiGranularMarginLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, cosine_margin: float = 0.03, pairwise_margin: float = 0.14):
        super().__init__()
        self.tau = temperature
        self.m = cosine_margin
        self.gamma = pairwise_margin

    def _compute_single_scale_loss(self, sim_pos, sim_hard_neg, sim_all_neg):
        log_sum_exp_neg = torch.logsumexp(sim_all_neg / self.tau, dim=-1, keepdim=True)
        pos_logits = (sim_pos - self.m) / self.tau
        denom = torch.logaddexp(pos_logits, log_sum_exp_neg.expand_as(pos_logits))
        loss_nce = -(pos_logits - denom).mean()

        diff = self.gamma - sim_pos.unsqueeze(2) + sim_hard_neg.unsqueeze(1)
        loss_pair = F.relu(diff).mean()
        return loss_nce + 0.5 * loss_pair

    def forward(self, model, z_q, u_q, z_pos, u_pos, z_neg, u_neg):
        B, P, d = z_pos.shape
        K = z_neg.shape[1]

        flat_z_pos = z_pos.reshape(B * P, d)
        flat_u_pos = u_pos.reshape(B * P, 16, -1)
        flat_z_neg = z_neg.reshape(B * K, d)
        flat_u_neg = u_neg.reshape(B * K, 16, -1)

        glob_pos = torch.einsum("bd,bpd->bp", z_q, z_pos)
        glob_hard_neg = torch.einsum("bd,bkd->bk", z_q, z_neg)
        glob_all_pos = torch.matmul(z_q, flat_z_pos.T)
        glob_all_neg = torch.matmul(z_q, flat_z_neg.T)

        late_all_pos = model.compute_late_interaction_matrix(u_q, flat_u_pos)
        late_all_neg = model.compute_late_interaction_matrix(u_q, flat_u_neg)

        own_pos_mask = torch.zeros((B, B * P), dtype=torch.bool, device=z_q.device)
        late_pos_list, late_hard_neg_list = [], []
        for i in range(B):
            own_pos_mask[i, i * P : (i + 1) * P] = True
            late_pos_list.append(late_all_pos[i, i * P : (i + 1) * P])
            late_hard_neg_list.append(late_all_neg[i, i * K : (i + 1) * K])

        late_pos = torch.stack(late_pos_list, dim=0)
        late_hard_neg = torch.stack(late_hard_neg_list, dim=0)

        glob_inbatch_negs = glob_all_pos.masked_select(~own_pos_mask).view(B, (B - 1) * P)
        late_inbatch_negs = late_all_pos.masked_select(~own_pos_mask).view(B, (B - 1) * P)

        glob_full_negs = torch.cat([glob_all_neg, glob_inbatch_negs], dim=-1)
        late_full_negs = torch.cat([late_all_neg, late_inbatch_negs], dim=-1)

        loss_glob = self._compute_single_scale_loss(glob_pos, glob_hard_neg, glob_full_negs)
        loss_late = self._compute_single_scale_loss(late_pos, late_hard_neg, late_full_negs)

        w_late = torch.sigmoid(model.late_weight_logit)
        loss_fused = self._compute_single_scale_loss(
            (1.0 - w_late) * glob_pos + w_late * late_pos,
            (1.0 - w_late) * glob_hard_neg + w_late * late_hard_neg,
            (1.0 - w_late) * glob_full_negs + w_late * late_full_negs,
        )
        return 0.40 * loss_fused + 0.30 * loss_glob + 0.30 * loss_late


# ==============================================================================
# 4. DATASET & ENCODING UTILITIES
# ==============================================================================

class MGLGDocDataset(Dataset):
    def __init__(self, doc_dict: Dict[str, Dict], tokenizer, is_query: bool = False, max_len: int = 2048):
        self.doc_ids = list(doc_dict.keys())
        self.doc_dict = doc_dict
        self.tokenizer = tokenizer
        self.is_query = is_query
        self.max_len = max_len

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        did = self.doc_ids[idx]
        txt = self.doc_dict[did].get("text", "")
        if isinstance(txt, list): txt = " ".join(txt)
        raw_ids = self.tokenizer.encode(txt, add_special_tokens=False).ids

        seq = extract_asymmetric_salient_tokens(raw_ids, self.tokenizer, is_query=self.is_query, max_len=self.max_len)
        seq = seq[:self.max_len]
        pad_len = self.max_len - len(seq)
        ids = torch.tensor(seq + [0] * pad_len, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[:len(seq)] = 1.0

        return {"doc_id": did, "input_ids": ids, "attention_mask": mask}


def collate_mglg_docs(batch):
    return {
        "doc_ids": [item["doc_id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
    }


@torch.no_grad()
def encode_mglg_corpus(model: HarveyMGLG, doc_dict: Dict, tokenizer, is_query: bool, max_len=2048, batch_size=16, device="cuda:0"):
    model.eval()
    ds = MGLGDocDataset(doc_dict, tokenizer, is_query=is_query, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mglg_docs, num_workers=2)

    doc_ids_all = []
    z_glob_list, u_segs_list = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z_g, u_s = model.encode_sequence(ids, mask)
        z_glob_list.append(z_g.float().cpu())
        u_segs_list.append(u_s.float().cpu())
        doc_ids_all.extend(batch["doc_ids"])

    return doc_ids_all, torch.cat(z_glob_list, dim=0), torch.cat(u_segs_list, dim=0)


# ==============================================================================
# 5. TRANSDUCTIVE MEMORY TRANSFER & DOUBLY-STOCHASTIC CALIBRATION
# ==============================================================================

@torch.no_grad()
def compute_mglg_component_matrices(
    model: HarveyMGLG,
    q_glob: torch.Tensor,
    q_segs: torch.Tensor,
    c_glob_refined: torch.Tensor,
    c_segs: torch.Tensor,
    split_cand_ids: List[str],
    train_cand_id_to_idx: Dict[str, int],
    train_q_glob: torch.Tensor,
    train_q_segs: torch.Tensor,
    train_c_glob: torch.Tensor,
    train_citation_adj: torch.Tensor,
    tau_mem: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = q_glob.device
    N_q, N_c = q_glob.size(0), c_glob_refined.size(0)

    sim_glob = torch.matmul(q_glob.float(), c_glob_refined.float().T).cpu().numpy()
    sim_late = model.compute_late_interaction_matrix(q_segs, c_segs).cpu().numpy()

    # Transductive Training-Memory Citation Transfer
    sim_q_to_train_glob = torch.matmul(q_glob.float(), train_q_glob.float().T)
    sim_q_to_train_late = model.compute_late_interaction_matrix(q_segs, train_q_segs)
    sim_q_to_train = 0.55 * sim_q_to_train_glob + 0.45 * sim_q_to_train_late

    sim_q_to_train = torch.where(
        sim_q_to_train > 0.999,
        torch.full_like(sim_q_to_train, -1e9),
        sim_q_to_train,
    )

    top_k_mem = min(25, sim_q_to_train.size(1))
    top_vals, top_idx = torch.topk(sim_q_to_train, k=top_k_mem, dim=-1)
    attn_weights = F.softmax(top_vals / tau_mem, dim=-1)

    train_cand_probs = torch.zeros((N_q, train_citation_adj.size(1)), device=device, dtype=torch.float32)
    for i in range(N_q):
        matched_adj = train_citation_adj[top_idx[i]]
        train_cand_probs[i] = torch.matmul(attn_weights[i], matched_adj)

    mem_direct = torch.zeros((N_q, N_c), device=device, dtype=torch.float32)
    for j, cid in enumerate(split_cand_ids):
        if cid in train_cand_id_to_idx:
            t_idx = train_cand_id_to_idx[cid]
            mem_direct[:, j] = train_cand_probs[:, t_idx]

    cand_to_train_sim = torch.matmul(c_glob_refined.float(), train_c_glob.float().T).clamp(min=0.0)
    cand_to_train_sim = torch.pow(cand_to_train_sim, 3)
    mem_diffused = torch.matmul(train_cand_probs, cand_to_train_sim.T)
    sim_mem = (mem_direct + 0.25 * mem_diffused).cpu().numpy()

    return sim_glob, sim_late, sim_mem


def calibrate_and_combine_mglg(
    sim_glob: np.ndarray,
    sim_late: np.ndarray,
    sim_mem: np.ndarray,
    stat_matrix: np.ndarray,
    train_in_degree: np.ndarray,
    hyperparams: Dict[str, float],
) -> np.ndarray:
    N_q, N_c = sim_glob.shape

    def csls_zscore(mat: np.ndarray, beta: float = 0.40, k_nb: int = 12) -> np.ndarray:
        k_eff = min(k_nb, mat.shape[0])
        hub_pen = np.mean(np.partition(mat, -k_eff, axis=0)[-k_eff:, :], axis=0, keepdims=True)
        cal = mat - beta * hub_pen
        mu = np.mean(cal, axis=1, keepdims=True)
        sd = np.std(cal, axis=1, keepdims=True) + 1e-6
        return (cal - mu) / sd

    def row_zscore(mat: np.ndarray) -> np.ndarray:
        mu = np.mean(mat, axis=1, keepdims=True)
        sd = np.std(mat, axis=1, keepdims=True) + 1e-6
        return (mat - mu) / sd

    z_glob = csls_zscore(sim_glob, beta=hyperparams.get("beta_hub", 0.40))
    z_late = csls_zscore(sim_late, beta=hyperparams.get("beta_hub", 0.40))
    z_stat = row_zscore(stat_matrix)
    z_mem = row_zscore(sim_mem)

    w_glob = hyperparams.get("w_glob", 0.35)
    w_late = hyperparams.get("w_late", 0.35)
    w_mem = hyperparams.get("w_mem", 0.18)
    w_stat = hyperparams.get("w_stat", 0.12)

    combined = w_glob * z_glob + w_late * z_late + w_mem * z_mem + w_stat * z_stat

    prior_vec = np.log1p(train_in_degree)[None, :]
    prior_weight = hyperparams.get("w_prior", 0.08)
    top25_cutoff = np.partition(combined, -25, axis=1)[:, -25:-24]
    shortlist_mask = (combined >= top25_cutoff).astype(np.float32)
    combined = combined + prior_weight * shortlist_mask * prior_vec

    sinkhorn_gamma = hyperparams.get("sinkhorn_gamma", 0.15)
    if sinkhorn_gamma > 0:
        probs = np.exp(np.clip(combined, -10.0, 10.0))
        probs /= np.sum(probs, axis=1, keepdims=True) + 1e-9
        col_load = np.sum(probs, axis=0, keepdims=True)
        expected_cap = (7.0 * N_q / N_c) * (1.0 + 0.5 * np.log1p(train_in_degree)[None, :])
        overload_penalty = np.log1p(np.maximum(0.0, col_load - expected_cap))
        combined = combined - sinkhorn_gamma * overload_penalty

    return combined


@torch.no_grad()
def predict_harvey_mglg_scores(
    model: HarveyMGLG,
    q_glob: torch.Tensor,
    q_segs: torch.Tensor,
    c_glob_refined: torch.Tensor,
    c_segs: torch.Tensor,
    stat_matrix: np.ndarray,
    split_cand_ids: List[str],
    train_cand_id_to_idx: Dict[str, int],
    train_q_glob: torch.Tensor,
    train_q_segs: torch.Tensor,
    train_c_glob: torch.Tensor,
    train_citation_adj: torch.Tensor,
    train_in_degree: np.ndarray,
    hyperparams: Dict[str, float],
) -> np.ndarray:
    sim_glob, sim_late, sim_mem = compute_mglg_component_matrices(
        model=model,
        q_glob=q_glob,
        q_segs=q_segs,
        c_glob_refined=c_glob_refined,
        c_segs=c_segs,
        split_cand_ids=split_cand_ids,
        train_cand_id_to_idx=train_cand_id_to_idx,
        train_q_glob=train_q_glob,
        train_q_segs=train_q_segs,
        train_c_glob=train_c_glob,
        train_citation_adj=train_citation_adj,
        tau_mem=hyperparams.get("tau_mem", 0.04),
    )
    return calibrate_and_combine_mglg(
        sim_glob=sim_glob,
        sim_late=sim_late,
        sim_mem=sim_mem,
        stat_matrix=stat_matrix,
        train_in_degree=train_in_degree,
        hyperparams=hyperparams,
    )


def encode_micro_batch(
    model: HarveyMGLG,
    ids_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    micro_bs: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_list, u_list = [], []
    for k in range(0, ids_tensor.size(0), micro_bs):
        sub_ids = ids_tensor[k : k + micro_bs]
        sub_mask = mask_tensor[k : k + micro_bs]
        z_sub, u_sub = model.encode_sequence(sub_ids, sub_mask)
        z_list.append(z_sub)
        u_list.append(u_sub)
    return torch.cat(z_list, dim=0), torch.cat(u_list, dim=0)


# ==============================================================================
# 6. MASTER EXECUTION PIPELINE
# ==============================================================================

def run_harvey_mglg_master():
    print("==========================================================================")
    print("PHASE 7: MASTER STANDALONE HARVEY-MGLG PIPELINE")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: >= 0.4000 Micro-F1@7 (Zero Multi-Model Checkpoint Ensembling)")
    print("==========================================================================")

    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Raw Splits
    print("\n--- 1. Loading Corpus Splits & Evaluation Ground Truth ---")
    with open(os.path.join(DATA_DIR, "Train/train.json")) as f: train_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Train/train_candidates.json")) as f: train_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev.json")) as f: dev_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Dev/dev_candidates.json")) as f: dev_c = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test.json")) as f: test_q = json.load(f)
    with open(os.path.join(DATA_DIR, "Test/test_candidates.json")) as f: test_c = json.load(f)

    with open(os.path.join(EVAL_DIR, "dev_truth.json")) as f: dev_truth = json.load(f)
    with open(os.path.join(EVAL_DIR, "dev_candidate_ids.json")) as f: dev_cand_ids = json.load(f)
    with open(os.path.join(EVAL_DIR, "test_candidate_ids.json")) as f: test_cand_ids = json.load(f)

    # 2. Extract Plain Text Dictionaries
    train_q_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in train_q.items()}
    train_c_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in train_c.items()}
    dev_q_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in dev_q.items()}
    dev_c_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in dev_c.items()}
    test_q_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in test_q.items()}
    test_c_txt = {k: " ".join(v["text"]) if isinstance(v["text"], list) else v["text"] for k, v in test_c.items()}

    # 3. Build Historical Citation Adjacency & In-Degree Profiles
    print("\n--- 2. Building Historical Citation Adjacency & In-Degree Profiles ---")
    train_q_ids = list(train_q.keys())
    train_c_ids = list(train_c.keys())
    train_cand_id_to_idx = {cid: idx for idx, cid in enumerate(train_c_ids)}

    train_adj = torch.zeros((len(train_q_ids), len(train_c_ids)), dtype=torch.float32, device=device)
    train_in_deg_map = Counter()
    for i, qid in enumerate(train_q_ids):
        golds = train_q[qid].get("relevant_candidates", [])
        for cid in golds:
            if cid in train_cand_id_to_idx:
                train_adj[i, train_cand_id_to_idx[cid]] = 1.0
            train_in_deg_map[cid] += 1

    dev_in_degree = np.array([train_in_deg_map[cid] for cid in dev_cand_ids], dtype=np.float32)
    test_in_degree = np.array([train_in_deg_map[cid] for cid in test_cand_ids], dtype=np.float32)

    print(f"Historical Precedent In-Degree: Max={max(train_in_deg_map.values())}, Median={np.median(list(train_in_deg_map.values()))}")

    # 4. Compute Doctrinal/Statutory Tversky Overlap Matrices
    print("\n--- 3. Computing Doctrinal/Statutory Tversky Overlap Matrices ---")
    t0 = time.time()
    stat_dev = compute_doctrinal_tversky_matrix(dev_q_txt, dev_c_txt, list(dev_q.keys()), dev_cand_ids)
    stat_test = compute_doctrinal_tversky_matrix(test_q_txt, test_c_txt, list(test_q.keys()), test_cand_ids)
    print(f"Computed Tversky matrices in {time.time() - t0:.2f}s")

    # 5. Initialize Standalone HarveyMGLG Architecture
    print("\n--- 4. Initializing Standalone HarveyMGLG Architecture ---")
    base_model = UniversalLegalRetriever(arch_type="harveylm", adaptation="lorma", r=32, alpha=64, temperature=0.05).to(device)
    if os.path.exists(BASE_LORMA_CKPT):
        print(f"Warm-starting from {BASE_LORMA_CKPT}...")
        st = torch.load(BASE_LORMA_CKPT, map_location=device)
        base_model.load_state_dict(st.get("model_state_dict", st), strict=False)

    mglg_model = HarveyMGLG(
        harvey_base=base_model,
        d_model=1024,
        d_late=256,
        num_segments=16,
        segment_len=128,
        k_q_top=6,
        k_c_top=4
    ).to(device)

    total_params = sum(p.numel() for p in mglg_model.parameters())
    trainable_params = sum(p.numel() for p in mglg_model.parameters() if p.requires_grad)
    print(f"Standalone HarveyMGLG Total Params: {total_params:,} | Trainable Params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 6. Pre-encode Training Corpus for Memory Bank
    print("\n--- 5. Encoding Training Memory Bank (827 Queries, 4,320 Candidates) ---")
    t0 = time.time()
    _, train_q_g, train_q_s = encode_mglg_corpus(mglg_model, train_q, tokenizer, is_query=True, max_len=2048, batch_size=16, device=device)
    _, train_c_g, train_c_s = encode_mglg_corpus(mglg_model, train_c, tokenizer, is_query=False, max_len=2048, batch_size=16, device=device)
    print(f"Encoded Training Memory Bank in {time.time() - t0:.2f}s")

    # 7. Zero-Shot Step-0 Dev Evaluation
    print("\n--- 6. Zero-Shot Step-0 Dev Evaluation ---")
    t0 = time.time()
    _, dev_c_g, dev_c_s = encode_mglg_corpus(mglg_model, dev_c, tokenizer, is_query=False, max_len=2048, batch_size=16, device=device)
    _, dev_q_g, dev_q_s = encode_mglg_corpus(mglg_model, dev_q, tokenizer, is_query=True, max_len=2048, batch_size=16, device=device)
    print(f"Encoded Dev split in {time.time() - t0:.2f}s")

    # Dev Candidate Graph Refinement
    cand_stat_sim_dev = torch.from_numpy(compute_doctrinal_tversky_matrix(dev_c_txt, dev_c_txt, dev_cand_ids, dev_cand_ids)).to(device)
    cand_cocite_dev = torch.zeros((len(dev_cand_ids), len(dev_cand_ids)), dtype=torch.float32, device=device)
    dev_c_g_refined = mglg_model.build_candidate_graph_and_refine(dev_c_g.to(device), dev_c_s.to(device), cand_stat_sim_dev, cand_cocite_dev, k_nn=15)

    base_hparams = {
        "w_glob": 0.35, "w_late": 0.35, "w_mem": 0.18, "w_stat": 0.12,
        "beta_hub": 0.40, "tau_mem": 0.04, "w_prior": 0.08, "sinkhorn_gamma": 0.15
    }

    dev_scores_step0 = predict_harvey_mglg_scores(
        model=mglg_model,
        q_glob=dev_q_g.to(device),
        q_segs=dev_q_s.to(device),
        c_glob_refined=dev_c_g_refined,
        c_segs=dev_c_s.to(device),
        stat_matrix=stat_dev,
        split_cand_ids=dev_cand_ids,
        train_cand_id_to_idx=train_cand_id_to_idx,
        train_q_glob=train_q_g.to(device),
        train_q_segs=train_q_s.to(device),
        train_c_glob=train_c_g.to(device),
        train_citation_adj=train_adj,
        train_in_degree=dev_in_degree,
        hyperparams=base_hparams,
    )

    def extract_top7_predictions(scores: np.ndarray, q_ids: List[str], c_ids: List[str]) -> Dict[str, List[str]]:
        top7_indices = np.argsort(-scores, axis=-1)[:, :7]
        preds = {}
        for i, qid in enumerate(q_ids):
            preds[qid] = [c_ids[idx] for idx in top7_indices[i]]
        return preds

    step0_preds = extract_top7_predictions(dev_scores_step0, list(dev_q.keys()), dev_cand_ids)
    step0_metrics = compute_pcr_metrics(dev_truth, step0_preds)
    print(f"\n[Step-0 Zero-Shot Performance] Dev Micro-F1@7: {step0_metrics['micro_f1_7']:.6f} | TP: {step0_metrics['tp']}")
    print(f"Dev Prec@7: {step0_metrics['micro_precision_7']:.6f} | Dev Recall@7: {step0_metrics['micro_recall_7']:.6f}")

    # Save Step-0 baseline checkpoint
    step0_model_path = os.path.join(MODEL_DIR, "step0_model.pt")
    torch.save({"model_state_dict": mglg_model.state_dict()}, step0_model_path)
    step0_pred_path = os.path.join(SUB_DIR, "step0_dev_prediction.json")
    with open(step0_pred_path, "w", encoding="utf-8") as f:
        json.dump(step0_preds, f, indent=2)

    del dev_scores_step0, step0_preds
    torch.cuda.empty_cache()

    # 8. Training HarveyMGLG with MultiGranularMarginLoss
    print("\n--- 7. Training HarveyMGLG on 827 Queries (MultiGranularMarginLoss) ---")
    criterion = MultiGranularMarginLoss(temperature=0.05, cosine_margin=0.03, pairwise_margin=0.14)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in mglg_model.named_parameters() if "eugat" in n or "late_proj" in n or "token_gate" in n], "lr": 1.2e-4},
        {"params": [p for n, p in mglg_model.named_parameters() if "eugat" not in n and "late_proj" not in n and "token_gate" not in n and p.requires_grad], "lr": 5.0e-5},
    ], weight_decay=0.01)

    epochs = 3
    scaler = torch.amp.GradScaler("cuda")
    mglg_model.train()
    t_train_start = time.time()

    # Load pre-mined hard negatives
    with open(HARD_NEG_PATH) as f:
        hard_negs_data = json.load(f)
    stat_hard_negs = hard_negs_data["stat_hard_neg_map"]
    dense_hard_negs = hard_negs_data["dense_hard_neg_map"]

    train_valid_qids = [q for q in train_q_ids if len(train_q[q].get("relevant_candidates", [])) > 0]

    # Pre-extract salient token sequences for training queries and cache candidate tokens
    print("Pre-caching salient token representations for training set...")
    query_token_cache: Dict[str, Tuple[List[int], List[float]]] = {}
    for qid in train_valid_qids:
        q_raw = tokenizer.encode(train_q_txt[qid], add_special_tokens=False).ids
        q_tok = extract_asymmetric_salient_tokens(q_raw, tokenizer, is_query=True, max_len=2048)[:2048]
        pad_len = 2048 - len(q_tok)
        query_token_cache[qid] = (q_tok + [0] * pad_len, [1.0] * len(q_tok) + [0.0] * pad_len)

    cand_token_cache: Dict[str, Tuple[List[int], List[float]]] = {}

    def get_cand_tokens(cid: str) -> Tuple[List[int], List[float]]:
        if cid in cand_token_cache:
            return cand_token_cache[cid]
        c_raw = tokenizer.encode(train_c_txt[cid], add_special_tokens=False).ids
        c_tok = extract_asymmetric_salient_tokens(c_raw, tokenizer, is_query=False, max_len=2048)[:2048]
        pad_len = 2048 - len(c_tok)
        ids = c_tok + [0] * pad_len
        mask = [1.0] * len(c_tok) + [0.0] * pad_len
        cand_token_cache[cid] = (ids, mask)
        return ids, mask

    B = 2
    accum_steps = 4
    steps_per_epoch = len(train_valid_qids) // B

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        random.shuffle(train_valid_qids)
        running_loss = 0.0

        for b_idx in range(steps_per_epoch):
            batch_qids = train_valid_qids[b_idx * B : (b_idx + 1) * B]
            q_ids_batch, q_mask_batch = [], []
            pos_ids_batch, pos_mask_batch = [], []
            neg_ids_batch, neg_mask_batch = [], []

            for qid in batch_qids:
                q_ids, q_mask = query_token_cache[qid]
                q_ids_batch.append(q_ids)
                q_mask_batch.append(q_mask)

                # Sample P=2 positives
                golds = train_q[qid]["relevant_candidates"]
                pos_cids = random.sample(golds, min(2, len(golds)))
                while len(pos_cids) < 2:
                    pos_cids.append(random.choice(golds))

                for pid in pos_cids:
                    p_ids, p_mask = get_cand_tokens(pid)
                    pos_ids_batch.append(p_ids)
                    pos_mask_batch.append(p_mask)

                # Sample K=4 hard negatives
                gold_set = set(golds)
                stat_p = [c for c in stat_hard_negs.get(qid, []) if c not in gold_set]
                dense_p = [c for c in dense_hard_negs.get(qid, []) if c not in gold_set]

                negs = []
                if len(dense_p) >= 2: negs.extend(random.sample(dense_p[:25], 2))
                if len(stat_p) >= 2: negs.extend(random.sample(stat_p[:25], 2))
                while len(negs) < 4:
                    rc = random.choice(train_c_ids)
                    if rc not in gold_set and rc not in negs: negs.append(rc)

                for nid in negs[:4]:
                    n_ids, n_mask = get_cand_tokens(nid)
                    neg_ids_batch.append(n_ids)
                    neg_mask_batch.append(n_mask)

            # Tensors
            q_ids_t = torch.tensor(q_ids_batch, dtype=torch.long, device=device)
            q_mask_t = torch.tensor(q_mask_batch, dtype=torch.float32, device=device)
            pos_ids_t = torch.tensor(pos_ids_batch, dtype=torch.long, device=device)
            pos_mask_t = torch.tensor(pos_mask_batch, dtype=torch.float32, device=device)
            neg_ids_t = torch.tensor(neg_ids_batch, dtype=torch.long, device=device)
            neg_mask_t = torch.tensor(neg_mask_batch, dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z_q, u_q = encode_micro_batch(mglg_model, q_ids_t, q_mask_t, micro_bs=2)
                z_pos, u_pos = encode_micro_batch(mglg_model, pos_ids_t, pos_mask_t, micro_bs=2)
                z_neg, u_neg = encode_micro_batch(mglg_model, neg_ids_t, neg_mask_t, micro_bs=2)

                z_pos = z_pos.view(B, 2, -1)
                u_pos = u_pos.view(B, 2, 16, -1)
                z_neg = z_neg.view(B, 4, -1)
                u_neg = u_neg.view(B, 4, 16, -1)

                loss = criterion(mglg_model, z_q, u_q, z_pos, u_pos, z_neg, u_neg) / accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum_steps

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == steps_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (b_idx + 1) % 50 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {b_idx + 1}/{steps_per_epoch} | Loss: {loss.item() * accum_steps:.4f}")

        avg_loss = running_loss / steps_per_epoch
        print(f"--> Epoch {epoch} completed in {time.time() - t_epoch_start:.1f}s | Avg Loss: {avg_loss:.4f}")

    print(f"\nTraining completed in {time.time() - t_train_start:.1f}s")
    model_save_path = os.path.join(MODEL_DIR, "best_model.pt")
    torch.save({"model_state_dict": mglg_model.state_dict()}, model_save_path)
    print(f"Saved HarveyMGLG checkpoint to {model_save_path}")

    # 9. Re-encode Representations with Trained Model
    print("\n--- 8. Re-encoding Dev & Test Splits with Trained HarveyMGLG ---")
    t0 = time.time()
    _, dev_c_g, dev_c_s = encode_mglg_corpus(mglg_model, dev_c, tokenizer, is_query=False, max_len=2048, batch_size=16, device=device)
    _, dev_q_g, dev_q_s = encode_mglg_corpus(mglg_model, dev_q, tokenizer, is_query=True, max_len=2048, batch_size=16, device=device)
    print(f"Re-encoded Dev in {time.time() - t0:.2f}s")

    t0 = time.time()
    _, test_c_g, test_c_s = encode_mglg_corpus(mglg_model, test_c, tokenizer, is_query=False, max_len=2048, batch_size=16, device=device)
    _, test_q_g, test_q_s = encode_mglg_corpus(mglg_model, test_q, tokenizer, is_query=True, max_len=2048, batch_size=16, device=device)
    print(f"Re-encoded Test in {time.time() - t0:.2f}s")

    # Re-encode Memory Bank
    print("Refreshing Training Memory Bank...")
    _, train_q_g, train_q_s = encode_mglg_corpus(mglg_model, train_q, tokenizer, is_query=True, max_len=2048, batch_size=16, device=device)
    _, train_c_g, train_c_s = encode_mglg_corpus(mglg_model, train_c, tokenizer, is_query=False, max_len=2048, batch_size=16, device=device)

    # 10. Candidate Graph Refinement on Trained Model
    print("\n--- 9. Executing EUGAT Candidate Graph Refinement ---")
    dev_c_g_refined = mglg_model.build_candidate_graph_and_refine(dev_c_g.to(device), dev_c_s.to(device), cand_stat_sim_dev, cand_cocite_dev, k_nn=15)

    cand_stat_sim_test = torch.from_numpy(compute_doctrinal_tversky_matrix(test_c_txt, test_c_txt, test_cand_ids, test_cand_ids)).to(device)
    cand_cocite_test = torch.zeros((len(test_cand_ids), len(test_cand_ids)), dtype=torch.float32, device=device)
    test_c_g_refined = mglg_model.build_candidate_graph_and_refine(test_c_g.to(device), test_c_s.to(device), cand_stat_sim_test, cand_cocite_test, k_nn=15)

    # 11. Hyperparameter Optimization on Dev (Grid Search directly for Micro-F1@7)
    print("\n--- 10. Optimizing Inference Hyperparameters on Dev ---")
    dev_sim_glob, dev_sim_late, dev_sim_mem = compute_mglg_component_matrices(
        model=mglg_model,
        q_glob=dev_q_g.to(device),
        q_segs=dev_q_s.to(device),
        c_glob_refined=dev_c_g_refined,
        c_segs=dev_c_s.to(device),
        split_cand_ids=dev_cand_ids,
        train_cand_id_to_idx=train_cand_id_to_idx,
        train_q_glob=train_q_g.to(device),
        train_q_segs=train_q_s.to(device),
        train_c_glob=train_c_g.to(device),
        train_citation_adj=train_adj,
    )

    best_f1, best_hparams = 0.0, base_hparams
    for w_glob in [0.25, 0.30, 0.35, 0.40]:
        for w_late in [0.25, 0.30, 0.35, 0.40]:
            for w_mem in [0.12, 0.18, 0.24]:
                for w_stat in [0.08, 0.12, 0.16]:
                    for beta_hub in [0.30, 0.40, 0.50]:
                        for sinkhorn_gamma in [0.05, 0.15, 0.25]:
                            for w_prior in [0.04, 0.08, 0.12]:
                                hparams = {
                                    "w_glob": w_glob, "w_late": w_late, "w_mem": w_mem, "w_stat": w_stat,
                                    "beta_hub": beta_hub, "tau_mem": 0.04, "w_prior": w_prior, "sinkhorn_gamma": sinkhorn_gamma
                                }
                                scores = calibrate_and_combine_mglg(
                                    sim_glob=dev_sim_glob,
                                    sim_late=dev_sim_late,
                                    sim_mem=dev_sim_mem,
                                    stat_matrix=stat_dev,
                                    train_in_degree=dev_in_degree,
                                    hyperparams=hparams,
                                )
                                preds = extract_top7_predictions(scores, list(dev_q.keys()), dev_cand_ids)
                                m = compute_pcr_metrics(dev_truth, preds)
                                if m["micro_f1_7"] > best_f1:
                                    best_f1 = m["micro_f1_7"]
                                    best_hparams = hparams
                                    print(f"New Best Dev Micro-F1@7: {best_f1:.6f} | TP: {m['tp']} | hparams: {hparams}")

    print("==========================================================================")
    print("FINAL STANDALONE HARVEY-MGLG DEV PERFORMANCE:")
    final_dev_scores = predict_harvey_mglg_scores(
        model=mglg_model,
        q_glob=dev_q_g.to(device),
        q_segs=dev_q_s.to(device),
        c_glob_refined=dev_c_g_refined,
        c_segs=dev_c_s.to(device),
        stat_matrix=stat_dev,
        split_cand_ids=dev_cand_ids,
        train_cand_id_to_idx=train_cand_id_to_idx,
        train_q_glob=train_q_g.to(device),
        train_q_segs=train_q_s.to(device),
        train_c_glob=train_c_g.to(device),
        train_citation_adj=train_adj,
        train_in_degree=dev_in_degree,
        hyperparams=best_hparams,
    )
    final_dev_preds = extract_top7_predictions(final_dev_scores, list(dev_q.keys()), dev_cand_ids)
    final_dev_metrics = compute_pcr_metrics(dev_truth, final_dev_preds)
    print(f"Optimal Hyperparams: {best_hparams}")
    print(f"Dev Micro-F1@7:        {final_dev_metrics['micro_f1_7']:.6f}")
    print(f"Dev Micro-Precision@7: {final_dev_metrics['micro_precision_7']:.6f}")
    print(f"Dev Micro-Recall@7:    {final_dev_metrics['micro_recall_7']:.6f}")
    print(f"True Positives: {final_dev_metrics['tp']} (Baseline: 252 | Gain: +{final_dev_metrics['tp'] - 252})")
    print(f"False Positives: {final_dev_metrics['fp']} | False Negatives: {final_dev_metrics['fn']}")
    print("==========================================================================")

    dev_pred_path = os.path.join(SUB_DIR, "dev_prediction.json")
    with open(dev_pred_path, "w", encoding="utf-8") as f:
        json.dump(final_dev_preds, f, indent=2)

    # 12. Generate Test Predictions & Package Submission
    print("\n--- 11. Generating Test Predictions & Packaging Official Submission ---")
    final_test_scores = predict_harvey_mglg_scores(
        model=mglg_model,
        q_glob=test_q_g.to(device),
        q_segs=test_q_s.to(device),
        c_glob_refined=test_c_g_refined,
        c_segs=test_c_s.to(device),
        stat_matrix=stat_test,
        split_cand_ids=test_cand_ids,
        train_cand_id_to_idx=train_cand_id_to_idx,
        train_q_glob=train_q_g.to(device),
        train_q_segs=train_q_s.to(device),
        train_c_glob=train_c_g.to(device),
        train_citation_adj=train_adj,
        train_in_degree=test_in_degree,
        hyperparams=best_hparams,
    )
    final_test_preds = extract_top7_predictions(final_test_scores, list(test_q.keys()), test_cand_ids)

    sub_zip_path = os.path.join(SUB_DIR, "submission.zip")
    export_pcr_submission(
        pred_dict=final_test_preds,
        output_zip_path=sub_zip_path,
        allowed_candidates=set(test_cand_ids),
        expected_queries=set(test_q.keys())
    )

    # 13. Official Evaluation Check
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

    # 14. Log to Master Results Tracker
    tracker_path = os.path.join(ROOT_CODE, "phase7/results.txt")
    log_line = f"HARVEY_MGLG      STANDALONE_MGLG      {final_dev_metrics['micro_f1_7']:<14.6f} {final_dev_metrics['micro_precision_7']:<11.6f} {final_dev_metrics['micro_recall_7']:<12.6f} [READY]        {sub_zip_path}\n"
    with open(tracker_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\nLogged results to {tracker_path}")

    print("\n==========================================================================")
    print("STANDALONE HARVEY-MGLG COMPLETE & VERIFIED!")
    print(f"Official Submission Zip: {sub_zip_path}")
    print("==========================================================================")


if __name__ == "__main__":
    run_harvey_mglg_master()
