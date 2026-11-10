"""
Phase 5: Universal Multi-Label Classifier for Legal Statute Identification (LSI).

Supports:
1. All 4 foundational backbones:
   - HarveyLM (Custom ALiBi, 219M)
   - BaselineLM (LLaMA RoPE, 217M)
   - HarveyMoE (Top-2 Mixture of Experts, 420M - with router dynamically trainable)
   - LegalEncoderDecoder (125M)
2. Adaptation strategies:
   - LORMA (Truncated SVD Low-Rank Matrix Approximation, Compulsory)
   - LoRA (Standard Low-Rank Adaptation)
   - QLoRA (4-bit NF4 Quantization + LoRA)
   - Knowledge Distillation (Student distillation from teacher soft logits)
   - Full Fine-Tuning
3. Asymmetric Loss (ASL) for extreme multi-label long-tail class imbalance.
4. Multi-Query Attention Pooling (K=4 aspect queries) over dense 2,048-token Head-Tail contexts.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)
CODE_DIR = ROOT_CODE

from phase1.src.model.baseline import BaselineLM, BaselineConfig
from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig
from phase1.src.model.moe import HarveyMoE
from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from safetensors.torch import load_file
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase3.src.model.qlora_from_scratch import apply_qlora_to_model, mark_only_qlora_as_trainable
from phase2.src.model.lora_from_scratch import apply_lora_to_model, mark_only_lora_as_trainable

ARCH_CKPTS = {
    "baseline": os.path.join(CODE_DIR, "phase1/models/final_architectures/1_Baseline/epoch_5"),
    "harveylm": os.path.join(CODE_DIR, "phase1/models/final_architectures/2_HarveyLM/epoch_99"),
    "moe":      os.path.join(CODE_DIR, "phase1/models/final_architectures/3_HarveyMoE/epoch_10"),
    "enc_dec":  os.path.join(CODE_DIR, "phase1/models/final_architectures/4_EncoderDecoder/epoch_10")
}

ARCH_TARGETS = {
    "baseline": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "harveylm": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "moe":      ["wq", "wk", "wv", "wo"],
    "enc_dec":  ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
}


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification (Ridnik et al., ICCV 2021).
    Decouples positive and negative focusing. Suppresses easy negative gradients
    to prevent majority negative classes from swamping rare statute learning.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 100], targets: [B, 100] (binary 0/1)
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Margin Shifting on negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Log losses
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                pt0 = xs_pos * targets
                pt1 = xs_neg * (1.0 - targets)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
                loss *= one_sided_w
            else:
                pt1 = xs_neg * (1.0 - targets)
                one_sided_gamma = self.gamma_neg * (1.0 - targets)
                one_sided_w = torch.pow(1.0 - pt1, one_sided_gamma)
                loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


class MultiLabelKDLoss(nn.Module):
    """
    Knowledge Distillation Loss for Multi-Label LSI.
    Combines hard-label Asymmetric Loss with soft-label sigmoid KL/BCE distillation.
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.50):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)

    def forward(self, student_logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        hard_loss = self.asl(student_logits, targets)

        # Soft distillation over 100 binary sigmoid dimensions
        s_scaled = student_logits / self.temperature
        t_scaled = (teacher_logits / self.temperature).detach()

        t_prob = torch.sigmoid(t_scaled)
        soft_loss = F.binary_cross_entropy_with_logits(s_scaled, t_prob, reduction="mean") * (self.temperature ** 2)

        return self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss


class MultiQueryAttentionPooling(nn.Module):
    """Multi-Query Cross-Attention Pooling to extract K aspect features from long document tokens."""
    def __init__(self, hidden_dim: int, num_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: [B, S, D]
        # key_padding_mask: [B, S] (True for valid tokens, or False for padding)
        bsz = hidden_states.size(0)
        queries = self.query_tokens.expand(bsz, -1, -1)

        attn_mask = None
        if key_padding_mask is not None:
            # MultiheadAttention takes key_padding_mask where True indicates value to IGNORE
            attn_mask = ~key_padding_mask

        pooled, _ = self.mha(queries, hidden_states, hidden_states, key_padding_mask=attn_mask)
        pooled = self.norm(pooled)
        return pooled.reshape(bsz, -1)  # [B, K * D]


class SwiGLUHead(nn.Module):
    """SwiGLU Multi-Label Projection Head."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 100, dropout: float = 0.15):
        super().__init__()
        self.fc_gate = nn.Linear(in_dim, hidden_dim)
        self.fc_up   = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc_gate(x)) * self.fc_up(x)
        h = self.dropout(h)
        return self.fc_out(h)


class UniversalLSIClassifier(nn.Module):
    """
    Unified Multi-Label Legal Statute Identification Classifier.
    """
    def __init__(
        self,
        arch_type: str = "harveylm",
        ckpt_dir: Optional[str] = None,
        num_classes: int = 100,
        num_queries: int = 4,
        head_hidden: int = 1024,
        dropout: float = 0.15,
        kd_temperature: float = 2.0,
        kd_alpha: float = 0.50
    ):
        super().__init__()
        self.arch_type = arch_type.lower()
        if ckpt_dir is None:
            ckpt_dir = ARCH_CKPTS[self.arch_type]
        self.ckpt_dir = ckpt_dir
        self.num_classes = num_classes

        print(f"Instantiating UniversalLSIClassifier for arch='{self.arch_type}' from {ckpt_dir}...")

        if self.arch_type == "harveylm":
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyLM(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type == "baseline":
            self.config = BaselineConfig.from_pretrained(ckpt_dir)
            self.backbone = BaselineLM(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type in ["moe", "harveymoe"]:
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyMoE(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type in ["enc_dec", "encoder_decoder"]:
            self.config = EncDecConfig.from_pretrained(ckpt_dir)
            self.backbone = LegalEncoderDecoder(self.config)
            self.hidden_dim = self.config.d_model
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        # Load weights from safetensors or bin
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            st = load_file(safetensor_path)
            self.backbone.load_state_dict(st, strict=False)
            print(f"Loaded backbone weights from {safetensor_path}")
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(st, strict=False)
            print(f"Loaded backbone weights from {bin_path}")

        self.backbone.gradient_checkpointing = True

        # Multi-Query Attention Pooling
        self.pooler = MultiQueryAttentionPooling(hidden_dim=self.hidden_dim, num_queries=num_queries, dropout=dropout)

        # Classification Head (100 statute logits)
        self.classifier = SwiGLUHead(
            in_dim=num_queries * self.hidden_dim,
            hidden_dim=head_hidden,
            out_dim=num_classes,
            dropout=dropout
        )

        # Losses
        self.criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        self.kd_loss = MultiLabelKDLoss(temperature=kd_temperature, alpha=kd_alpha)

    def forward_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape

        if self.arch_type == "harveylm":
            h = self.backbone.tok_embeddings(input_ids)
            if self.backbone.config.use_alibi:
                alibi = -torch.abs(self.backbone.alibi[:, :seqlen, :seqlen].to(device=h.device, dtype=h.dtype))
            else:
                alibi = None
            freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device) if not self.backbone.config.use_alibi else None

            mask = None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()

            for layer in self.backbone.layers:
                if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                    def custom_fwd(m):
                        return lambda *args: m(*args)
                    h, _ = torch.utils.checkpoint.checkpoint(custom_fwd(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False)
                else:
                    h, _ = layer(h, freqs_cis, mask, alibi, None, False)
            return self.backbone.norm(h)

        elif self.arch_type == "baseline":
            h = self.backbone.tok_embeddings(input_ids)
            freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device)
            mask = None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()
            for layer in self.backbone.layers:
                if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                    def custom_fwd(m):
                        return lambda *args: m(*args)
                    h, _ = torch.utils.checkpoint.checkpoint(custom_fwd(layer), h, freqs_cis, mask, None, False, use_reentrant=False)
                else:
                    h, _ = layer(h, freqs_cis, mask, None, False)
            return self.backbone.norm(h)

        elif self.arch_type in ["moe", "harveymoe"]:
            h = self.backbone.tok_embeddings(input_ids)
            freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device) if not self.backbone.config.use_alibi else None
            alibi = -torch.abs(self.backbone.alibi[:, :seqlen, :seqlen].to(h.device)) if self.backbone.config.use_alibi else None
            mask = None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()
            for layer in self.backbone.layers:
                if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                    def custom_fwd(m):
                        return lambda *args: m(*args)
                    h, _ = torch.utils.checkpoint.checkpoint(custom_fwd(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False)
                else:
                    h, _ = layer(h, freqs_cis, mask, alibi, None, False)
            return self.backbone.norm(h)

        elif self.arch_type in ["enc_dec", "encoder_decoder"]:
            positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
            h = self.backbone.embedding(input_ids) + self.backbone.pos_encoder(positions)
            enc_mask = None
            if attention_mask is not None:
                enc_mask = (1.0 - attention_mask.float()) * -10000.0
                enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)
            for layer in self.backbone.encoders:
                if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                    def custom_fwd(m):
                        return lambda *args: m(*args)
                    h = torch.utils.checkpoint.checkpoint(custom_fwd(layer), h, enc_mask, use_reentrant=False)
                else:
                    h = layer(h, enc_mask)
            return self.backbone.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(input_ids, attention_mask)
        pooled = self.pooler(features, key_padding_mask=attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if teacher_logits is not None:
            loss = self.kd_loss(logits, labels, teacher_logits)
        elif labels is not None:
            loss = self.criterion(logits, labels)

        return {
            "logits": logits,
            "loss": loss
        }
