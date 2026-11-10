"""
Phase 7: Universal Legal Case Retriever supporting all 4 Phase 1 backbones:
1. HarveyLM (210M, ALiBi)
2. BaselineLM (242M, RoPE)
3. HarveyMoE (270M, Sparse MoE)
4. LegalEncoderDecoder (125M, Seq2Seq)

Supports:
- LORMA (Truncated SVD - Compulsory)
- LoRA
- QLoRA (4-bit NF4)
- Knowledge Distillation Loss (Teacher -> Student)
- InfoNCE Contrastive Loss with In-Batch Negatives
- Fast batch encoding and L2-normalized embedding extraction.
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

from phase1.src.model.baseline import BaselineLM, BaselineConfig, precompute_freqs_cis
from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig, build_alibi_tensor
from phase1.src.model.moe import HarveyMoE
from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from safetensors.torch import load_file
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase3.src.model.qlora_from_scratch import apply_qlora_to_model, mark_only_qlora_as_trainable
from phase2.src.model.lora_from_scratch import apply_lora_to_model, mark_only_lora_as_trainable

ARCH_CKPTS = {
    "baseline": os.path.join(ROOT_CODE, "phase1/models/final_architectures/1_Baseline/epoch_5"),
    "harveylm": os.path.join(ROOT_CODE, "phase1/models/final_architectures/2_HarveyLM/epoch_99"),
    "moe":      os.path.join(ROOT_CODE, "phase1/models/final_architectures/3_HarveyMoE/epoch_10"),
    "enc_dec":  os.path.join(ROOT_CODE, "phase1/models/final_architectures/4_EncoderDecoder/epoch_10")
}

ARCH_TARGETS = {
    "baseline": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "harveylm": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "moe":      ["wq", "wk", "wv", "wo"],
    "enc_dec":  ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
}


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss with in-batch negatives."""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_embed: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        # q_embed: [B, D], c_embed: [B, D] (both L2 normalized)
        sim_matrix = torch.matmul(q_embed, c_embed.T) / self.temperature  # [B, B]
        labels = torch.arange(q_embed.size(0), device=q_embed.device)
        loss_q2c = F.cross_entropy(sim_matrix, labels)
        loss_c2q = F.cross_entropy(sim_matrix.T, labels)
        return 0.5 * (loss_q2c + loss_c2q)


class RetrievalKDLoss(nn.Module):
    """Knowledge Distillation loss transferring similarity distribution from Teacher to Student."""
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.infonce = InfoNCELoss(temperature=0.05)

    def forward(
        self,
        student_q: torch.Tensor,
        student_c: torch.Tensor,
        teacher_q: torch.Tensor,
        teacher_c: torch.Tensor
    ) -> torch.Tensor:
        hard_loss = self.infonce(student_q, student_c)

        # Soft teacher distribution over in-batch candidates
        sim_s = torch.matmul(student_q, student_c.T) / self.temperature
        sim_t = torch.matmul(teacher_q, teacher_c.T) / self.temperature

        log_p_s = F.log_softmax(sim_s, dim=-1)
        p_t = F.softmax(sim_t.detach(), dim=-1)

        kl_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        return (1.0 - self.alpha) * hard_loss + self.alpha * kl_loss


class UniversalLegalRetriever(nn.Module):
    """
    Universal Legal Case Retriever supporting all 4 Phase 1 backbones and PEFT adaptations.
    """
    def __init__(
        self,
        arch_type: str = "harveylm",
        ckpt_dir: Optional[str] = None,
        adaptation: str = "lorma",
        r: int = 32,
        alpha: int = 64,
        temperature: float = 0.05,
        kd_temperature: float = 2.0,
        kd_alpha: float = 0.5
    ):
        super().__init__()
        self.arch_type = arch_type.lower()
        self.adaptation = adaptation.lower()
        self.temperature = temperature

        if ckpt_dir is None:
            ckpt_dir = ARCH_CKPTS.get(self.arch_type)
        self.ckpt_dir = ckpt_dir

        # Initialize backbone
        if self.arch_type in ["enc_dec", "encoder_decoder", "legal_enc_dec"]:
            self.config = EncDecConfig.from_pretrained(ckpt_dir)
            self.backbone = LegalEncoderDecoder(self.config)
            self.hidden_dim = self.config.d_model
        elif self.arch_type == "harveylm":
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyLM(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type in ["moe", "harveymoe"]:
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyMoE(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type == "baseline":
            self.config = BaselineConfig.from_pretrained(ckpt_dir)
            self.backbone = BaselineLM(self.config)
            self.hidden_dim = self.config.dim
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

        # Load weights
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            st = load_file(safetensor_path)
            self.backbone.load_state_dict(st, strict=False)
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(st, strict=False)

        # Apply Adaptation Strategy
        targets = ARCH_TARGETS.get(self.arch_type, ["wq", "wk", "wv", "wo"])
        if self.adaptation == "lorma":
            apply_lorma_to_model(self.backbone, r=r, lorma_alpha=alpha, target_modules=targets)
            mark_only_lorma_as_trainable(self.backbone)
        elif self.adaptation == "lora":
            apply_lora_to_model(self.backbone, r=r, lora_alpha=alpha, target_modules=targets)
            mark_only_lora_as_trainable(self.backbone)
        elif self.adaptation == "qlora":
            apply_qlora_to_model(self.backbone, r=r, lora_alpha=alpha, target_modules=targets)
            mark_only_qlora_as_trainable(self.backbone)
        elif self.adaptation in ["kd", "distillation"]:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif self.adaptation == "partial":
            # Train only top 4 layers + norm
            for p in self.backbone.parameters():
                p.requires_grad = False
            if hasattr(self.backbone, "layers"):
                for layer in self.backbone.layers[-4:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
        elif self.adaptation == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True

        # Projection head to normalized embedding space
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.kd_loss = RetrievalKDLoss(temperature=kd_temperature, alpha=kd_alpha)

    def forward_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extracts contextual representations across the sequence."""
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
                h, _ = layer(h, freqs_cis, mask, alibi, None, False)
            return self.backbone.norm(h)

        elif self.arch_type == "baseline":
            h = self.backbone.tok_embeddings(input_ids)
            freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device)
            mask = None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()
            for layer in self.backbone.layers:
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
                h = layer(h, enc_mask)
            return h

        raise ValueError(f"Unknown arch: {self.arch_type}")

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes L2-normalized document embedding via attention-masked mean pooling."""
        features = self.forward_features(input_ids, attention_mask=attention_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = features.mean(dim=1)

        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        c_ids: torch.Tensor,
        c_mask: torch.Tensor,
        teacher_model: Optional["UniversalLegalRetriever"] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_embed = self.encode(q_ids, q_mask)
        c_embed = self.encode(c_ids, c_mask)

        if teacher_model is not None:
            with torch.no_grad():
                t_q_embed = teacher_model.encode(q_ids, q_mask)
                t_c_embed = teacher_model.encode(c_ids, c_mask)
            loss = self.kd_loss(q_embed, c_embed, t_q_embed, t_c_embed)
        else:
            loss = self.infonce_loss(q_embed, c_embed)

        return loss, q_embed, c_embed
