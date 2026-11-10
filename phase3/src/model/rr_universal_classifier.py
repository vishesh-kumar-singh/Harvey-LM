"""
Phase 3 - Universal Legal Classifier for All 4 Foundation Architectures.

Supports:
1. BaselineLM (Custom LLaMA style, 16 layers, d=1024)
2. HarveyLM (Custom ALiBi SwiGLU, 12 layers, d=1024)
3. HarveyMoE (MoE routing, 12 layers, d=768)
4. LegalEncoderDecoder (Encoder-Decoder, 6 enc + 6 dec, d=768)

Innovations:
- Target-Sentence Span Pooling: Bidirectional representation pooled strictly over target sentence tokens.
- Continuous Document Depth Fourier MLP (128-d) + Domain Indicator Embedding (32-d).
- Class-Balanced Focal Loss + Knowledge Distillation Soft Target Alignment.
- Compatible with custom LoRA, LORMA, and QLoRA.
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../phase1")))
from src.model.baseline import BaselineLM, BaselineConfig
from src.model.architecture import HarveyLM, HarveyTransformerConfig
from src.model.moe import HarveyMoE
from src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from safetensors.torch import load_file


class DocumentPositionEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

    def forward(self, pos: torch.Tensor):
        pi = math.pi
        w_dtype = self.mlp[0].weight.dtype
        feat = torch.cat([pos, torch.sin(pos * pi), torch.cos(pos * pi), pos ** 2], dim=-1).to(w_dtype)
        return self.mlp(feat)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        valid = targets != self.ignore_index
        return focal[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.5, alpha: float = 0.60, gamma: float = 2.0,
                 weight: torch.Tensor = None, ignore_index: int = -1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.focal_loss = ClassBalancedFocalLoss(gamma=gamma, weight=weight, ignore_index=ignore_index)

    def forward(self, student_logits: torch.Tensor, targets: torch.Tensor, teacher_probs_or_logits: torch.Tensor):
        # Hard label task loss
        hard_loss = self.focal_loss(student_logits, targets) if targets is not None else 0.0

        if teacher_probs_or_logits is None:
            return hard_loss

        # Soft label distillation loss
        t = self.temperature
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        if teacher_probs_or_logits.dim() == student_logits.dim() and teacher_probs_or_logits.shape == student_logits.shape:
            # Check if teacher input is probabilities or raw logits
            if (teacher_probs_or_logits >= 0).all() and (teacher_probs_or_logits.sum(dim=-1).sub(1.0).abs() < 1e-3).all():
                teacher_probs = teacher_probs_or_logits
            else:
                teacher_probs = F.softmax(teacher_probs_or_logits / t, dim=-1)
        else:
            teacher_probs = F.softmax(teacher_probs_or_logits / t, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)
        return (1.0 - self.alpha) * hard_loss + self.alpha * kl_loss


class UniversalLegalClassifier(nn.Module):
    """
    Universal Rhetorical Role Classifier adaptable to any of the 4 Phase 1 legal LLM architectures.
    """
    def __init__(self, arch_type: str, ckpt_dir: str, num_labels: int = 13,
                 pos_dim: int = 128, domain_dim: int = 32, hidden_dropout: float = 0.15,
                 class_weights: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.arch_type = arch_type.lower()
        self.num_labels = num_labels
        self.pos_dim = pos_dim
        self.domain_dim = domain_dim

        # Load Backbone
        if self.arch_type == "baseline":
            self.config = BaselineConfig.from_pretrained(ckpt_dir)
            self.backbone = BaselineLM(self.config)
            self.hidden_dim = self.config.dim
        elif self.arch_type == "harveylm":
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyLM(self.config)
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

        # Load weights from safetensors if exists
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(safetensor_path):
            sd = load_file(safetensor_path)
            self.backbone.load_state_dict(sd, strict=False)
            print(f"Loaded pretrained backbone weights from {safetensor_path}")

        # Task Heads
        self.pos_enc = DocumentPositionEncoder(pos_dim)
        self.domain_enc = nn.Embedding(2, domain_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        total_dim = self.hidden_dim + pos_dim + domain_dim
        self.pre_classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(hidden_dropout)
        )
        self.classifier = nn.Linear(512, num_labels)
        self.loss_fn = ClassBalancedFocalLoss(gamma=gamma, weight=class_weights, ignore_index=-1)
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature=2.5, alpha=0.60, gamma=gamma, weight=class_weights)

    def extract_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        bsz, seqlen = input_ids.shape

        if self.arch_type == "harveylm":
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_causal_mask=False
            )
            return outputs.hidden_states[0]

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
            alibi = self.backbone.alibi[:, :seqlen, :seqlen].to(h.device) if self.backbone.config.use_alibi else None
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
                enc_mask = (1.0 - attention_mask) * -10000.0
                enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)
            for layer in self.backbone.encoders:
                h = layer(h, enc_mask)
            return self.backbone.norm(h)

    def extract_features(self, input_ids, attention_mask=None, target_mask=None, rel_pos=None, domain_id=None):
        hidden_states = self.extract_hidden_states(input_ids, attention_mask=attention_mask)

        if target_mask is not None:
            m = target_mask.unsqueeze(-1).to(hidden_states.dtype)
            target_pooled = (hidden_states * m).sum(1) / m.sum(1).clamp(min=1e-9)
        elif attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            target_pooled = (hidden_states * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            target_pooled = hidden_states.mean(1)

        dt = target_pooled.dtype
        dev = target_pooled.device
        pos_feat = self.pos_enc(rel_pos).to(dt) if rel_pos is not None else torch.zeros(target_pooled.size(0), self.pos_dim, device=dev, dtype=dt)
        dom_feat = self.domain_enc(domain_id.squeeze(-1)).to(dt) if domain_id is not None else torch.zeros(target_pooled.size(0), self.domain_dim, device=dev, dtype=dt)

        full_rep = torch.cat([target_pooled, pos_feat, dom_feat], dim=-1)
        return target_pooled, full_rep

    def forward(self, input_ids, attention_mask=None, target_mask=None, rel_pos=None, domain_id=None,
                labels=None, teacher_targets=None):
        target_pooled, full_rep = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=target_mask,
            rel_pos=rel_pos,
            domain_id=domain_id
        )
        h = self.pre_classifier(full_rep)
        logits = self.classifier(h)

        loss = None
        if teacher_targets is not None:
            loss = self.kd_loss_fn(logits, labels, teacher_targets)
        elif labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits, "pooled": target_pooled, "features": full_rep}
