"""
Phase 4: Court Judgment Prediction (CJPE) Model Architecture Module.

Implements:
1. LogitAdjustedLoss: Bayes-optimal loss under training prior shift (Menon et al., ICLR 2021).
2. HarveyHeadTailClassifier: Dense Head-Tail context encoder with multi-query attentive pooling.
3. HarveyHierarchicalClassifier: Three-stage Hierarchical Chunk-Transformer with multi-query bottleneck:
   - Stage 1: Intra-Chunk Attention Pooling
   - Stage 2: 2-Layer Bidirectional Cross-Chunk Transformer
   - Stage 3: 4-Query Cross-Attention Bottleneck (Q-Former style)
   - Stage 4: 2-Layer SwiGLU Classification Projection
4. Compatible with compulsory LORMA (Truncated SVD from scratch) and LoRA.
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

try:
    from phase1.src.model.baseline import BaselineLM, BaselineConfig
    from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig
    from phase1.src.model.moe import HarveyMoE
    from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
except ImportError:
    from src.model.baseline import BaselineLM, BaselineConfig
    from src.model.architecture import HarveyLM, HarveyTransformerConfig
    from src.model.moe import HarveyMoE
    from src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
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


class LogitAdjustedLoss(nn.Module):
    """
    Logit-Adjusted Loss for Long-Tail Learning (Menon et al., ICLR 2021).
    Shifts output logits by tau * ln(pi_y) inside the loss function.
    At inference time, raw logits directly estimate the prior-free log-likelihood ratio,
    making them optimal for a 50/50 balanced test set.
    """
    def __init__(self, class_priors: List[float] = [0.5857, 0.4143], tau: float = 1.0, label_smoothing: float = 0.05):
        super().__init__()
        self.tau = tau
        self.label_smoothing = label_smoothing
        priors_tensor = torch.tensor(class_priors, dtype=torch.float32)
        log_priors = torch.log(priors_tensor.clamp(min=1e-6))
        self.register_buffer("log_priors", log_priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 2]
        # targets: [B]
        adjusted_logits = logits + self.tau * self.log_priors.to(logits.device).unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


class KnowledgeDistillationLoss(nn.Module):
    """
    Task-specific Knowledge Distillation Loss (Hinton et al., 2015).
    Combines hard label logit-adjusted task loss with soft label KL divergence.
    """
    def __init__(self, temperature: float = 2.5, alpha: float = 0.50, tau_la: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_criterion = LogitAdjustedLoss(tau=tau_la)

    def forward(self, student_logits: torch.Tensor, targets: Optional[torch.Tensor], teacher_logits: torch.Tensor) -> torch.Tensor:
        hard_loss = self.hard_criterion(student_logits, targets) if targets is not None and targets.min() >= 0 else torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is None:
            return hard_loss
        t = self.temperature
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)
        return (1.0 - self.alpha) * hard_loss + self.alpha * kl_loss


class MultiQueryAttentionPooling(nn.Module):
    """
    Multi-Query Attentive Pooling (Q-Former style).
    K learnable aspect queries attend across variable-length token representations.
    """
    def __init__(self, hidden_dim: int = 1024, num_queries: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.norm_keys = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, seq_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # seq_states: [B, seq_len, hidden_dim]
        # key_padding_mask: [B, seq_len] (True for padded positions if boolean, or 0/1 mask)
        bsz = seq_states.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1) # [B, K, hidden_dim]

        keys = self.norm_keys(seq_states)

        # PyTorch MultiheadAttention key_padding_mask expects True on positions to be ignored
        bool_pad_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                bool_pad_mask = (key_padding_mask == 0)
            else:
                bool_pad_mask = key_padding_mask

        attn_out, _ = self.cross_attn(query=queries, key=keys, value=keys, key_padding_mask=bool_pad_mask)
        out = self.norm_out(queries + attn_out) # [B, K, hidden_dim]
        return out.reshape(bsz, self.num_queries * self.hidden_dim) # [B, K * hidden_dim]


class SwiGLUHead(nn.Module):
    """2-layer SwiGLU Classification Head matching HarveyLM's FFN structure."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.15):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        h = F.silu(self.w1(x)) * self.w2(x)
        h = self.dropout(h)
        return self.w_out(h)


class HarveyHeadTailClassifier(nn.Module):
    """
    Dense Head-Tail HarveyLM Classifier.
    Encodes fused 2048-token context through HarveyLM and aggregates with Multi-Query Pooling.
    """
    def __init__(
        self,
        ckpt_dir: str,
        num_classes: int = 2,
        num_queries: int = 4,
        head_hidden: int = 1024,
        dropout: float = 0.15,
        tau_la: float = 1.0
    ):
        super().__init__()
        print(f"Loading HarveyLM backbone from {ckpt_dir}...")
        self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
        self.backbone = HarveyLM(self.config)
        self.hidden_dim = self.config.dim

        # Load weights from safetensors or bin
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            st = load_file(safetensor_path)
            self.backbone.load_state_dict(st, strict=False)
            print("Successfully loaded backbone weights from model.safetensors")
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu")
            self.backbone.load_state_dict(st, strict=False)
            print("Successfully loaded backbone weights from pytorch_model.bin")

        self.backbone.gradient_checkpointing = True

        # Multi-Query Attention Pooling (K=4 aspect queries)
        self.pooler = MultiQueryAttentionPooling(hidden_dim=self.hidden_dim, num_queries=num_queries, dropout=dropout)

        # Classification Head
        self.classifier = SwiGLUHead(
            in_dim=num_queries * self.hidden_dim,
            hidden_dim=head_hidden,
            out_dim=num_classes,
            dropout=dropout
        )

        # Logit-Adjusted Loss
        self.criterion = LogitAdjustedLoss(class_priors=[0.5857, 0.4143], tau=tau_la, label_smoothing=0.05)

    def forward_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        h = self.backbone.tok_embeddings(input_ids)

        if self.backbone.config.use_alibi:
            # Symmetrical ALiBi for bidirectional document representation (Press et al., 2022)
            alibi = -torch.abs(self.backbone.alibi[:, :seqlen, :seqlen].to(device=h.device, dtype=h.dtype))
        else:
            alibi = None
        freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device) if not self.backbone.config.use_alibi else None

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()

        for layer in self.backbone.layers:
            if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                h, _ = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False)
            else:
                h, _ = layer(h, freqs_cis, mask, alibi, None, False)

        return self.backbone.norm(h) # [B, seqlen, hidden_dim]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(input_ids, attention_mask)
        pooled = self.pooler(features, key_padding_mask=attention_mask)
        logits = self.classifier(pooled) # [B, 2]

        loss = None
        if labels is not None and labels.min() >= 0:
            loss = self.criterion(logits, labels)

        return {
            "logits": logits,
            "loss": loss
        }


class UniversalHeadTailClassifier(nn.Module):
    """
    Universal Head-Tail Legal Classifier supporting all 4 Phase 1 Architectures:
    1. HarveyLM (Custom ALiBi, SwiGLU, 16 layers, d=1024)
    2. BaselineLM (Custom LLaMA RoPE, 16 layers, d=1024)
    3. HarveyMoE (Top-2 Sparse MoE, 12 layers, d=768)
    4. LegalEncoderDecoder (6 enc + 6 dec layers, d=768)
    """
    def __init__(
        self,
        arch_type: str = "harveylm",
        ckpt_dir: Optional[str] = None,
        num_classes: int = 2,
        num_queries: int = 4,
        head_hidden: int = 1024,
        dropout: float = 0.15,
        tau_la: float = 1.0,
        kd_temperature: float = 2.5,
        kd_alpha: float = 0.50
    ):
        super().__init__()
        self.arch_type = arch_type.lower()
        if ckpt_dir is None:
            ckpt_dir = ARCH_CKPTS[self.arch_type]
        self.ckpt_dir = ckpt_dir
        print(f"Instantiating UniversalHeadTailClassifier for arch='{self.arch_type}' from {ckpt_dir}...")

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
            print(f"Successfully loaded backbone weights from {safetensor_path}")
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(st, strict=False)
            print(f"Successfully loaded backbone weights from {bin_path}")

        self.backbone.gradient_checkpointing = True

        # Multi-Query Attention Pooling (K=4 aspect queries)
        self.pooler = MultiQueryAttentionPooling(hidden_dim=self.hidden_dim, num_queries=num_queries, dropout=dropout)

        # Classification Head (SwiGLU)
        self.classifier = SwiGLUHead(
            in_dim=num_queries * self.hidden_dim,
            hidden_dim=head_hidden,
            out_dim=num_classes,
            dropout=dropout
        )

        # Logit-Adjusted Loss and KD Loss
        self.criterion = LogitAdjustedLoss(class_priors=[0.5857, 0.4143], tau=tau_la, label_smoothing=0.05)
        self.kd_loss = KnowledgeDistillationLoss(temperature=kd_temperature, alpha=kd_alpha, tau_la=tau_la)

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
                enc_mask = (1.0 - attention_mask) * -10000.0
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
        elif labels is not None and labels.min() >= 0:
            loss = self.criterion(logits, labels)

        return {
            "logits": logits,
            "loss": loss
        }


class CrossChunkTransformerLayer(nn.Module):
    """Bidirectional Transformer Block for Cross-Chunk Reasoning."""
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class HarveyHierarchicalClassifier(nn.Module):
    """
    Role-Salient Hierarchical Chunk-Transformer for Extra-Long Judgments.
    Stages:
      1. Intra-Chunk HarveyLM Forward (M chunks x 1024 tokens)
      2. 2-Layer Bidirectional Cross-Chunk Transformer
      3. 4-Query Cross-Attention Bottleneck
      4. SwiGLU Binary Classification
    """
    def __init__(
        self,
        ckpt_dir: str,
        num_classes: int = 2,
        max_chunks: int = 16,
        num_queries: int = 4,
        num_cross_layers: int = 2,
        head_hidden: int = 1024,
        dropout: float = 0.15,
        tau_la: float = 1.0
    ):
        super().__init__()
        print(f"Loading HarveyLM backbone for Hierarchical Classifier from {ckpt_dir}...")
        self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
        self.backbone = HarveyLM(self.config)
        self.hidden_dim = self.config.dim
        self.max_chunks = max_chunks

        # Load weights
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            st = load_file(safetensor_path)
            self.backbone.load_state_dict(st, strict=False)
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu")
            self.backbone.load_state_dict(st, strict=False)

        self.backbone.gradient_checkpointing = True

        # Stage 1: Intra-Chunk Attention Scoring
        self.intra_chunk_attn = nn.Linear(self.hidden_dim, 1, bias=False)

        # Stage 2: Learned Chunk Position Embeddings + Cross-Chunk Transformer
        self.chunk_pos_embed = nn.Embedding(max_chunks, self.hidden_dim)
        self.cross_chunk_layers = nn.ModuleList([
            CrossChunkTransformerLayer(hidden_dim=self.hidden_dim, num_heads=8, ffn_dim=2048, dropout=dropout)
            for _ in range(num_cross_layers)
        ])
        self.cross_norm = nn.LayerNorm(self.hidden_dim)

        # Stage 3: Multi-Query Bottleneck Aggregation
        self.pooler = MultiQueryAttentionPooling(hidden_dim=self.hidden_dim, num_queries=num_queries, dropout=dropout)

        # Stage 4: Classification Projection
        self.classifier = SwiGLUHead(
            in_dim=num_queries * self.hidden_dim,
            hidden_dim=head_hidden,
            out_dim=num_classes,
            dropout=dropout
        )

        self.criterion = LogitAdjustedLoss(class_priors=[0.5857, 0.4143], tau=tau_la, label_smoothing=0.05)

    def encode_single_chunk(self, chunk_ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        # chunk_ids: [Total_Chunks, chunk_len]
        bsz, seqlen = chunk_ids.shape
        h = self.backbone.tok_embeddings(chunk_ids)

        if self.backbone.config.use_alibi:
            # Symmetrical ALiBi for bidirectional document representation (Press et al., 2022)
            alibi = -torch.abs(self.backbone.alibi[:, :seqlen, :seqlen].to(device=h.device, dtype=h.dtype))
        else:
            alibi = None
        freqs_cis = self.backbone.freqs_cis[:seqlen].to(h.device) if not self.backbone.config.use_alibi else None

        mask = chunk_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen).bool()
        for layer in self.backbone.layers:
            if getattr(self.backbone, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                h, _ = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False)
            else:
                h, _ = layer(h, freqs_cis, mask, alibi, None, False)
        h = self.backbone.norm(h) # [Total_Chunks, chunk_len, hidden_dim]

        # Intra-Chunk Attention Pooling blended with final token representation (RESEARCH.md 2.C)
        attn_scores = self.intra_chunk_attn(h).squeeze(-1) # [Total_Chunks, chunk_len]
        attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e4)
        weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        attentive_pool = (h * weights).sum(dim=1) # [Total_Chunks, hidden_dim]
        last_tok = h[:, -1, :] # [Total_Chunks, hidden_dim]
        pooled_chunk = 0.5 * attentive_pool + 0.5 * last_tok
        return pooled_chunk

    def forward(
        self,
        chunk_input_ids: torch.Tensor,    # [B, M, chunk_len]
        chunk_masks: torch.Tensor,        # [B, M, chunk_len]
        doc_chunk_masks: torch.Tensor,    # [B, M] (1.0 for valid chunk, 0.0 for padded chunk)
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, M, L_c = chunk_input_ids.shape

        # Flatten chunks for batched backbone forward pass
        flat_ids = chunk_input_ids.view(B * M, L_c)
        flat_masks = chunk_masks.view(B * M, L_c)

        # Encode chunks
        flat_chunk_embs = self.encode_single_chunk(flat_ids, flat_masks) # [B * M, hidden_dim]
        chunk_embs = flat_chunk_embs.view(B, M, self.hidden_dim)         # [B, M, hidden_dim]

        # Stage 2: Add Chunk Position Embeddings & pass through Cross-Chunk Transformer
        positions = torch.arange(M, device=chunk_embs.device).clamp(max=self.max_chunks - 1)
        chunk_embs = chunk_embs + self.chunk_pos_embed(positions).unsqueeze(0)

        cross_key_padding = (doc_chunk_masks == 0.0) # True for ignored chunks
        for layer in self.cross_chunk_layers:
            chunk_embs = layer(chunk_embs, key_padding_mask=cross_key_padding)
        chunk_embs = self.cross_norm(chunk_embs)

        # Stage 3: Multi-Query Aspect Aggregation
        doc_vector = self.pooler(chunk_embs, key_padding_mask=cross_key_padding)

        # Stage 4: Classification
        logits = self.classifier(doc_vector)

        loss = None
        if labels is not None and labels.min() >= 0:
            loss = self.criterion(logits, labels)

        return {
            "logits": logits,
            "loss": loss
        }
