"""
Phase 3 - HarveyLM V4 for Rhetorical Role Prediction:
Attentive Token Pooling + Multi-Scale Representation Head + Focal Loss.

Key Architectural Innovations:
1. Target-Sentence Span Isolation:
   Tokens attend bidirectionally with Symmetric ALiBi over [PREV] + [TARGET] + [NEXT].
2. Attentive Token Pooling Head:
   Instead of uniform mean-pooling, uses a 4-head self-attention mechanism over target tokens,
   penalizing common grammatical stopwords and punctuation.
3. Multi-View Representation:
   Concatenates Attentive Pooled (1024-d) + Max-Over-Time (1024-d) + Mean Pooled (1024-d)
   + Fourier Document Depth (128-d) + Domain Indicator (32-d) -> 3232-d representation.
4. Class-Balanced Focal Loss (gamma=2.0) with adaptive class-frequency weighting.
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))
from phase2.src.model.architecture import HarveyLM, HarveyTransformerConfig
from phase3.src.model.rr_harveylm_v3 import DocumentPositionEncoder, ClassBalancedFocalLoss
from phase3.src.model.attentive_token_head import AttentiveTokenPoolingHead


class HarveyForRhetoricalRoleV4(nn.Module):
    """
    HarveyLM V4 with Attentive Token-Level Representation Learning.
    """
    def __init__(self, config: HarveyTransformerConfig, num_labels: int,
                 pos_dim: int = 128, domain_dim: int = 32, num_attn_heads: int = 4,
                 hidden_dropout: float = 0.15, class_weights: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.num_labels = num_labels
        self.config     = config
        self.pos_dim    = pos_dim
        self.domain_dim = domain_dim

        self.harvey     = HarveyLM(config)
        self.pos_enc    = DocumentPositionEncoder(pos_dim)
        self.domain_enc = nn.Embedding(2, domain_dim)

        self.token_head = AttentiveTokenPoolingHead(
            embed_dim=config.dim,
            attn_dim=256,
            num_heads=num_attn_heads,
            pos_dim=pos_dim,
            domain_dim=domain_dim,
            num_labels=num_labels,
            dropout=hidden_dropout
        )

        self.loss_fn = ClassBalancedFocalLoss(gamma=gamma, weight=class_weights, ignore_index=-1)

    def extract_token_features(self, input_ids, attention_mask=None, target_mask=None,
                               rel_pos=None, domain_id=None, stopword_mask=None):
        """Extracts token hidden representations and multi-view pooled sentence vectors."""
        outputs = self.harvey(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_causal_mask=False
        )
        hidden_states = outputs.hidden_states[0] # (B, L, D)

        pos_feat = self.pos_enc(rel_pos) if rel_pos is not None else torch.zeros(input_ids.size(0), self.pos_dim, device=input_ids.device)
        dom_feat = self.domain_enc(domain_id.squeeze(-1)) if domain_id is not None else torch.zeros(input_ids.size(0), self.domain_dim, device=input_ids.device)

        logits, fused_rep, attn_weights = self.token_head(
            hidden_states=hidden_states,
            target_mask=target_mask,
            pos_feat=pos_feat,
            dom_feat=dom_feat,
            stopword_mask=stopword_mask
        )

        return logits, fused_rep, attn_weights

    def forward(self, input_ids, attention_mask=None, target_mask=None,
                rel_pos=None, domain_id=None, stopword_mask=None, labels=None):
        logits, fused_rep, attn_weights = self.extract_token_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=target_mask,
            rel_pos=rel_pos,
            domain_id=domain_id,
            stopword_mask=stopword_mask
        )

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits, "features": fused_rep, "attn_weights": attn_weights}

    @classmethod
    def from_pretrained_harvey(cls, checkpoint_dir: str, num_labels: int,
                               pos_dim: int = 128, domain_dim: int = 32, num_attn_heads: int = 4,
                               hidden_dropout: float = 0.15, class_weights: torch.Tensor = None, gamma: float = 2.0):
        config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
        model = cls(config, num_labels=num_labels, pos_dim=pos_dim, domain_dim=domain_dim,
                    num_attn_heads=num_attn_heads, hidden_dropout=hidden_dropout,
                    class_weights=class_weights, gamma=gamma)

        safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            from safetensors.torch import load_file
            state = load_file(safetensor_path)
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_dir}")

        missing, unexpected = model.harvey.load_state_dict(state, strict=False)
        print(f"HarveyLM V4 backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")
        return model
