"""
Phase 3 - Enhanced HarveyLM for Rhetorical Role Prediction (V3).

Key Architectural Innovations:
1. Target-Sentence Span Pooling:
   Bidirectional attention over [PREV] + [EOS] + [TARGET] + [EOS] + [NEXT].
   Target tokens attend to surrounding context via symmetric ALiBi, but pooling
   extracts ONLY the target tokens, preventing context dilution.
2. Document Relative Position & Domain Modulation:
   Continuous document depth (i / N) mapped via Fourier MLP (128-d).
   Domain indicator (CL=0, IT=1) mapped via learnable embedding (32-d).
3. Class-Balanced Focal Loss (gamma=2.0):
   Amplifies gradients for rare but crucial legal classes (Issue, RulingByLowerCourt,
   PrecedentNotReliedUpon, RulingByPresentCourt).
4. Feature Extraction Interface:
   Direct extraction of task-specialized sentence representations for downstream Hierarchical CRF.
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../code")))
from phase2.src.model.architecture import HarveyLM, HarveyTransformerConfig


class DocumentPositionEncoder(nn.Module):
    """Encodes normalized sentence index (i / N in [0, 1]) via Fourier features."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

    def forward(self, pos: torch.Tensor):
        # pos shape: (B, 1) in [0, 1]
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


class HarveyForRhetoricalRoleV3(nn.Module):
    """
    HarveyLM V3 for Rhetorical Role Classification:
    Chronological Context + Target-Span Isolation + Fourier Depth + Domain Embed + Focal Loss.
    """
    def __init__(self, config: HarveyTransformerConfig, num_labels: int,
                 pos_dim: int = 128, domain_dim: int = 32, hidden_dropout: float = 0.15,
                 class_weights: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.num_labels = num_labels
        self.config     = config
        self.pos_dim    = pos_dim
        self.domain_dim = domain_dim

        self.harvey     = HarveyLM(config)
        self.pos_enc    = DocumentPositionEncoder(pos_dim)
        self.domain_enc = nn.Embedding(2, domain_dim)
        self.dropout    = nn.Dropout(hidden_dropout)

        total_dim = config.dim + pos_dim + domain_dim
        self.pre_classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(hidden_dropout)
        )
        self.classifier = nn.Linear(512, num_labels)
        self.loss_fn    = ClassBalancedFocalLoss(gamma=gamma, weight=class_weights, ignore_index=-1)

    def extract_features(self, input_ids, attention_mask=None, target_mask=None, rel_pos=None, domain_id=None):
        """Extracts 1024-d target sentence embeddings + 160-d position/domain context."""
        outputs = self.harvey(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_causal_mask=False
        )
        hidden_states = outputs.hidden_states[0]   # (B, T, D)

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

    def forward(self, input_ids, attention_mask=None, target_mask=None, rel_pos=None, domain_id=None, labels=None):
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
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits, "features": target_pooled}

    @classmethod
    def from_pretrained_harvey(cls, checkpoint_dir: str, num_labels: int,
                               pos_dim: int = 128, domain_dim: int = 32, hidden_dropout: float = 0.15,
                               class_weights: torch.Tensor = None, gamma: float = 2.0):
        config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
        model = cls(config, num_labels=num_labels, pos_dim=pos_dim, domain_dim=domain_dim,
                    hidden_dropout=hidden_dropout, class_weights=class_weights, gamma=gamma)

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
        print(f"HarveyLM V3 backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")
        return model
