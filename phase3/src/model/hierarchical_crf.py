"""
Phase 3 - Hierarchical Document Sequence Model with BiLSTM, Transformer Contextualizer, and CRF.

Architecture:
1. Input: HarveyLM 1024-d Sentence Embeddings + Fourier Document Position Depth.
2. Document Contextualizer: 2-layer Bidirectional LSTM + 2-layer Document Self-Attention Transformer.
   Captures long-range judicial narrative structure across all sentences in a court judgment.
3. Dual-Objective Loss:
   - Sequence CRF Loss (Negative Log-Likelihood over label transitions).
   - Auxiliary Macro-F1 Balanced Focal Loss on emission logits.
4. Exact Viterbi Decoding over learned legal transition matrix.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class DocumentPositionEncoder(nn.Module):
    """Maps normalized sentence depth (tau = i / N in [0, 1]) to Fourier features."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

    def forward(self, pos: torch.Tensor):
        # pos: (B, N, 1) in [0, 1]
        pi = math.pi
        feat = torch.cat([
            pos,
            torch.sin(pos * pi),
            torch.cos(pos * pi),
            torch.sin(pos * 2 * pi),
            torch.cos(pos * 2 * pi),
            pos ** 2
        ], dim=-1)
        return self.mlp(feat)


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        valid = targets != self.ignore_index
        return focal[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)


class HierarchicalLegalDocCRF(nn.Module):
    """
    Hierarchical Document Sequence Model for Rhetorical Role Prediction.
    """
    def __init__(self, embed_dim: int = 1024, pos_dim: int = 128, hidden_dim: int = 512,
                 num_labels: int = 13, class_weights: torch.Tensor = None,
                 focal_gamma: float = 2.0, focal_weight: float = 0.5):
        super().__init__()
        self.num_labels = num_labels
        self.pos_enc    = DocumentPositionEncoder(pos_dim)
        self.in_proj    = nn.Sequential(
            nn.Linear(embed_dim + pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )

        # 1. Sequential Document Contextualizer (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # 2. Document Transformer Contextualizer (Sentence-level global self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True,
            activation="gelu"
        )
        self.doc_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Emission Head
        self.emission = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_labels)
        )

        # 4. Linear-Chain CRF
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        # 5. Auxiliary Weighted Focal Loss
        self.focal_loss = WeightedFocalLoss(gamma=focal_gamma, weight=class_weights, ignore_index=-1)
        self.focal_weight = focal_weight

    def forward(self, embeddings, positions, mask=None, labels=None):
        """
        embeddings: (B, N_sents, 1024)
        positions:  (B, N_sents, 1)
        mask:       (B, N_sents) bool tensor
        labels:     (B, N_sents) long tensor in [0, num_labels - 1] (or -1 for padding/unknown)
        """
        B, N, _ = embeddings.shape

        # 1. Fuse sentence embedding with document depth
        pos_emb = self.pos_enc(positions)
        x = self.in_proj(torch.cat([embeddings, pos_emb], dim=-1))  # (B, N, hidden_dim)

        # 2. BiLSTM contextualization
        lstm_out, _ = self.bilstm(x)

        # 3. Document Transformer contextualization with padding mask
        src_key_padding_mask = ~mask if mask is not None else None
        trans_out = self.doc_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 4. Residual fusion & LayerNorm
        h = self.norm(lstm_out + trans_out)

        # 5. Emission logits
        emissions = self.emission(h)  # (B, N, num_labels)

        loss = None
        if labels is not None and mask is not None:
            # Clean labels for CRF (replace -1 with 0, mask ensures they don't count)
            clean_labels = labels.clone()
            clean_labels[~mask] = 0

            # CRF sequence negative log-likelihood
            crf_loss = -self.crf(emissions, clean_labels, mask=mask, reduction="token_mean")

            # Auxiliary Macro-F1 Focal Loss on emission tokens
            flat_emissions = emissions[mask]
            flat_targets   = labels[mask]
            aux_focal = self.focal_loss(flat_emissions, flat_targets)

            loss = crf_loss + self.focal_weight * aux_focal

        return {"loss": loss, "emissions": emissions}

    def decode(self, embeddings, positions, mask=None):
        """Runs global Viterbi sequence decoding over the document."""
        B, N, _ = embeddings.shape
        pos_emb = self.pos_enc(positions)
        x = self.in_proj(torch.cat([embeddings, pos_emb], dim=-1))

        lstm_out, _ = self.bilstm(x)
        src_key_padding_mask = ~mask if mask is not None else None
        trans_out = self.doc_transformer(x, src_key_padding_mask=src_key_padding_mask)

        h = self.norm(lstm_out + trans_out)
        emissions = self.emission(h)

        best_paths = self.crf.decode(emissions, mask=mask)
        return best_paths, emissions
