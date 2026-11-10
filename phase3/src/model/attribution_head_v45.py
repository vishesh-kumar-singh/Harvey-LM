"""
Phase 3 - Attribution-Aware Head V4.5 with Within-Document SupCon & Cosine Transitions.

Clean architecture:
- Preserves V4 Attentive Pooling Head (3,232-d foundation).
- Replaces noisy 3,072-d causal difference vectors with 2 scale-invariant cosine similarity scalars.
- Appends 32-d boundary encodings (start/end distances) and 16-d causal party-state projections.
- Dual forward pass for R-Drop consistency.
- 256-d L2-normalized projection head for Within-Document SupCon.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=rel_pos.device) / half_dim
        )
        args = rel_pos * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


class BoundaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, bound_feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(bound_feats)


class PartyStateProjection(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, party_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(party_state)


class LormaV45Head(nn.Module):
    """
    Surgical Upgrade of LORMA V4 Head.
    """
    def __init__(self, d_model=1024, num_classes=13, d_pos=128, d_dom=32, d_bound=32, d_party=16):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # 4-head attention pooling with -2.5 stopword mask and gentle +0.5 init for legal cues
        self.q_proj = nn.Linear(d_model, 4, bias=False)
        self.type_bias = nn.Parameter(torch.tensor([
            [0.0, -2.5, 0.5, 0.2, 0.2],
            [0.0, -2.5, 0.2, 0.6, 0.2],
            [0.0, -2.5, 0.2, 0.2, 0.6],
            [0.0, -1.5, 0.0, 0.0, 0.0],
        ])) # (4 heads, 5 token categories)

        self.pos_enc = FourierPositionalEncoding(embed_dim=d_pos)
        self.dom_enc = nn.Embedding(2, d_dom)
        self.bound_enc = BoundaryPositionalEncoding(embed_dim=d_bound)
        self.party_enc = PartyStateProjection(embed_dim=d_party)

        # Total dimension:
        # h_att(1024) + h_max(1024) + h_mean(1024) + pos(128) + dom(32) + bound(32) + party(16) + cos(2) + struct(4)
        # = 3072 + 212 + 6 = 3290
        in_dim = (3 * d_model) + d_pos + d_dom + d_bound + d_party + 2 + 4
        self.in_dim = in_dim

        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(in_dim, num_classes)

        # Discarded projection head for Within-Document SupCon
        self.supcon_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )

    def pool_span(self, H, span_mask, token_type_ids=None):
        mask_f = span_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        h_mean = (H * mask_f).sum(dim=1) / denom

        H_masked = H.masked_fill(~span_mask.unsqueeze(-1), -1e9)
        h_max = H_masked.max(dim=1).values

        if token_type_ids is None:
            return h_mean, h_max

        # Attentive pooling strictly over target sentence tokens
        logits = self.q_proj(H) / math.sqrt(self.d_model // 4)
        type_biases = self.type_bias[:, token_type_ids].permute(1, 2, 0)
        logits = logits + type_biases
        logits = logits.masked_fill(~span_mask.unsqueeze(-1), -1e9)
        attn = torch.softmax(logits, dim=1)
        h_att = torch.einsum('blh,bld->bhd', attn, H).mean(dim=1)
        return h_att, h_max, h_mean

    def forward(self, H, target_mask, prev_mask, next_mask, token_type_ids,
                rel_pos, bound_feats, party_state, struct_scalars, domain_id):
        # 1. Target sentence representations
        h_att, h_max, h_mean = self.pool_span(H, target_mask, token_type_ids)

        # 2. Scale-invariant cosine transitions with neighbor sentences
        h_prev, _ = self.pool_span(H, prev_mask)
        h_next, _ = self.pool_span(H, next_mask)

        cos_prev = F.cosine_similarity(h_mean, h_prev, dim=-1, eps=1e-6).unsqueeze(-1)
        cos_next = F.cosine_similarity(h_mean, h_next, dim=-1, eps=1e-6).unsqueeze(-1)

        # 3. Position, domain, boundary, and party embeddings
        p_emb = self.pos_enc(rel_pos)
        d_emb = self.dom_enc(domain_id.squeeze(-1))
        b_emb = self.bound_enc(bound_feats)
        party_emb = self.party_enc(party_state)

        # 4. Concatenation
        z = torch.cat([
            h_att, h_max, h_mean,
            p_emb, d_emb, b_emb, party_emb,
            cos_prev, cos_next, struct_scalars
        ], dim=-1)

        z = self.norm(z)

        # 5. Dual classification passes for R-Drop consistency
        logits1 = self.classifier(self.dropout(z))
        logits2 = self.classifier(self.dropout(z))

        # 6. SupCon projection
        u_supcon = F.normalize(self.supcon_proj(z), dim=-1)

        return {
            "logits": logits1,
            "logits2": logits2,
            "features": z,
            "u_supcon": u_supcon
        }
