"""
Attentive Token Pooling and Multi-Representation Classification Head for Legal RR.

Innovations:
1. Multi-Head Self-Attentive Pooling (Lin et al., ICLR):
   Learns multiple attention hops across target tokens to extract distinct rhetorical semantic aspects.
2. Max-Over-Time Token Pooling:
   Captures peak activations of decisive legal keyword markers without averaging dilution.
3. Stopword & Punctuation Penalty in Attention Logits:
   Prevents common grammatical filler tokens from dominating sentence representations.
4. Multi-View Representation Fusion:
   Concatenates [h_att (1024), h_max (1024), h_mean (1024), pos_feat (128), dom_feat (32)] -> 3232-d representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveTokenPoolingHead(nn.Module):
    def __init__(self, embed_dim: int = 1024, attn_dim: int = 256, num_heads: int = 4,
                 pos_dim: int = 128, domain_dim: int = 32, num_labels: int = 13,
                 dropout: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.domain_dim = domain_dim

        # Multi-Head Attention Scoring over tokens
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, num_heads, bias=False)
        )

        # Multi-Head Projection
        self.head_proj = nn.Sequential(
            nn.Linear(num_heads * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fused classifier: [h_att (1024), h_max (1024), h_mean (1024), pos (128), dom (32)] = 3232-d
        total_dim = embed_dim * 3 + pos_dim + domain_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, hidden_states, target_mask, pos_feat, dom_feat, stopword_mask=None):
        """
        hidden_states: (B, L, D) - token embeddings from HarveyLM
        target_mask:   (B, L)    - 1 for target sentence tokens, 0 elsewhere
        pos_feat:      (B, pos_dim)
        dom_feat:      (B, domain_dim)
        stopword_mask: (B, L)    - 1 for stopwords/punctuation, 0 for content tokens
        """
        B, L, D = hidden_states.shape

        # Compute raw attention scores: (B, L, num_heads)
        scores = self.attn_net(hidden_states)

        # Mask out non-target tokens with large negative logits
        mask = (target_mask == 1).unsqueeze(-1) # (B, L, 1)
        scores = scores.masked_fill(~mask, -1e9)

        # Suppress stopword tokens in attention weights
        if stopword_mask is not None:
            sw = (stopword_mask == 1).unsqueeze(-1)
            scores = scores - 2.5 * sw.float()

        # Softmax over sequence length
        attn_weights = F.softmax(scores, dim=1) # (B, L, num_heads)

        # Weighted sum: (B, num_heads, D)
        # B x num_heads x L @ B x L x D -> B x num_heads x D
        h_att = torch.bmm(attn_weights.transpose(1, 2), hidden_states)
        h_att = self.head_proj(h_att.reshape(B, -1)) # (B, D)

        # Max-pooling over target tokens
        masked_h = hidden_states.masked_fill(~mask, -1e9)
        h_max = masked_h.max(dim=1)[0] # (B, D)
        # Safety clamp for empty masks (rare edge case)
        h_max = torch.where(torch.isneginf(h_max), torch.zeros_like(h_max), h_max)

        # Mean-pooling over target tokens
        denom = mask.sum(dim=1).clamp(min=1.0).float()
        h_mean = (hidden_states * mask.float()).sum(dim=1) / denom # (B, D)

        # Concatenate rich multi-view representations
        fused = torch.cat([h_att, h_max, h_mean, pos_feat, dom_feat], dim=-1) # (B, total_dim)
        logits = self.classifier(fused)

        return logits, fused, attn_weights
