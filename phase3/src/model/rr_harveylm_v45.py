"""
Phase 3 - HarveyLM V4.5 Complete Model Architecture.

Backbone: Pretrained HarveyLM (219M, epoch 99, 12 layers, 1024-d, symmetrical ALiBi).
Extensions:
- Segment Distance Embeddings (5 distances {-2, -1, 0, +1, +2} -> 1024-d).
- Custom LORMA injected into all attention and feed-forward linear layers (r=64, alpha=128, trainable diagonal scale).
- Attribution-Aware Head V4.5 (3,290-d fused representation).
- Multi-Task Loss:
  - Standard Cross-Entropy with label smoothing (eps=0.05) - NO logit adjustment!
  - R-Drop consistency regularization (alpha=0.5).
  - Within-Document Supervised Contrastive Loss (lambda_con=0.15).
  - Confusable-Pair Weight Orthogonality (lambda_ortho=0.05).
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2.src.model.architecture import HarveyLM, HarveyTransformerConfig
from phase3.src.model.attribution_head_v45 import LormaV45Head


class HarveyLMV45ForRR(nn.Module):
    def __init__(self, config: HarveyTransformerConfig, num_classes: int = 13,
                 r_drop_weight: float = 0.5, supcon_weight: float = 0.15,
                 ortho_weight: float = 0.05, label_smoothing: float = 0.05):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.d_model = config.dim

        self.harvey = HarveyLM(config)

        # Segment Distance Embeddings for 5-sentence window:
        # Distance -2: 0, Distance -1: 1, Distance 0 (target): 2, Distance +1: 3, Distance +2: 4
        self.seg_embeddings = nn.Embedding(5, self.d_model)
        nn.init.normal_(self.seg_embeddings.weight, mean=0.0, std=0.02)

        # Attribution-Aware Head V4.5 (3,290-d)
        self.head = LormaV45Head(d_model=self.d_model, num_classes=num_classes)

        self.r_drop_weight = r_drop_weight
        self.supcon_weight = supcon_weight
        self.ortho_weight  = ortho_weight
        self.label_smoothing = label_smoothing

        # Confusable Pairs: (ArgPet, ArgResp), (PrecRelied, Ratio), (PrecRelied, PrecNotRelied), (LowerCourt, Fact)
        # Assuming label indices: Fact:0, Issue:1, ArgPet:2, ArgResp:3, Statute:4, Dissent:5,
        # PrecRelied:6, PrecNotRelied:7, PrecOverruled:8, LowerCourt:9, Ratio:10, PresentCourt:11, None:12
        self.conf_pairs = [(2, 3), (6, 10), (6, 7), (9, 0)]

    def compute_supcon_loss(self, u_supcon, labels, doc_ids, temperature=0.07):
        """
        Within-Document Supervised Contrastive Loss:
        Positives and negatives are sampled strictly within the same judgment document.
        """
        device = u_supcon.device
        batch_size = u_supcon.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # Mask of identical documents
        same_doc = torch.tensor([[doc_ids[i] == doc_ids[j] for j in range(batch_size)] for i in range(batch_size)], device=device)

        # Mask of identical labels
        labels_col = labels.unsqueeze(1)
        same_label = torch.eq(labels_col, labels_col.t())

        # Valid positives: same doc, same label, distinct sentence
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = same_doc & same_label & (~diag_mask)

        # Valid candidates in denominator: same document, distinct sentence
        cand_mask = same_doc & (~diag_mask)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(u_supcon, u_supcon.t()) / temperature

        # Numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_exp = torch.exp(sim_matrix - sim_max.detach()) * cand_mask.float()
        denom = sim_exp.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Log-probabilities
        log_prob = (sim_matrix - sim_max.detach()) - torch.log(denom)

        # Mean over positive pairs per sentence
        pos_counts = pos_mask.sum(dim=1)
        valid_anchors = pos_counts > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        loss_per_anchor = (log_prob * pos_mask.float()).sum(dim=1)[valid_anchors] / pos_counts[valid_anchors].float()
        return -loss_per_anchor.mean()

    def compute_ortho_loss(self):
        """Penalizes cosine similarity > 0.1 between confusable classifier rows."""
        W = self.head.classifier.weight # (13, in_dim)
        W_norm = F.normalize(W, dim=-1)
        loss = 0.0
        for (a, b) in self.conf_pairs:
            cos = (W_norm[a] * W_norm[b]).sum()
            loss = loss + torch.relu(cos - 0.10) ** 2
        return loss

    def forward(self, input_ids, attention_mask, segment_ids, target_mask,
                prev_mask, next_mask, token_type_ids, rel_pos, bound_feats,
                party_state, struct_scalars, domain_id, labels=None, doc_ids=None):
        # 1. Embed input IDs + Segment Distance Embeddings
        x = self.harvey.tok_embeddings(input_ids) + self.seg_embeddings(segment_ids)

        # 2. Pass through HarveyLM Backbone with Symmetrical ALiBi (use_causal_mask=False)
        outputs = self.harvey(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_causal_mask=False
        )
        H = outputs.hidden_states[0] # Last hidden layer: (B, L, d_model)

        # 3. Head forward pass
        head_out = self.head(
            H=H,
            target_mask=target_mask,
            prev_mask=prev_mask,
            next_mask=next_mask,
            token_type_ids=token_type_ids,
            rel_pos=rel_pos,
            bound_feats=bound_feats,
            party_state=party_state,
            struct_scalars=struct_scalars,
            domain_id=domain_id
        )

        logits1  = head_out["logits"]
        logits2  = head_out["logits2"]
        features = head_out["features"]
        u_supcon = head_out["u_supcon"]

        loss = None
        if labels is not None:
            # 1. Standard Cross-Entropy with label smoothing (NO logit adjustment!)
            ce_loss = F.cross_entropy(logits1, labels, label_smoothing=self.label_smoothing)

            # 2. R-Drop consistency loss
            p1 = F.log_softmax(logits1, dim=-1)
            p2 = F.log_softmax(logits2, dim=-1)
            kl1 = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction="batchmean")
            kl2 = F.kl_div(p2, F.softmax(logits1, dim=-1), reduction="batchmean")
            rdrop_loss = 0.5 * (kl1 + kl2)

            # 3. Within-Document SupCon loss
            if doc_ids is not None:
                supcon_loss = self.compute_supcon_loss(u_supcon, labels, doc_ids)
            else:
                supcon_loss = torch.tensor(0.0, device=logits1.device)

            # 4. Confusable weight orthogonality loss
            ortho_loss = self.compute_ortho_loss()

            loss = ce_loss + (self.r_drop_weight * rdrop_loss) + (self.supcon_weight * supcon_loss) + (self.ortho_weight * ortho_loss)

        return {
            "loss": loss,
            "logits": logits1,
            "features": features,
            "u_supcon": u_supcon
        }

    @classmethod
    def from_pretrained_harvey(cls, checkpoint_dir: str, num_classes: int = 13):
        config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
        model = cls(config, num_classes=num_classes)

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
        print(f"HarveyLM V4.5 backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")
        return model
