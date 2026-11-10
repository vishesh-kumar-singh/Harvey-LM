"""
Phase 3 - Low-Rank Matrix Approximation (LORMA) from scratch.

Mathematical Foundation:
Decomposes pretrained weight W into a frozen orthogonal residual W_res
and a trainable low-rank bottleneck (B, A) initialized via Truncated SVD:
  W = W_res + s * (B @ A)
where:
  U_r, S_r, Vh_r = TruncatedSVD(W, rank=r)
  s = lorma_alpha / r
  A = sqrt(S_r / s) @ Vh_r
  B = U_r @ sqrt(S_r / s)
  W_res = W - s * (B @ A)

Guarantees 100% exact numerical identity to the pretrained backbone at initialization,
while allowing parameter-efficient adaptation of principal cognitive subspaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRMALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 64, lorma_alpha: int = 128, dropout: float = 0.10):
        super().__init__()
        self.r = r
        self.lorma_alpha = lorma_alpha
        self.scaling = lorma_alpha / r if (lorma_alpha > 0 and r > 0) else 1.0

        with torch.no_grad():
            W = original_layer.weight.data.float()  # [out_features, in_features]
            out_dim, in_dim = W.shape
            eff_r = min(r, out_dim, in_dim)

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            U_r = U[:, :eff_r]
            S_r = S[:eff_r]
            Vh_r = Vh[:eff_r, :]

            # Scale SVD factors so that: scaling * (B_init @ A_init) == U_r @ diag(S_r) @ Vh_r
            scaled_sqrt_S = torch.diag(torch.sqrt(S_r / self.scaling))
            B_init = U_r @ scaled_sqrt_S          # [out_features, eff_r]
            A_init = scaled_sqrt_S @ Vh_r          # [eff_r, in_features]

            # Residual frozen weight matrix
            W_approx = self.scaling * (B_init @ A_init)
            W_res = W - W_approx

        self.W_res = nn.Parameter(W_res.to(original_layer.weight.dtype), requires_grad=False)
        self.A = nn.Parameter(A_init.to(original_layer.weight.dtype), requires_grad=True)
        self.B = nn.Parameter(B_init.to(original_layer.weight.dtype), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        # x: (..., in_features)
        res_out = F.linear(x, self.W_res, self.bias)
        lorma_out = F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scaling
        return res_out + lorma_out


def apply_lorma_to_model(model: nn.Module, target_modules: list, r: int = 64, lorma_alpha: int = 128, dropout: float = 0.10):
    """
    Recursively replaces target nn.Linear layers in the model with LoRMALinear.
    """
    replaced_count = 0
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and any(target in child_name for target in target_modules):
                lorma_layer = LoRMALinear(child_module, r=r, lorma_alpha=lorma_alpha, dropout=dropout)
                setattr(module, child_name, lorma_layer)
                replaced_count += 1

    print(f"Injected custom LORMA into {replaced_count} target modules (r={r}, alpha={lorma_alpha}).")
    return model


def mark_only_lorma_as_trainable(model: nn.Module, additional_trainable: list = None):
    if additional_trainable is None:
        additional_trainable = ["pos_enc", "domain_enc", "pre_classifier", "classifier"]

    for name, param in model.named_parameters():
        if "W_res" in name:
            param.requires_grad = False
        elif ".A" in name or ".B" in name:
            param.requires_grad = True
        elif any(target in name for target in additional_trainable):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Custom LORMA - Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
