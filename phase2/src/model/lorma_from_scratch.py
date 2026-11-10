import torch
import torch.nn as nn
import math

class LoRMALinear(nn.Module):
    """
    Low-Rank Matrix Approximation (LORMA).
    Initializes a Low-Rank bottleneck A and B using the truncated SVD of the original weight matrix W.
    W \approx W_res + B @ A
    A = sqrt(S_r) * V_r^T
    B = U_r * sqrt(S_r)
    W_res is frozen, A and B are trained.
    """
    def __init__(self, original_layer: nn.Linear, r: int = 16, lorma_alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        
        with torch.no_grad():
            W = original_layer.weight.data.float() # [out_features, in_features]
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            
            S_sqrt = torch.diag(torch.sqrt(S_r))
            B_init = U_r @ S_sqrt # [out_features, r]
            A_init = S_sqrt @ Vh_r # [r, in_features]
            
            W_res = W - B_init @ A_init
            
        self.W_res = nn.Parameter(W_res, requires_grad=False)
        self.A = nn.Parameter(A_init.clone())
        self.B = nn.Parameter(B_init.clone())
        
        self.scaling = lorma_alpha / r if lorma_alpha > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout)
        
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor):
        res_out = nn.functional.linear(x, self.W_res, self.bias)
        lorma_out = nn.functional.linear(nn.functional.linear(self.dropout(x), self.A), self.B) * self.scaling
        return res_out + lorma_out

def apply_lorma_to_model(model: nn.Module, target_modules: list, r: int = 16, lorma_alpha: int = 32, dropout: float = 0.05):
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
                
    print(f"Injected custom LORMA into {replaced_count} target modules.")
    return model

def mark_only_lorma_as_trainable(model: nn.Module, additional_trainable_modules: list = None):
    if additional_trainable_modules is None:
        additional_trainable_modules = []
        
    for name, param in model.named_parameters():
        if "W_res" in name:
            param.requires_grad = False
        elif ".A" in name or ".B" in name:
            param.requires_grad = True
        elif any(target in name for target in additional_trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Custom LORMA - Trainable params: {trainable:,} || Total params: {total:,} || Trainable %: {100 * trainable / total:.2f}%")
