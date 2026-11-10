import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Custom implementation of Low-Rank Adaptation (LoRA) for linear layers.
    """
    def __init__(self, original_layer: nn.Linear, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        self.original_layer = original_layer
        
        # Freeze original layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize A with Kaiming uniform (same as standard linear)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Initialize B with zeros, so the initial forward pass is identical to the original layer
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_output + lora_output

def apply_lora_to_model(model: nn.Module, target_modules: list, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Recursively replaces target nn.Linear layers in the model with LoRALinear.
    """
    replaced_count = 0
    for name, module in model.named_modules():
        # Iterate over immediate children to allow in-place replacement
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and any(target in child_name for target in target_modules):
                # Replace with LoRALinear
                lora_layer = LoRALinear(child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                setattr(module, child_name, lora_layer)
                replaced_count += 1
                
    print(f"Injected custom LoRA into {replaced_count} target modules.")
    return model

def mark_only_lora_as_trainable(model: nn.Module, additional_trainable_modules: list = None):
    """
    Freezes everything except LoRA parameters and any specific modules.
    """
    if additional_trainable_modules is None:
        additional_trainable_modules = []
        
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif any(target in name for target in additional_trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Custom LoRA - Trainable params: {trainable:,} || Total params: {total:,} || Trainable %: {100 * trainable / total:.2f}%")
