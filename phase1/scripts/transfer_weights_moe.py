import os
import torch
import safetensors.torch
from collections import OrderedDict
import shutil

def transfer_weights():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_ckpt = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved", "epoch_40", "model.safetensors")
    
    print("Loading source checkpoint (Epoch 40)...")
    src_sd = safetensors.torch.load_file(src_ckpt)
    
    print("Initializing MoE model...")
    from src.model.moe import get_moe_model
    # We must match the dimension of the MoE we are training: dim=768, hidden_dim=2048, n_layers=12
    # WAIT! Epoch 40 has dim=1024, hidden_dim=2752, n_layers=16.
    # The shapes are completely incompatible!
    pass

if __name__ == "__main__":
    transfer_weights()
