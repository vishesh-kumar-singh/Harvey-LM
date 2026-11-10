import os
import torch
import safetensors.torch
from collections import OrderedDict
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved")

print("Loading checkpoints for DARE merge...")
base_sd = safetensors.torch.load_file(os.path.join(checkpoint_dir, "epoch_28", "model.safetensors"))
sd30 = safetensors.torch.load_file(os.path.join(checkpoint_dir, "epoch_30", "model.safetensors"))
sd40 = safetensors.torch.load_file(os.path.join(checkpoint_dir, "epoch_40", "model.safetensors"))

merged_sd = OrderedDict()
drop_rate = 0.5  # Drop bottom 50% of task-specific weight updates

print("Performing mathematical DARE (Drop And Rescale) merge...")
for key in base_sd.keys():
    t_base = base_sd[key].to(torch.float32)
    t30 = sd30[key].to(torch.float32)
    t40 = sd40[key].to(torch.float32)
    
    # Compute Task Vectors (the difference between the future epochs and the base epoch)
    tv30 = t30 - t_base
    tv40 = t40 - t_base
    
    def drop_and_rescale(tv, p):
        if tv.numel() == 0:
            return tv
        
        # Calculate the threshold magnitude below which weights will be dropped
        k = max(1, int(tv.numel() * (1 - p)))
        flat_tv = tv.flatten()
        # TopK is memory intensive on large matrices, so we use kthvalue or just sample
        # But for model weights, topk is usually fine on CPU
        threshold = torch.topk(torch.abs(flat_tv), k).values[-1]
        
        # Drop the lowest magnitude weights and rescale the rest
        mask = torch.abs(tv) >= threshold
        rescaled = (tv * mask) / (1 - p)
        return rescaled

    # Drop the redundant parameters from the task vectors
    tv30_dare = drop_and_rescale(tv30, drop_rate)
    tv40_dare = drop_and_rescale(tv40, drop_rate)
    
    # Average the refined, high-magnitude task vectors
    avg_tv = (tv30_dare + tv40_dare) / 2.0
    
    # Add them back to the base model
    merged_tensor = t_base + avg_tv
    merged_sd[key] = merged_tensor.to(base_sd[key].dtype)

out_dir = os.path.join(checkpoint_dir, "epoch_100")
os.makedirs(out_dir, exist_ok=True)
out_model_path = os.path.join(out_dir, "model.safetensors")

print(f"Saving DARE model to {out_model_path}...")
safetensors.torch.save_file(merged_sd, out_model_path)

print("Copying config and tokenizer files...")
src_dir = os.path.join(checkpoint_dir, "epoch_40")
for file_name in os.listdir(src_dir):
    if file_name != "model.safetensors" and not file_name.startswith("accelerator_state"):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(out_dir, file_name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

print("DARE Checkpoint created successfully as epoch_100!")
