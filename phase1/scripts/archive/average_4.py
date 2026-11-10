import os
import torch
import safetensors.torch
from collections import OrderedDict
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved")

print("Loading checkpoints for SWA (10, 20, 30, 40)...")
epochs_to_average = [10, 20, 30, 40]
state_dicts = []

for ep in epochs_to_average:
    sd = safetensors.torch.load_file(os.path.join(checkpoint_dir, f"epoch_{ep}", "model.safetensors"))
    state_dicts.append(sd)

merged_sd = OrderedDict()
print("Averaging weights...")

for key in state_dicts[0].keys():
    tensors = [sd[key].to(torch.float32) for sd in state_dicts]
    avg_tensor = sum(tensors) / len(tensors)
    merged_sd[key] = avg_tensor.to(state_dicts[0][key].dtype)

out_dir = os.path.join(checkpoint_dir, "epoch_102")  # 102 for this specific SWA
os.makedirs(out_dir, exist_ok=True)
out_model_path = os.path.join(out_dir, "model.safetensors")

print(f"Saving averaged model to {out_model_path}...")
safetensors.torch.save_file(merged_sd, out_model_path)

print("Copying config and tokenizer files...")
src_dir = os.path.join(checkpoint_dir, "epoch_40")
for file_name in os.listdir(src_dir):
    if file_name != "model.safetensors" and not file_name.startswith("accelerator_state"):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(out_dir, file_name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

print("SWA Checkpoint 102 created successfully!")
