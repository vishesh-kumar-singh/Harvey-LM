import os
import torch
import safetensors.torch
from collections import OrderedDict
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved")

epochs_to_average = [28, 30, 40]
print(f"Averaging checkpoints: {epochs_to_average}")

state_dicts = []
for ep in epochs_to_average:
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{ep}", "model.safetensors")
    print(f"Loading {ckpt_path}...")
    sd = safetensors.torch.load_file(ckpt_path)
    state_dicts.append(sd)

print("Averaging weights...")
averaged_sd = OrderedDict()
for key in state_dicts[0].keys():
    # Convert to float32 to prevent precision loss during summation
    tensor_sum = state_dicts[0][key].clone().to(torch.float32)
    for sd in state_dicts[1:]:
        tensor_sum += sd[key].to(torch.float32)
    # Average and convert back to original dtype
    averaged_sd[key] = (tensor_sum / len(state_dicts)).to(state_dicts[0][key].dtype)

out_dir = os.path.join(checkpoint_dir, "epoch_99")
os.makedirs(out_dir, exist_ok=True)

out_model_path = os.path.join(out_dir, "model.safetensors")
print(f"Saving averaged model to {out_model_path}...")
safetensors.torch.save_file(averaged_sd, out_model_path)

# Copy config and tokenizer files from epoch 40
print("Copying config and tokenizer files...")
src_dir = os.path.join(checkpoint_dir, f"epoch_{epochs_to_average[-1]}")
for file_name in os.listdir(src_dir):
    if file_name != "model.safetensors" and not file_name.startswith("accelerator_state"):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(out_dir, file_name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

print("SWA Checkpoint created successfully as epoch_99!")
