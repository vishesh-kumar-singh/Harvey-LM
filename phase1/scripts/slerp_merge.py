import os
import torch
import safetensors.torch
from collections import OrderedDict
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved")

print("Loading checkpoints for SLERP merge...")
sd0 = safetensors.torch.load_file(os.path.join(checkpoint_dir, "epoch_30", "model.safetensors"))
sd1 = safetensors.torch.load_file(os.path.join(checkpoint_dir, "epoch_40", "model.safetensors"))

merged_sd = OrderedDict()
t = 0.5  # 50% from each model

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # Flatten the tensors
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    # Calculate norms
    v0_norm = torch.norm(v0_flat)
    v1_norm = torch.norm(v1_flat)
    
    # If a tensor is completely empty/zero, fallback to lerp
    if v0_norm == 0.0 or v1_norm == 0.0:
        return (1.0 - t) * v0 + t * v1
        
    v0_norm_vec = v0_flat / v0_norm
    v1_norm_vec = v1_flat / v1_norm
    
    dot = torch.sum(v0_norm_vec * v1_norm_vec)
    
    # If the inputs are too close, linearly interpolate
    if torch.abs(dot) > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
        
    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    # Calculate angle of the interpolated vector
    theta_t = theta_0 * t
    
    sin_theta_0 = torch.sin(theta_0)
    sin_theta_t = torch.sin(theta_t)
    
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    res = (s0 * v0_flat) + (s1 * v1_flat)
    return res.reshape(v0.shape)

print("Performing SLERP (Spherical Linear Interpolation) merge...")
for key in sd0.keys():
    t0 = sd0[key].to(torch.float32)
    t1 = sd1[key].to(torch.float32)
    
    merged_tensor = slerp(t, t0, t1)
    merged_sd[key] = merged_tensor.to(sd0[key].dtype)

out_dir = os.path.join(checkpoint_dir, "epoch_101")
os.makedirs(out_dir, exist_ok=True)
out_model_path = os.path.join(out_dir, "model.safetensors")

print(f"Saving SLERP model to {out_model_path}...")
safetensors.torch.save_file(merged_sd, out_model_path)

print("Copying config and tokenizer files...")
src_dir = os.path.join(checkpoint_dir, "epoch_40")
for file_name in os.listdir(src_dir):
    if file_name != "model.safetensors" and not file_name.startswith("accelerator_state"):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(out_dir, file_name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

print("SLERP Checkpoint created successfully as epoch_101!")
