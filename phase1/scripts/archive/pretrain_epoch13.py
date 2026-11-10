import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator
from tqdm.auto import tqdm
import json
import random

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved", "epoch_12")
    
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    if accelerator.is_local_main_process:
        print("Loading Tokenizer...")
        
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    
    if accelerator.is_local_main_process:
        print("Initializing Model...")
        
    from src.model.architecture import HarveyTransformerConfig, HarveyLM
    config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    model = HarveyLM(config)
    
    import safetensors.torch
    state_dict = safetensors.torch.load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    
    if accelerator.is_local_main_process:
        print("Loading and Balancing SC and HC Datasets...")
        
    # We will load SC and HC datasets and create a balanced, shuffled subset
    sc_data = []
    hc_data = []
    
    # Read chunked datasets (we assume they are already converted into long token arrays)
    # Actually, let's load from the raw JSON and re-tokenize, or just load the pre-tokenized chunks.
    # The pre-tokenized dataset is in data/processed/train_dataset
    dataset_path = os.path.join(base_dir, "data", "processed", "train_dataset_2048")
    from datasets import load_from_disk
    full_dataset = load_from_disk(dataset_path)
    
    # In earlier scripts, we didn't have source labels in the tokenized dataset, so we just relied on order.
    # The order was SC (108829 seqs of 2048), HC (1223346 seqs), etc.
    # So SC is index 0 to 108829. HC is index 108829 to 1332175.
    # Let's verify this mathematically or just slice!
    
    sc_subset = full_dataset.select(range(0, 10000))
    hc_subset = full_dataset.select(range(108829, 108829 + 10000))
    
    from datasets import concatenate_datasets
    balanced_dataset = concatenate_datasets([sc_subset, hc_subset])
    balanced_dataset = balanced_dataset.shuffle(seed=42)
    
    if accelerator.is_local_main_process:
        print(f"Created balanced dataset with {len(balanced_dataset)} sequences (10k SC + 10k HC).")
    
    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"][:2048], dtype=torch.long) for x in batch]
        input_ids = torch.stack(input_ids)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
        
    train_dataloader = DataLoader(
        balanced_dataset,
        batch_size=1, # Very small batch size for 2048 length
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Use standard AdamW for fine-tuning stability instead of Muon
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.1)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    GRAD_ACCUM = 64
    
    model.train()
    progress_bar = tqdm(total=len(train_dataloader) // GRAD_ACCUM, disable=not accelerator.is_local_main_process)
    
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs["loss"]
            accelerator.backward(loss)
            
            if (step + 1) % GRAD_ACCUM == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
    if accelerator.is_local_main_process:
        out_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved", "epoch_13")
        os.makedirs(out_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        state_dict = unwrapped_model.state_dict()
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        clean_state_dict.pop("output.weight", None)
        safetensors.torch.save_file(clean_state_dict, os.path.join(out_dir, "model.safetensors"))
        print("Epoch 13 fine-tuning complete!")

if __name__ == "__main__":
    main()
