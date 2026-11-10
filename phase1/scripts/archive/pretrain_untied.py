import os
import math
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from src.model.architecture import get_300m_harvey_model

# Paper Optimization: FlashAttention-2 (Dao, 2023)
# Forces PyTorch to use FlashAttention for exact, memory-efficient, and incredibly fast attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# Adjust these according to VRAM limits. 
PER_DEVICE_BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 64
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 0.0005
EPOCHS = 5  # 5 epochs for the time limit
EXPERIMENT_NAME = "untied_rope_experiment"
USE_ALIBI = True

# Resume functionality. Set to a checkpoint path (e.g., "checkpoints/alibi_experiment/step_500") to resume, or None.
RESUME_FROM_CHECKPOINT = None

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tokenizer_path = os.path.join(base_dir, "models", "tokenizer", "legal_bpe_tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Tokenizer not found! Run scripts/train_tokenizer.py first.")
        
    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "pad_token": "[PAD]"
    })
    
    # Initialize Accelerator for mixed precision (BF16 for stability with Muon and LLaMA)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    
    print("Initializing Model...")
    model = get_300m_harvey_model(vocab_size=len(tokenizer), context_length=MAX_SEQ_LENGTH, use_alibi=False)
    model.config.tie_word_embeddings = False
    
    # Manually initialize the output layer since we untied it
    import torch.nn as nn
    model.output = nn.Linear(model.config.dim, model.config.vocab_size, bias=False)
    torch.nn.init.normal_(model.output.weight, mean=0.0, std=0.02)
    model._tied_weights_keys = {}
    
    # 1. Disable Gradient Checkpointing to massively accelerate training (trades VRAM for speed)
    model.gradient_checkpointing = False
    
    # 2. PyTorch 2.0 Compilation (Ansel et al., 2024) - Fuses kernels to reduce training time
    if hasattr(torch, "compile"):
        print("Compiling model with TorchInductor for 20-30% speedup...")
        model = torch.compile(model)
    
    from datasets import load_from_disk
    
    dataset_path = os.path.join(base_dir, "data", "processed", "train_dataset_2048")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run src/data/process_data.py first.")
        
    print(f"Loading Dataset from {dataset_path}...")
    full_dataset = load_from_disk(dataset_path)
    
    print(f"Dataset loaded. Will truncate sequences to {MAX_SEQ_LENGTH} during collation.")
    
    print("Splitting dataset into train (99%) and validation (1%) without shuffling to preserve curriculum order...")
    # We must NOT shuffle here, to preserve the SC -> HC -> Trib -> Acts ordering!
    split_dataset = full_dataset.train_test_split(test_size=0.01, shuffle=False)
    train_dataset = split_dataset['train']
    
    CHUNK_SIZE = 50000 # Adjusted from 200,000 for 4x longer context (2048) to keep epoch scale identical
    eval_dataset = split_dataset['test']
    if len(eval_dataset) > 500:
        eval_dataset = eval_dataset.select(range(500))  # Drastically reduce validation time
    
    def custom_collate(batch):
        # Truncate to MAX_SEQ_LENGTH in case the dataset has longer sequences (e.g. 2048)
        input_ids = [item['input_ids'][:MAX_SEQ_LENGTH] for item in batch]
        max_len = max(len(x) for x in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [3] * pad_len) # 3 is PAD
            padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
            
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)
        
        labels_tensor = input_ids_tensor.clone()
        labels_tensor[attention_mask_tensor == 0] = -100
        
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor
        }
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=custom_collate, 
        batch_size=PER_DEVICE_BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer (Muon)
    from src.model.optim_muon import Muon
    optimizer = Muon(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Scheduler
    samples_per_epoch = min(len(train_dataset), CHUNK_SIZE)
    steps_per_epoch = math.ceil(samples_per_epoch / PER_DEVICE_BATCH_SIZE)
    num_update_steps_per_epoch = math.ceil(steps_per_epoch / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(0.05 * max_train_steps) # 5% warmup
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare everything except train_dataloader with accelerator
    model, optimizer, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, eval_dataloader, lr_scheduler
    )
    
    # We DO NOT register lr_scheduler for checkpointing here because the old checkpoint was based on 512 context length step counts.
    
    starting_epoch = 0
    if RESUME_FROM_CHECKPOINT is not None:
        print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
        accelerator.load_state(os.path.join(RESUME_FROM_CHECKPOINT, "accelerator_state"))
        import re
        match = re.search(r'epoch_(\d+)', RESUME_FROM_CHECKPOINT)
        if match:
            starting_epoch = int(match.group(1))
            print(f"Resuming at epoch {starting_epoch}")
    
    # Custom Training Loop
    print(f"Starting Training for {EPOCHS} epochs (Data Chunking Enabled)...")
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, EPOCHS):
        # Proportional Curriculum Learning: Sample from all 4 categories every epoch, and keep them ordered!
        # Approximate distribution of the 1.93M chunks: SC (1.4%), HC (72.9%), Tribunals (25.0%), Acts (0.7%)
        N = len(train_dataset)
        bounds = {
            "SC": (0, int(0.014 * N)),
            "HC": (int(0.014 * N), int(0.743 * N)),
            "Trib": (int(0.743 * N), int(0.993 * N)),
            "Acts": (int(0.993 * N), N)
        }
        
        import random
        
        sizes = {
            "SC": int(0.30 * CHUNK_SIZE),
            "HC": int(0.30 * CHUNK_SIZE),
            "Trib": int(0.30 * CHUNK_SIZE),
            "Acts": int(0.10 * CHUNK_SIZE)
        }
        sizes["Acts"] = CHUNK_SIZE - sum([sizes["SC"], sizes["HC"], sizes["Trib"]])
        
        from datasets import concatenate_datasets
        
        epoch_sub_datasets = []
        for cat in ["SC", "HC", "Trib", "Acts"]:
            cat_start, cat_end = bounds[cat]
            # Randomly sample with replacement to support any size
            indices = random.choices(range(cat_start, cat_end), k=sizes[cat])
            epoch_sub_datasets.append(train_dataset.select(indices))
                
        current_train_dataset = concatenate_datasets(epoch_sub_datasets)
        print(f"\nEpoch {epoch+1}: Proportional Curriculum Chunking ({sizes['SC']} SC -> {sizes['HC']} HC -> {sizes['Trib']} Trib -> {sizes['Acts']} Acts)")
        
        train_dataloader = DataLoader(
            current_train_dataset, 
            shuffle=True, # SHUFFLE IS CRITICAL to prevent catastrophic forgetting of previous categories
            collate_fn=custom_collate, 
            batch_size=PER_DEVICE_BATCH_SIZE,
            num_workers=2,
            pin_memory=True
        )
        
        train_dataloader = accelerator.prepare(train_dataloader)
        
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                accelerator.backward(loss)
                
                # Critical for FP16 stability: Gradient Clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed! Average Training Loss: {avg_epoch_loss:.4f}")
        
        # Append loss to training_losses.txt
        if accelerator.is_local_main_process:
            with open(os.path.join(base_dir, "training_losses.txt"), "a") as f:
                f.write(f"Epoch {epoch+1} completed! Average Training Loss: {avg_epoch_loss:.4f}\n")
        
        # Validation Phase
        model.eval()
        total_eval_loss = 0
        print("Running Validation...")
        for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process, desc="Validation"):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.item()
                
        # Gather evaluation loss from all GPUs
        total_eval_loss_tensor = torch.tensor(total_eval_loss, device=accelerator.device)
        total_eval_loss_tensor = accelerator.reduce(total_eval_loss_tensor, reduction="sum")
        
        # Save model checkpoint and print validation metrics
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            avg_eval_loss = total_eval_loss_tensor.item() / (len(eval_dataloader) * accelerator.num_processes)
            print(f"Epoch {epoch+1} - Validation Loss: {avg_eval_loss:.4f}")
            
            unwrapped_model = accelerator.unwrap_model(model)
            save_dir = os.path.join(base_dir, "models", "checkpoints", EXPERIMENT_NAME, f"epoch_{epoch+1}")
            unwrapped_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            accelerator.save_state(os.path.join(save_dir, "accelerator_state"))

    print("Training Complete!")

if __name__ == "__main__":
    main()
