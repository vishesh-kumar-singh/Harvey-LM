#!/usr/bin/env python3
"""
Phase 5: Teacher Soft Target Extraction for Knowledge Distillation.

Extracts calibrated 100-way multi-label logits from peak HarveyLM LORMA
on all 42,750 training documents and saves them to code/phase5/data/cache/teacher_soft_targets.pt.
"""

import os
import sys


import json
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase5.src.data.lsi_dataset import load_raw_lsi_data, LSIDataset, lsi_collate_fn_head_tail
from phase5.src.model.lsi_classifier import UniversalLSIClassifier, apply_lorma_to_model

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
HARVEYLM_CKPT  = os.path.join(CODE_DIR, "phase1/models/final_architectures/2_HarveyLM/epoch_99")
TEACHER_CKPT   = os.path.join(CODE_DIR, "phase5/models/harveylm/lorma_headtail/best_model.pt")
TRAIN_DATA     = os.path.join(CODE_DIR, "phase5/data/raw/Phase_5_LSI/Train/train.json")
OUTPUT_CACHE   = os.path.join(CODE_DIR, "phase5/data/cache/teacher_soft_targets.pt")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Generating Teacher Soft Targets on {device}...")
    print(f"Teacher Model Checkpoint: {TEACHER_CKPT}")
    print(f"Output Cache: {OUTPUT_CACHE}")
    print("=" * 80)

    os.makedirs(os.path.dirname(OUTPUT_CACHE), exist_ok=True)

    # 2. Dataset (Pretokenized Cache)
    cached_train = os.path.join(CODE_DIR, "phase5/data/cache/lsi_headtail_train.pt")
    print(f"Loading pre-tokenized cache from {cached_train}...")
    train_data = torch.load(cached_train, weights_only=False)
    print(f"Loaded {len(train_data)} training items directly into memory.")

    class FastCachedDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = self.items[idx]
            return {"id": str(item["id"]), "input_ids": item["input_ids"].long()}

    def fast_collate(batch):
        ids = [b["id"] for b in batch]
        max_len = max(b["input_ids"].size(0) for b in batch)
        pad_id = 3
        batch_input_ids = []
        batch_attn_masks = []
        for b in batch:
            cur_len = b["input_ids"].size(0)
            pad_len = max_len - cur_len
            if pad_len > 0:
                padded_ids = torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
                padded_mask = torch.cat([torch.ones(cur_len, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
            else:
                padded_ids = b["input_ids"]
                padded_mask = torch.ones(cur_len, dtype=torch.bool)
            batch_input_ids.append(padded_ids)
            batch_attn_masks.append(padded_mask)
        return {
            "ids": ids,
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attn_masks, dim=0).bool()
        }

    train_ds = FastCachedDataset(train_data)
    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=fast_collate)

    # 3. Model
    model = UniversalLSIClassifier(arch_type="harveylm", ckpt_dir=HARVEYLM_CKPT, num_classes=100)
    model.backbone = apply_lorma_to_model(model.backbone, target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"], r=32, lorma_alpha=64)

    ckpt = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).bfloat16().eval()
    print("Teacher model loaded and set to eval mode.")

    # 4. Extract Logits
    teacher_targets = {}
    print("Extracting teacher logits across all training cases...")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"].float().cpu()

            for cid, l in zip(batch["ids"], logits):
                teacher_targets[cid] = l

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                print(f"Extracted [{step+1}/{len(loader)}] batches ({len(teacher_targets)} cases)...", flush=True)

    # 5. Save Cache
    torch.save(teacher_targets, OUTPUT_CACHE)
    print(f"\nSuccessfully cached teacher soft targets for {len(teacher_targets)} cases into {OUTPUT_CACHE}!")


if __name__ == "__main__":
    main()
