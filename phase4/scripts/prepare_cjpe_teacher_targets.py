#!/usr/bin/env python3
"""
Phase 4: Teacher Soft Target Extraction for Knowledge Distillation.

Extracts calibrated soft probability distributions and logits from our peak
HarveyLM Head-Tail LORMA model (0.7712 Test Macro-F1) on all unique training cases.

Saves:
code/phase4/data/cache/teacher_soft_targets.pt
Mapping { case_id: Tensor([logit_0, logit_1]) }
"""

import os
import sys


import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")
PHASE1_DIR = os.path.join(CODE_DIR, "phase1")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase4.src.data.cjpe_dataset import load_raw_cjpe_data, CJPEDataset, cjpe_collate_fn_head_tail
from phase4.src.model.cjpe_classifier import HarveyHeadTailClassifier, apply_lorma_to_model

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
HARVEYLM_CKPT  = os.path.join(CODE_DIR, "phase1/models/final_architectures/2_HarveyLM/epoch_99")
BEST_TEACHER   = os.path.join(CODE_DIR, "phase4/models/harveylm/lorma_headtail/best_model.pt")
TRAIN_DATA     = os.path.join(CODE_DIR, "phase4/data/raw/Phase_4_CJPE/Train/train.json")
OUTPUT_CACHE   = os.path.join(CODE_DIR, "phase4/data/cache/teacher_soft_targets.pt")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Generating Teacher Soft Targets on {device}...")
    print(f"Teacher Model Checkpoint: {BEST_TEACHER}")
    print(f"Output Cache: {OUTPUT_CACHE}")
    print("=" * 80)

    os.makedirs(os.path.dirname(OUTPUT_CACHE), exist_ok=True)

    # 1. Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 3

    # 2. Dataset
    raw_train = load_raw_cjpe_data(TRAIN_DATA, deduplicate=True)
    print(f"Loaded {len(raw_train)} unique training cases.")

    train_ds = CJPEDataset(
        raw_items=raw_train,
        tokenizer=tokenizer,
        mode="head_tail",
        max_seq_len=2048,
        is_training=False,
        anti_leakage=False,
        tail_drop_prob=0.0
    )
    loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cjpe_collate_fn_head_tail)

    # 3. Model
    model = HarveyHeadTailClassifier(ckpt_dir=HARVEYLM_CKPT, num_classes=2, num_queries=4, head_hidden=1024)
    model.backbone = apply_lorma_to_model(model.backbone, target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"], r=32, lorma_alpha=64)

    ckpt = torch.load(BEST_TEACHER, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).bfloat16().eval()
    print("Teacher model loaded and set to eval mode.")

    # 4. Extract Logits
    teacher_targets = {}
    print("Extracting teacher logits across training cases...")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"].float().cpu()

            for cid, l in zip(batch["ids"], logits):
                teacher_targets[cid] = l

            if (step + 1) % 200 == 0 or (step + 1) == len(loader):
                print(f"Extracted [{step+1}/{len(loader)}] batches ({len(teacher_targets)} cases)...")

    # 5. Save Cache
    torch.save(teacher_targets, OUTPUT_CACHE)
    print(f"\nSuccessfully cached teacher soft targets for {len(teacher_targets)} cases into {OUTPUT_CACHE}!")


if __name__ == "__main__":
    main()
