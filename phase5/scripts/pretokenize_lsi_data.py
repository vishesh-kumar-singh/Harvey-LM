#!/usr/bin/env python3
"""
Phase 5: High-Speed Multi-Core Pre-Tokenization for LSI.

Tokenizes Train (42,750), Dev (10,181), and Test (13,019) cases into
Head-Tail (2,048 token) tensor caches using multi-processing across 32 CPU cores.
"""

import os
import sys
import json
import time
import torch
from concurrent.futures import ProcessPoolExecutor
from transformers import PreTrainedTokenizerFast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CODE_DIR = os.path.join(ROOT_DIR, "code")

TOKENIZER_PATH = os.path.join(CODE_DIR, "phase1/models/tokenizer/legal_bpe_tokenizer.json")
DATA_DIR       = os.path.join(CODE_DIR, "phase5/data/raw/Phase_5_LSI")
CACHE_DIR      = os.path.join(CODE_DIR, "phase5/data/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_tokenizer = None

def init_worker(tok_path):
    global _tokenizer
    _tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token_id = 3
    if _tokenizer.bos_token is None:
        _tokenizer.bos_token_id = 1
    if _tokenizer.eos_token is None:
        _tokenizer.eos_token_id = 2

def process_item(item):
    global _tokenizer
    cid = str(item["id"])
    text = item.get("text", "")
    token_ids = _tokenizer.encode(text, add_special_tokens=False)

    # Head-Tail budgeting (1024 + 1024)
    total_budget = 2048 - 2
    if len(token_ids) <= total_budget:
        seq_ids = [_tokenizer.bos_token_id] + token_ids + [_tokenizer.eos_token_id]
    else:
        h_len = min(1024, total_budget // 2)
        t_len = total_budget - h_len
        seq_ids = [_tokenizer.bos_token_id] + token_ids[:h_len] + token_ids[-t_len:] + [_tokenizer.eos_token_id]

    labels = []
    if "labels" in item and item["labels"] is not None:
        labels = [int(l) for l in item["labels"] if 0 <= int(l) < 100]

    return {
        "id": cid,
        "input_ids": torch.tensor(seq_ids, dtype=torch.int32),
        "labels": torch.tensor(labels, dtype=torch.int16)
    }

def pretokenize_split(split_name, json_path, out_path, num_workers=32):
    print(f"Pre-tokenizing {split_name} ({json_path}) with {num_workers} workers...")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(TOKENIZER_PATH,)) as ex:
        results = list(ex.map(process_item, data, chunksize=128))

    torch.save(results, out_path)
    elapsed = time.time() - t0
    print(f" Saved {split_name} ({len(results)} items) to {out_path} in {elapsed:.1f}s ({len(results)/elapsed:.0f} docs/sec)")

def main():
    print("=" * 80)
    print("Pre-tokenizing Phase 5 LSI Data across 32 CPU Cores...")
    print("=" * 80)

    pretokenize_split("dev",   os.path.join(DATA_DIR, "Train/dev.json"),   os.path.join(CACHE_DIR, "lsi_headtail_dev.pt"))
    pretokenize_split("test",  os.path.join(DATA_DIR, "Test/test.json"),   os.path.join(CACHE_DIR, "lsi_headtail_test.pt"))
    pretokenize_split("train", os.path.join(DATA_DIR, "Train/train.json"), os.path.join(CACHE_DIR, "lsi_headtail_train.pt"))

    print("\nPre-tokenization Complete! Cached files ready in:", CACHE_DIR)

if __name__ == "__main__":
    main()
