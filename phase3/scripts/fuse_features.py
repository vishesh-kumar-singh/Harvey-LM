"""
Phase 3 - Feature Fusion: Concatenate Contextual Embeddings.
Generates:
  1. code/phase3/data/cache/harveylm_fused_features.pt (2048-d: LoRA + Full FT)
  2. code/phase3/data/cache/harveylm_lorma_fused_features.pt (2048-d: LORMA + Full FT)
  3. code/phase3/data/cache/harveylm_triple_fused_features.pt (3072-d: LoRA + Full FT + LORMA)
"""
import os
import argparse
import torch

LORA_CACHE  = "code/phase3/data/cache/harveylm_context_features.pt"
FULL_CACHE  = "code/phase3/data/cache/harveylm_full_features.pt"
LORMA_CACHE = "code/phase3/data/cache/harveylm_lorma_features.pt"

def fuse(inputs, output_path):
    print(f"\nFusing {len(inputs)} feature sets -> {output_path}")
    loaded = []
    for p in inputs:
        print(f"  Loading: {p}")
        loaded.append(torch.load(p, map_location="cpu", weights_only=True))

    fused_splits = {}
    for split in ["train", "dev", "test"]:
        base_docs = loaded[0][split]
        for l in loaded[1:]:
            assert len(l[split]) == len(base_docs), f"Mismatch in {split} doc count!"
        
        fused_docs = []
        for i in range(len(base_docs)):
            d0 = base_docs[i]
            embs = [d0["embeddings"]]
            for other in loaded[1:]:
                do = other[split][i]
                assert do["id"] == d0["id"], f"Doc ID mismatch: {do['id']} vs {d0['id']}"
                assert do["n_sents"] == d0["n_sents"], f"Sent count mismatch: {do['n_sents']} vs {d0['n_sents']}"
                assert torch.equal(do["labels"], d0["labels"]), f"Label mismatch in {d0['id']}"
                embs.append(do["embeddings"])

            emb_cat = torch.cat(embs, dim=-1)
            fused_docs.append({
                "id": d0["id"],
                "n_sents": d0["n_sents"],
                "embeddings": emb_cat,
                "positions": d0["positions"],
                "labels": d0["labels"]
            })
        fused_splits[split] = fused_docs
        print(f"  Fused {len(fused_docs)} documents for {split} split. Feature dimension: {fused_docs[0]['embeddings'].shape[-1]}")

    fused_splits["label2id"] = loaded[0]["label2id"]
    fused_splits["id2label"] = loaded[0]["id2label"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(fused_splits, output_path)
    print(f"Successfully saved to: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "lorma_full", "triple", "lora_full"], default="all")
    args = parser.parse_args()

    if args.mode in ["all", "lora_full"]:
        fuse([LORA_CACHE, FULL_CACHE], "code/phase3/data/cache/harveylm_fused_features.pt")

    if args.mode in ["all", "lorma_full"]:
        fuse([LORMA_CACHE, FULL_CACHE], "code/phase3/data/cache/harveylm_lorma_fused_features.pt")

    if args.mode in ["all", "triple"]:
        fuse([LORA_CACHE, FULL_CACHE, LORMA_CACHE], "code/phase3/data/cache/harveylm_triple_fused_features.pt")

