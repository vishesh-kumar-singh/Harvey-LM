#!/usr/bin/env python3
"""
Explore Phase 6 Abstractive Legal Summarization dataset.
Analyzes document lengths, summary lengths, compression ratios, and sample formats.
"""

import os
import json
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/Phase_6_Summ"))
TRAIN_PATH = os.path.join(DATA_DIR, "Train/train.json")
TEST_PATH = os.path.join(DATA_DIR, "Test/test.json")

def load_json(path):
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def analyze_dataset():
    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)
    
    print("\n=======================================================")
    print("PHASE 6 DATASET HIGH-LEVEL OVERVIEW")
    print("=======================================================")
    print(f"Train data type: {type(train_data)}")
    if isinstance(train_data, list):
        print(f"Train records count: {len(train_data)}")
        train_keys = list(train_data[0].keys())
        print(f"Train sample keys: {train_keys}")
        train_ids = [r.get("id") for r in train_data]
    elif isinstance(train_data, dict):
        print(f"Train keys count: {len(train_data)}")
        first_k = next(iter(train_data))
        print(f"Train first key: {first_k}")
        print(f"Train sample inner value type: {type(train_data[first_k])}")
        if isinstance(train_data[first_k], dict):
            print(f"Train sample inner keys: {list(train_data[first_k].keys())}")
        train_ids = list(train_data.keys())

    print(f"\nTest data type: {type(test_data)}")
    if isinstance(test_data, list):
        print(f"Test records count: {len(test_data)}")
        test_keys = list(test_data[0].keys())
        print(f"Test sample keys: {test_keys}")
        test_ids = [r.get("id") for r in test_data]
    elif isinstance(test_data, dict):
        print(f"Test keys count: {len(test_data)}")
        first_k = next(iter(test_data))
        print(f"Test first key: {first_k}")
        test_ids = list(test_data.keys())

    # ID overlaps
    overlap = set(train_ids).intersection(set(test_ids))
    print(f"Overlap between Train and Test IDs: {len(overlap)}")

    # Standardize to list of records for statistical profiling
    def to_records(raw, is_test=False):
        records = []
        if isinstance(raw, list):
            for item in raw:
                doc = item.get("document", item.get("text", ""))
                summ = item.get("summary", "") if not is_test else ""
                records.append({"id": item.get("id"), "document": doc, "summary": summ})
        elif isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    doc = v.get("document", v.get("text", ""))
                    summ = v.get("summary", "") if not is_test else ""
                    records.append({"id": k, "document": doc, "summary": summ})
                elif isinstance(v, str):
                    records.append({"id": k, "document": v, "summary": ""})
        return records

    train_recs = to_records(train_data, is_test=False)
    test_recs = to_records(test_data, is_test=True)

    print("\n=======================================================")
    print("DOCUMENT & SUMMARY LENGTH STATISTICS")
    print("=======================================================")
    train_doc_words = [len(r["document"].split()) for r in train_recs]
    train_summ_words = [len(r["summary"].split()) for r in train_recs]
    test_doc_words = [len(r["document"].split()) for r in test_recs]

    print(f"TRAIN Documents ({len(train_recs)} cases):")
    print(f"  Word count  - Mean: {np.mean(train_doc_words):.1f}, Median: {np.median(train_doc_words):.1f}, Std: {np.std(train_doc_words):.1f}")
    print(f"  Word count  - Min: {np.min(train_doc_words)}, Max: {np.max(train_doc_words)}")
    print(f"  Percentiles - 25%: {np.percentile(train_doc_words, 25):.1f}, 75%: {np.percentile(train_doc_words, 75):.1f}, 90%: {np.percentile(train_doc_words, 90):.1f}, 95%: {np.percentile(train_doc_words, 95):.1f}, 99%: {np.percentile(train_doc_words, 99):.1f}")

    print(f"\nTRAIN Summaries ({len(train_recs)} cases):")
    print(f"  Word count  - Mean: {np.mean(train_summ_words):.1f}, Median: {np.median(train_summ_words):.1f}, Std: {np.std(train_summ_words):.1f}")
    print(f"  Word count  - Min: {np.min(train_summ_words)}, Max: {np.max(train_summ_words)}")
    print(f"  Percentiles - 25%: {np.percentile(train_summ_words, 25):.1f}, 75%: {np.percentile(train_summ_words, 75):.1f}, 90%: {np.percentile(train_summ_words, 90):.1f}, 95%: {np.percentile(train_summ_words, 95):.1f}, 99%: {np.percentile(train_summ_words, 99):.1f}")

    compression_ratios = [d / max(1, s) for d, s in zip(train_doc_words, train_summ_words)]
    print(f"\nCompression Ratio (Doc Words / Summary Words):")
    print(f"  Mean: {np.mean(compression_ratios):.2f}x, Median: {np.median(compression_ratios):.2f}x, Min: {np.min(compression_ratios):.2f}x, Max: {np.max(compression_ratios):.2f}x")

    print(f"\nTEST Documents ({len(test_recs)} cases):")
    print(f"  Word count  - Mean: {np.mean(test_doc_words):.1f}, Median: {np.median(test_doc_words):.1f}, Std: {np.std(test_doc_words):.1f}")
    print(f"  Word count  - Min: {np.min(test_doc_words)}, Max: {np.max(test_doc_words)}")
    print(f"  Percentiles - 25%: {np.percentile(test_doc_words, 25):.1f}, 75%: {np.percentile(test_doc_words, 75):.1f}, 90%: {np.percentile(test_doc_words, 90):.1f}, 95%: {np.percentile(test_doc_words, 95):.1f}")

    print("\n=======================================================")
    print("SAMPLE RECORD INSPECTION")
    print("=======================================================")
    sample = train_recs[0]
    print(f"Sample ID: {sample['id']}")
    print(f"\nDocument snippet (first 350 chars):\n{sample['document'][:350]}...")
    print(f"\nDocument tail snippet (last 350 chars):\n{sample['document'][-350:]}...")
    print(f"\nReference Summary ({len(sample['summary'].split())} words):\n{sample['summary']}")

    test_sample = test_recs[0]
    print(f"\nTest Sample ID: {test_sample['id']}")
    print(f"Test Document snippet (first 350 chars):\n{test_sample['document'][:350]}...")

if __name__ == "__main__":
    analyze_dataset()
