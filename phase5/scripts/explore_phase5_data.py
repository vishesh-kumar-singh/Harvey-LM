import os
import json
import numpy as np
from collections import Counter

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/Phase_5_LSI"))
TRAIN_PATH = os.path.join(DATA_DIR, "Train/train.json")
DEV_PATH   = os.path.join(DATA_DIR, "Train/dev.json")
TEST_PATH  = os.path.join(DATA_DIR, "Test/test.json")

def analyze_split(name, path, is_test=False):
    print(f"\n==================================================")
    print(f"Analyzing {name} split: {path}")
    print(f"==================================================")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total records: {len(data)}")
    
    sample = data[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample ID: {sample.get('id')}")
    if is_test:
        print(f"Sample text snippet: {repr(sample.get('text', ''))[:200]}...")
    else:
        print(f"Sample labels: {sample.get('labels')}")
        print(f"Sample text snippet: {repr(sample.get('text', ''))[:200]}...")
        
    # Text length stats
    char_lens = [len(r.get("text", "")) for r in data]
    word_lens = [len(r.get("text", "").split()) for r in data]
    print(f"Char length - Mean: {np.mean(char_lens):.1f}, Median: {np.median(char_lens):.1f}, Min: {np.min(char_lens)}, Max: {np.max(char_lens)}")
    print(f"Word length - Mean: {np.mean(word_lens):.1f}, Median: {np.median(word_lens):.1f}, Min: {np.min(word_lens)}, Max: {np.max(word_lens)}")

    label_counts = []
    all_labels = []
    if not is_test:
        for r in data:
            labs = r.get("labels", [])
            label_counts.append(len(labs))
            all_labels.extend(labs)
        print(f"Labels per doc - Mean: {np.mean(label_counts):.2f}, Median: {np.median(label_counts):.1f}, Min: {np.min(label_counts)}, Max: {np.max(label_counts)}")
        print(f"Total label assignments: {len(all_labels)}")
        unique_labels = set(all_labels)
        print(f"Unique labels in {name}: {len(unique_labels)}, Min label: {min(unique_labels)}, Max label: {max(unique_labels)}")
        
    return data, all_labels

def main():
    train_data, train_labels = analyze_split("Train", TRAIN_PATH, is_test=False)
    dev_data, dev_labels = analyze_split("Dev", DEV_PATH, is_test=False)
    test_data, _ = analyze_split("Test", TEST_PATH, is_test=True)
    
    # Combined label analysis
    combined_labels = train_labels + dev_labels
    all_unique = sorted(list(set(combined_labels)))
    print(f"\n==================================================")
    print("GLOBAL STATUTE LABEL ANALYSIS (Train + Dev)")
    print(f"==================================================")
    print(f"Total Unique Statute IDs: {len(all_unique)}")
    print(f"Statute ID range: [{min(all_unique)}, {max(all_unique)}]")
    
    # Check if continuous 0 to N-1
    expected_set = set(range(max(all_unique) + 1))
    missing_ids = expected_set - set(all_unique)
    print(f"Contiguous [0, {max(all_unique)}]? {len(missing_ids) == 0} (Missing IDs count: {len(missing_ids)})")
    if missing_ids:
        print(f"Missing IDs sample: {sorted(list(missing_ids))[:20]}")

    train_set = set(train_labels)
    dev_set = set(dev_labels)
    dev_only = dev_set - train_set
    print(f"Unique labels in Train: {len(train_set)}")
    print(f"Unique labels in Dev: {len(dev_set)}")
    print(f"Labels in Dev but NOT in Train (zero-shot/unseen): {len(dev_only)} ({dev_only})")
    
    counter = Counter(combined_labels)
    most_common = counter.most_common(10)
    print(f"\nTop 10 Most Frequent Statutes:")
    for lab, cnt in most_common:
        print(f"  Statute ID {lab:>3}: {cnt:>5} occurrences ({cnt/len(train_data+dev_data)*100:.2f}% of docs)")
        
    # Frequency distribution
    singletons = sum(1 for lab, cnt in counter.items() if cnt == 1)
    under_5 = sum(1 for lab, cnt in counter.items() if cnt < 5)
    under_10 = sum(1 for lab, cnt in counter.items() if cnt < 10)
    print(f"\nRare Statutes Analysis:")
    print(f"  Singletons (count == 1): {singletons} ({singletons/len(all_unique)*100:.1f}%)")
    print(f"  Fewer than 5 occurrences: {under_5} ({under_5/len(all_unique)*100:.1f}%)")
    print(f"  Fewer than 10 occurrences: {under_10} ({under_10/len(all_unique)*100:.1f}%)")
    
    # Check for duplicate document IDs or text leaks
    train_ids = set(r["id"] for r in train_data)
    dev_ids = set(r["id"] for r in dev_data)
    test_ids = set(r["id"] for r in test_data)
    print(f"\nSplit Overlaps:")
    print(f"  Train & Dev ID overlap: {len(train_ids & dev_ids)}")
    print(f"  Train & Test ID overlap: {len(train_ids & test_ids)}")
    print(f"  Dev & Test ID overlap: {len(dev_ids & test_ids)}")
    
    train_texts = {r["text"].strip(): r["labels"] for r in train_data}
    dev_text_overlap = sum(1 for r in dev_data if r["text"].strip() in train_texts)
    test_text_overlap = sum(1 for r in test_data if r["text"].strip() in train_texts)
    print(f"  Dev cases with identical text in Train: {dev_text_overlap}")
    print(f"  Test cases with identical text in Train: {test_text_overlap}")
    
    dev_texts = {r["text"].strip(): r["labels"] for r in dev_data}
    test_dev_text_overlap = sum(1 for r in test_data if r["text"].strip() in dev_texts)
    print(f"  Test cases with identical text in Dev: {test_dev_text_overlap}")

if __name__ == "__main__":
    main()
