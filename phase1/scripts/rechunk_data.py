import os
import multiprocessing
from datasets import load_from_disk, Dataset

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "data", "processed", "train_dataset")
    out_path_2048 = os.path.join(base_dir, "data", "processed", "train_dataset_2048")
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Original dataset size: {len(dataset)} chunks of length {len(dataset[0]['input_ids'])}")
    
    # We want to re-chunk the dataset to 2048
    MAX_SEQ_LENGTH = 2048
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder
        total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
        
        # Split by chunks of max_len
        result = {
            k: [t[i : i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Re-chunking into 2048-token chunks...")
    num_cores = min(16, max(1, multiprocessing.cpu_count() - 2))
    
    rechunked_dataset = dataset.map(
        group_texts,
        batched=True,
        batch_size=10000,
        num_proc=num_cores
    )
    
    print(f"New dataset size: {len(rechunked_dataset)} chunks of length {len(rechunked_dataset[0]['input_ids'])}")
    
    print(f"Saving to {out_path_2048}...")
    rechunked_dataset.save_to_disk(out_path_2048)
    print("Done!")

if __name__ == "__main__":
    main()
