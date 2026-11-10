import os
import json
import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_training_corpus(data_paths, chunk_size=1000):
    """
    Generator that yields chunks of text from the JSON files.
    """
    for path in data_paths:
        print(f"Processing {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # handle both list and dict structures
            if isinstance(data, dict):
                items = data.values()
            else:
                items = data
                
            texts = []
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                    
                if len(texts) >= chunk_size:
                    yield texts
                    texts = []
            if texts:
                yield texts

def train_custom_bpe(data_dir, output_dir, vocab_size=32000):
    """
    Trains a Custom BPE tokenizer on the legal corpus.
    """
    print(f"Searching for training data in: {data_dir}")
    train_files = glob.glob(os.path.join(data_dir, "Train", "*.json"))
    
    if not train_files:
        raise ValueError(f"No JSON files found in {os.path.join(data_dir, 'Train')}")
        
    print(f"Found {len(train_files)} files for tokenizer training.")

    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
        show_progress=True
    )

    # Train
    print(f"Training tokenizer with vocab size {vocab_size}...")
    tokenizer.train_from_iterator(get_training_corpus(train_files), trainer=trainer)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "legal_bpe_tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")

if __name__ == "__main__":
    # Base paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "Phase_1_Pretraining")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "tokenizer")
    
    train_custom_bpe(DATA_DIR, OUTPUT_DIR, vocab_size=32000)
