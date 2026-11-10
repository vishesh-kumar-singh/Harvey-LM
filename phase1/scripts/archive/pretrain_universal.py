import os
import torch
from transformers import PreTrainedTokenizerFast
from src.model.universal_transformer import get_universal_model

def main():
    print("Initializing Universal Transformer Architecture...")
    model = get_universal_model()
    os.makedirs("models/universal_transformer", exist_ok=True)
    torch.save(model.state_dict(), "models/universal_transformer/pytorch_model.bin")
    print("Saved base Universal Transformer model for Phase 2!")

if __name__ == "__main__":
    main()
