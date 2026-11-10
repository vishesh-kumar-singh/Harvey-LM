import os
import torch
from transformers import PreTrainedTokenizerFast
from src.model.local_attention import get_local_attention_model

def main():
    print("Initializing Local Attention Architecture...")
    model = get_local_attention_model()
    # Save a dummy pre-trained checkpoint right away so we have it for Phase 2
    os.makedirs("models/local_attention", exist_ok=True)
    torch.save(model.state_dict(), "models/local_attention/pytorch_model.bin")
    print("Saved base Local Attention model for Phase 2!")

if __name__ == "__main__":
    main()
