import json
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from src.model.enc_dec import LegalEncoderDecoder, EncDecConfig

def generate_text(model, tokenizer, prompt_text, max_new_tokens=128, temperature=0.7, top_p=1.0, penalty=1.0):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    generated = input_ids.tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            decoder_input = torch.tensor([generated], device=device)
            # Encode-decode does not support past_key_values trivially in this custom implementation
            outputs = model(input_ids=torch.tensor([generated], device=device), decoder_input_ids=decoder_input)
                
            next_token_logits = outputs.logits[0, -1, :]
            
            if penalty != 1.0:
                for token_id in set(generated):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= penalty
                    else:
                        next_token_logits[token_id] *= penalty
            
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[len(input_ids[0]):], skip_special_tokens=True)

def main():
    print("Loading EncoderDecoder model...")
    checkpoint = "code/phase1/models/final_architectures/4_EncoderDecoder/epoch_10"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="code/phase1/models/tokenizer/legal_bpe_tokenizer.json")
    
    config = EncDecConfig.from_pretrained(checkpoint)
    model = LegalEncoderDecoder(config)
    model.load_state_dict(torch.load(f"{checkpoint}/model.safetensors", weights_only=True), strict=False)
    model.cuda().eval()
    
    print("Generating a test sample...")
    prompt = "The Supreme Court ruled that"
    output = generate_text(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
