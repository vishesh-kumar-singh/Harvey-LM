import os
import json
import torch
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from src.model.architecture import HarveyLM

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def ensemble_generate(models, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id, top_p=0.9, temperature=0.3, repetition_penalty=1.2):
    past_key_values_list = [None] * len(models)
    batch_size = input_ids.shape[0]
    
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    all_generated = []
    
    full_sequence = input_ids.clone()

    for _ in range(max_new_tokens):
        ensemble_logits = []
        new_past_key_values_list = []
        
        for i, model in enumerate(models):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values_list[i],
                use_cache=True
            )
            ensemble_logits.append(outputs.logits[:, -1, :])
            new_past_key_values_list.append(outputs.past_key_values)
            
        past_key_values_list = new_past_key_values_list
        
        # Average the logits across all models
        logits = torch.stack(ensemble_logits, dim=0).mean(dim=0)
        
        if repetition_penalty != 1.0:
            score = torch.gather(logits, 1, full_sequence)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, full_sequence, score)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        next_token = next_token.squeeze(-1)
        next_token = torch.where(unfinished, next_token, torch.tensor(pad_token_id, device=next_token.device))
        
        all_generated.append(next_token.unsqueeze(-1))
        full_sequence = torch.cat([full_sequence, next_token.unsqueeze(-1)], dim=-1)
        
        unfinished = unfinished & (next_token != eos_token_id)
        if not unfinished.any():
            break
            
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        
    return torch.cat(all_generated, dim=-1)

def load_model(checkpoint_dir, device):
    from src.model.architecture import HarveyTransformerConfig
    import safetensors.torch
    config = HarveyTransformerConfig.from_pretrained(checkpoint_dir)
    model = HarveyLM(config)
    state_dict = safetensors.torch.load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict, strict=False)
    model.gradient_checkpointing = False
    model = model.to(torch.bfloat16)
    
    from src.model.architecture import precompute_freqs_cis, build_alibi_tensor
    if config.use_alibi:
        model.alibi = build_alibi_tensor(2048, config.n_heads).to(device).to(torch.bfloat16)
    else:
        model.freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, 2048).to(device)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs='+', required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = []
    print(f"Loading Ensembled Models for epochs: {args.epochs}")
    for ep in args.epochs:
        ckpt_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved", f"epoch_{ep}")
        print(f"Loading {ckpt_dir}...")
        models.append(load_model(ckpt_dir, device))
    
    checkpoint_dir = os.path.join(base_dir, "models", "checkpoints", "harveylm_experiment3_improved", f"epoch_{args.epochs[0]}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        
    test_path = os.path.join(base_dir, "data", "raw", "Phase_1_Pretraining", "Test", "test.json")
    print(f"Loading Test Data from {test_path}...")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = {}
    batch_size = 5
    items = list(test_data.items())
    
    print(f"Generating predictions with Ensembled Logits (Temp={args.temperature}, Penalty={args.penalty})...")
    for i in tqdm(range(0, len(items), batch_size)):
        batch_items = items[i:i+batch_size]
        batch_ids = [item[0] for item in batch_items]
        batch_texts = [item[1]["text"] for item in batch_items]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            gen_tokens = ensemble_generate(
                models=models,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=400,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.penalty
            )
            
        for j, out_ids in enumerate(gen_tokens):
            gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            words = gen_text.split()
            if len(words) > 128:
                words = words[:128]
            
            final_text = " ".join(words)
            predictions[batch_ids[j]] = {"text": final_text}

    pred_name = f"prediction.json"
    pred_path = os.path.join(base_dir, pred_name)
    print(f"Saving predictions to {pred_path}...")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
    import zipfile
    zip_name = "ensemble_predictions.zip"
    zip_path = os.path.join(base_dir, zip_name)
    print(f"Creating submission zip at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_path, arcname=pred_name)
        
    print("Done!")

if __name__ == "__main__":
    main()
