import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.architecture import HarveyTransformerConfig, RMSNorm, Attention, precompute_freqs_cis, build_alibi_tensor, apply_rotary_emb

class MoEFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Experts
        self.experts_w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.experts_w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.experts_w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        
        nn.init.normal_(self.experts_w1, std=0.02)
        nn.init.normal_(self.experts_w2, std=0.02)
        nn.init.normal_(self.experts_w3, std=0.02)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
        routing_weights /= routing_weights.sum(dim=1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i)
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) > 0:
                expert_inputs = x_flat[token_indices]
                
                # Manual feedforward for the expert
                w1 = self.experts_w1[i]
                w2 = self.experts_w2[i]
                w3 = self.experts_w3[i]
                
                h1 = F.silu(expert_inputs.matmul(w1))
                h3 = expert_inputs.matmul(w3)
                expert_outputs = (h1 * h3).matmul(w2)
                
                weights = routing_weights[token_indices, k_indices].unsqueeze(-1)
                expert_outputs = expert_outputs * weights
                
                final_output.index_add_(0, token_indices, expert_outputs.to(final_output.dtype))
                
        return final_output.view(bsz, seqlen, dim)

class MoETransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = MoEFeedForward(dim, hidden_dim, num_experts=4, top_k=2)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn_gate = nn.Parameter(torch.ones(1))
        self.ffn_gate = nn.Parameter(torch.ones(1))

    def forward(self, x, freqs_cis, mask, alibi=None, past_key_value=None, use_cache=False):
        attn_out, present_key_value = self.attention(self.attention_norm(x), freqs_cis, mask, alibi, past_key_value, use_cache)
        ffn_out = self.feed_forward(self.ffn_norm(x))
        out = x + (self.attn_gate * attn_out) + (self.ffn_gate * ffn_out)
        return out, present_key_value

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class HarveyMoE(PreTrainedModel):
    config_class = HarveyTransformerConfig
    
    def __init__(self, config: HarveyTransformerConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(MoETransformerBlock(
                layer_id, config.dim, config.n_heads, config.n_kv_heads, config.hidden_dim, config.norm_eps
            ))
            
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        if not config.use_alibi:
            self.register_buffer("freqs_cis", precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len), persistent=False)
        else:
            self.register_buffer("alibi", build_alibi_tensor(config.max_seq_len, config.n_heads), persistent=False)

        if getattr(config, "tie_word_embeddings", True):
            self.output.weight = self.tok_embeddings.weight
            self._tied_weights_keys = {'output.weight': True}

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        
        past_seq_len = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        total_seq_len = seqlen + past_seq_len
        if not self.config.use_alibi:
            if total_seq_len > self.freqs_cis.size(0):
                freqs_cis = precompute_freqs_cis(self.config.dim // self.config.n_heads, total_seq_len).to(h.device)[past_seq_len:total_seq_len]
            else:
                freqs_cis = self.freqs_cis[past_seq_len:total_seq_len].to(h.device)
            alibi = None
        else:
            if total_seq_len > self.alibi.size(1):
                alibi = build_alibi_tensor(total_seq_len, self.config.n_heads).to(h.device)
            else:
                alibi = self.alibi[:, :total_seq_len, :total_seq_len].to(h.device)
            freqs_cis = None

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, total_seq_len).bool()
            causal_mask = torch.tril(torch.ones(seqlen, total_seq_len, dtype=torch.bool, device=h.device), diagonal=past_seq_len)
            mask = mask & causal_mask

        presents = () if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            h, present = layer(h, freqs_cis, mask, alibi, past_key_value, use_cache)
            if use_cache:
                presents += (present,)
            
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents)

def get_moe_model(vocab_size=32000, context_length=2048):
    config = HarveyTransformerConfig(vocab_size=vocab_size, max_seq_len=context_length, n_layers=12, dim=768, n_heads=12, hidden_dim=2048)
    model = HarveyMoE(config)
    return model
