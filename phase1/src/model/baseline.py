import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # shape: (1, seqlen, 1, head_dim/2)
    
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:, :seq_len, :, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask, past_key_value=None, use_cache=False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
            
        present_key_value = (xk, xv) if use_cache else None
        total_seq_len = xk.shape[1]

        if self.n_rep > 1:
            xk_rep = xk[:, :, :, None, :].expand(bsz, total_seq_len, self.n_kv_heads, self.n_rep, self.head_dim).reshape(bsz, total_seq_len, self.n_heads, self.head_dim)
            xv_rep = xv[:, :, :, None, :].expand(bsz, total_seq_len, self.n_kv_heads, self.n_rep, self.head_dim).reshape(bsz, total_seq_len, self.n_heads, self.head_dim)
        else:
            xk_rep = xk
            xv_rep = xv
            
        xq = xq.transpose(1, 2)
        xk_rep = xk_rep.transpose(1, 2)
        xv_rep = xv_rep.transpose(1, 2)

        is_causal = (mask is None) and (seqlen > 1) and (past_key_value is None)
        output = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, attn_mask=mask, is_causal=is_causal)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), present_key_value

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x, freqs_cis, mask, past_key_value=None, use_cache=False):
        attn_out, present_key_value = self.attention(self.attention_norm(x), freqs_cis, mask, past_key_value, use_cache)
        x = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_out
        return x, present_key_value

class BaselineConfig(PretrainedConfig):
    model_type = "baseline_lm"
    def __init__(
        self,
        vocab_size=32000,
        dim=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,
        hidden_dim=2752,
        norm_eps=1e-6,
        max_seq_len=2048,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

class BaselineLM(PreTrainedModel):
    config_class = BaselineConfig
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(
                layer_id, config.dim, config.n_heads, config.n_kv_heads, config.hidden_dim, config.norm_eps
            ))
            
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len), persistent=False)

        if getattr(config, "tie_word_embeddings", False):
            self.output.weight = self.tok_embeddings.weight
            self._tied_weights_keys = {'output.weight': True}
            
    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output
        
    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        
        past_seq_len = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        total_seq_len = seqlen + past_seq_len
        if total_seq_len > self.freqs_cis.size(0):
            freqs_cis = precompute_freqs_cis(self.config.dim // self.config.n_heads, total_seq_len).to(h.device)[past_seq_len:total_seq_len]
        else:
            freqs_cis = self.freqs_cis[past_seq_len:total_seq_len].to(h.device)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, total_seq_len).bool()
            causal_mask = torch.tril(torch.ones(seqlen, total_seq_len, dtype=torch.bool, device=h.device), diagonal=past_seq_len)
            mask = mask & causal_mask

        presents = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            h, present = layer(h, freqs_cis, mask, past_key_value, use_cache)
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

def get_baseline_model(vocab_size=32000, context_length=2048):
    config = BaselineConfig(vocab_size=vocab_size, max_seq_len=context_length, n_layers=16, dim=1024, n_heads=16, hidden_dim=2752)
    return BaselineLM(config)
