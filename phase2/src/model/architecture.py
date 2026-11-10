import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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

def get_alibi_slopes(n_heads: int):
    start = (2**(-2**-(math.log2(n_heads)-3)))
    ratio = start
    return [start*ratio**i for i in range(n_heads)]

def build_alibi_tensor(seqlen: int, n_heads: int):
    slopes = torch.tensor(get_alibi_slopes(n_heads))
    context_position = torch.arange(seqlen)[:, None]
    memory_position = torch.arange(seqlen)[None, :]
    distance = memory_position - context_position # <= 0
    alibi = distance.unsqueeze(0) * slopes.view(n_heads, 1, 1)
    return alibi

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
        
        # QK-Norm Optimization (Dehghani et al., 2023)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cis, mask, alibi=None, past_key_value=None, use_cache=False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply QK-Norm per head
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

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

        past_seq_len = total_seq_len - seqlen

        if alibi is not None:
            alibi_sliced = alibi[:, past_seq_len:total_seq_len, :total_seq_len]
            if mask is None:
                causal = torch.tril(torch.ones(seqlen, total_seq_len, dtype=torch.bool, device=x.device), diagonal=past_seq_len)
                float_mask = torch.zeros(seqlen, total_seq_len, dtype=x.dtype, device=x.device)
                float_mask = float_mask.masked_fill(~causal, float("-inf"))
                float_mask = float_mask.unsqueeze(0).unsqueeze(0) + alibi_sliced.unsqueeze(0)
            else:
                float_mask = torch.zeros_like(mask, dtype=x.dtype)
                float_mask = float_mask.masked_fill(~mask, float("-inf"))
                float_mask = float_mask + alibi_sliced.unsqueeze(0)
            output = F.scaled_dot_product_attention(xq, xk_rep, xv_rep, attn_mask=float_mask, is_causal=False)
        else:
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
        
        # Novel Architectural Modification: Learnable Residual Gating
        self.attn_gate = nn.Parameter(torch.ones(1))
        self.ffn_gate = nn.Parameter(torch.ones(1))

    def forward(self, x, freqs_cis, mask, alibi=None, past_key_value=None, use_cache=False):
        # Novelty: Parallel Attention and FFN with Learnable Gates
        # The model independently calculates both branches and dynamically learns how to balance them
        attn_out, present_key_value = self.attention(self.attention_norm(x), freqs_cis, mask, alibi, past_key_value, use_cache)
        ffn_out = self.feed_forward(self.ffn_norm(x))
        
        out = x + (self.attn_gate * attn_out) + (self.ffn_gate * ffn_out)
        return out, present_key_value

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class HarveyTransformerConfig(PretrainedConfig):
    model_type = "harveylm"
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
        use_alibi=False,
        tie_word_embeddings=True,
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
        self.use_alibi = use_alibi
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def num_attention_heads(self):
        return self.n_heads
        
    @property
    def hidden_size(self):
        return self.dim

class HarveyLM(PreTrainedModel):
    config_class = HarveyTransformerConfig
    
    def __init__(self, config: HarveyTransformerConfig):
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
        
        if not config.use_alibi:
            self.register_buffer("freqs_cis", precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len), persistent=False)
        else:
            self.register_buffer("alibi", build_alibi_tensor(config.max_seq_len, config.n_heads), persistent=False)

        self.apply(self._init_weights)
        
        # Manual weight tying optimization (Press & Wolf, 2017)
        if getattr(config, "tie_word_embeddings", True):
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
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings



    def forward(self, input_ids=None, labels=None, attention_mask=None, past_key_values=None, use_cache=False, use_causal_mask=True, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            bsz, seqlen, _ = inputs_embeds.shape
            h = inputs_embeds
        else:
            bsz, seqlen = input_ids.shape
            h = self.tok_embeddings(input_ids)
        
        past_seq_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_seq_len = past_key_values.get_seq_length()
            else:
                past_seq_len = past_key_values[0][0].shape[1]
            
        total_seq_len = past_seq_len + seqlen
        
        freqs_cis = None
        alibi = None
        if not self.config.use_alibi:
            freqs_cis = self.freqs_cis[past_seq_len:total_seq_len].to(h.device)
        else:
            alibi = self.alibi[:, :total_seq_len, :total_seq_len].to(h.device)
            if not use_causal_mask:
                # Symmetrical ALiBi for bidirectional attention (Press et al., 2022)
                alibi = -torch.abs(alibi)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, total_seq_len).bool()
            if seqlen > 1 and use_causal_mask:
                causal_mask = torch.tril(torch.ones(seqlen, total_seq_len, dtype=torch.bool, device=h.device), diagonal=past_seq_len)
                mask = mask & causal_mask

        next_decoder_cache = () if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = None
            if past_key_values is not None:
                if hasattr(past_key_values, "key_cache"):
                    if i < len(past_key_values.key_cache):
                        past_key_value = (past_key_values.key_cache[i], past_key_values.value_cache[i])
                else:
                    past_key_value = past_key_values[i]
            
            if getattr(self, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                h, _ = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), h, freqs_cis, mask, alibi, None, False, use_reentrant=False)
            else:
                h, present_key_value = layer(h, freqs_cis, mask, alibi, past_key_value, use_cache)
                if use_cache:
                    next_decoder_cache += (present_key_value,)
            
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
            # Z-Loss Regularization (Chowdhery et al., 2022 - PaLM)
            # Penalizes large logits to improve training stability and final accuracy
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2).mean()
            loss = ce_loss + 1e-4 * z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=(h,) if kwargs.get("output_hidden_states", False) else None
        )


def get_300m_harvey_model(vocab_size=32000, context_length=2048, use_alibi=False):
    config = HarveyTransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=context_length,
        use_alibi=use_alibi
    )
    model = HarveyLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"HarveyLM instantiated with {total_params / 1e6:.2f}M parameters (ALiBi={use_alibi}).")
    return model

if __name__ == "__main__":
    model = get_300m_harvey_model(use_alibi=True)
