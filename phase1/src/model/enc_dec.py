import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import math
from src.model.architecture import RMSNorm, apply_rotary_emb, precompute_freqs_cis

class EncDecConfig(PretrainedConfig):
    model_type = "legal_enc_dec"
    def __init__(
        self,
        vocab_size=32000,
        d_model=768,
        nhead=12,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_len=2048,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, q_x, k_x, v_x, mask=None):
        bsz, q_len, _ = q_x.shape
        _, kv_len, _ = k_x.shape
        
        xq = self.wq(q_x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(k_x).view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(v_x).view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, eps=1e-5):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ffn = FeedForward(dim, hidden_dim)
        self.norm1 = RMSNorm(dim, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, eps=1e-5):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.ffn = FeedForward(dim, hidden_dim)
        self.norm1 = RMSNorm(dim, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.norm3 = RMSNorm(dim, eps=eps)
    def forward(self, x, enc_out, self_mask, cross_mask):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), self_mask)
        x = x + self.cross_attn(self.norm2(x), enc_out, enc_out, cross_mask)
        x = x + self.ffn(self.norm3(x))
        return x

class LegalEncoderDecoder(PreTrainedModel):
    config_class = EncDecConfig
    def __init__(self, config: EncDecConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoder = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.encoders = nn.ModuleList([EncoderBlock(config.d_model, config.nhead, config.dim_feedforward) for _ in range(config.num_encoder_layers)])
        self.decoders = nn.ModuleList([DecoderBlock(config.d_model, config.nhead, config.dim_feedforward) for _ in range(config.num_decoder_layers)])
        
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if getattr(config, "tie_word_embeddings", True):
            self.output.weight = self.embedding.weight
            self._tied_weights_keys = {'output.weight': True}
            
    def get_input_embeddings(self):
        return self.embedding
        
    def set_input_embeddings(self, value):
        self.embedding = value
        
    def forward(self, input_ids, decoder_input_ids=None, labels=None, attention_mask=None, **kwargs):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
            
        bsz, seqlen = input_ids.shape
        _, dec_seqlen = decoder_input_ids.shape
        
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        dec_positions = torch.arange(dec_seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, dec_seqlen)
        
        emb = self.embedding(input_ids) + self.pos_encoder(positions)
        dec_emb = self.embedding(decoder_input_ids) + self.pos_encoder(dec_positions)
        
        enc_mask = None
        if attention_mask is not None:
            enc_mask = (1.0 - attention_mask.float()) * -10000.0
            enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)
            
        causal_mask = torch.triu(torch.full((dec_seqlen, dec_seqlen), float('-inf'), device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, dec_seqlen, dec_seqlen)
        
        cross_mask = None
        if attention_mask is not None:
            cross_mask = (1.0 - attention_mask.float()) * -10000.0
            cross_mask = cross_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, dec_seqlen, seqlen)

        h = emb
        for layer in self.encoders:
            h = layer(h, enc_mask)
            
        enc_out = self.norm(h)
        
        h = dec_emb
        for layer in self.decoders:
            h = layer(h, enc_out, causal_mask, cross_mask)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            
        return Seq2SeqLMOutput(loss=loss, logits=logits)

def get_enc_dec_model(vocab_size=32000, context_length=2048):
    config = EncDecConfig(vocab_size=vocab_size, max_seq_len=context_length)
    model = LegalEncoderDecoder(config)
    return model
