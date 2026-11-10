"""
Phase 6: Universal Legal Summarizer supporting all 4 Phase 1 backbones:
1. LegalEncoderDecoder (125M) - Native Seq2Seq
2. HarveyLM (219M) - ALiBi Causal LM
3. HarveyMoE (420M / 100M active) - Mixture of Experts Causal LM
4. BaselineLM (217M) - RoPE Causal LM

Supports:
- LORMA (Truncated SVD - Compulsory)
- LoRA
- QLoRA (4-bit NF4)
- Knowledge Distillation Loss
- Fast cached autoregressive generation with trigram blocking & repetition penalty.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

ROOT_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PHASE1_DIR = os.path.join(ROOT_CODE, "phase1")
if ROOT_CODE not in sys.path:
    sys.path.insert(0, ROOT_CODE)
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from phase1.src.model.baseline import BaselineLM, BaselineConfig, precompute_freqs_cis
from phase1.src.model.architecture import HarveyLM, HarveyTransformerConfig, build_alibi_tensor
from phase1.src.model.moe import HarveyMoE
from phase1.src.model.enc_dec import LegalEncoderDecoder, EncDecConfig
from safetensors.torch import load_file
from phase3.src.model.lorma_from_scratch import apply_lorma_to_model, mark_only_lorma_as_trainable
from phase3.src.model.qlora_from_scratch import apply_qlora_to_model, mark_only_qlora_as_trainable
from phase2.src.model.lora_from_scratch import apply_lora_to_model, mark_only_lora_as_trainable

ARCH_CKPTS = {
    "baseline": os.path.join(ROOT_CODE, "phase1/models/final_architectures/1_Baseline/epoch_5"),
    "harveylm": os.path.join(ROOT_CODE, "phase1/models/final_architectures/2_HarveyLM/epoch_99"),
    "moe":      os.path.join(ROOT_CODE, "phase1/models/final_architectures/3_HarveyMoE/epoch_10"),
    "enc_dec":  os.path.join(ROOT_CODE, "phase1/models/final_architectures/4_EncoderDecoder/epoch_10")
}

ARCH_TARGETS = {
    "baseline": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "harveylm": ["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    "moe":      ["wq", "wk", "wv", "wo"],
    "enc_dec":  ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
}


class GenerationKDLoss(nn.Module):
    """Sequence-level Knowledge Distillation Loss for Abstractive Summarization."""
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss on hard labels
        shift_s_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(shift_s_logits.view(-1, student_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # Distillation loss on non-ignored tokens
        mask = (shift_labels != -100)
        if not mask.any():
            return ce_loss

        s_logits_flat = shift_s_logits[mask] / self.temperature
        t_logits_flat = (teacher_logits[..., :-1, :].contiguous()[mask] / self.temperature).detach()

        kl_loss = F.kl_div(
            F.log_softmax(s_logits_flat, dim=-1),
            F.softmax(t_logits_flat, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss


class UniversalLegalSummarizer(nn.Module):
    """
    Universal Legal Summarization Model wrapper.
    """
    def __init__(
        self,
        arch_type: str = "enc_dec",
        ckpt_dir: Optional[str] = None,
        adaptation: str = "lorma",
        r: int = 32,
        lorma_alpha: int = 64,
        kd_temperature: float = 2.0,
        kd_alpha: float = 0.5
    ):
        super().__init__()
        self.arch_type = arch_type.lower()
        self.adaptation = adaptation.lower()

        if ckpt_dir is None:
            ckpt_dir = ARCH_CKPTS.get(self.arch_type)
        self.ckpt_dir = ckpt_dir

        print(f"Initializing UniversalLegalSummarizer (arch='{self.arch_type}', adaptation='{self.adaptation}') from {ckpt_dir}...")

        if self.arch_type in ["enc_dec", "encoder_decoder", "legal_enc_dec"]:
            self.config = EncDecConfig.from_pretrained(ckpt_dir)
            self.backbone = LegalEncoderDecoder(self.config)
            self.is_seq2seq = True
        elif self.arch_type == "harveylm":
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyLM(self.config)
            self.is_seq2seq = False
        elif self.arch_type in ["moe", "harveymoe"]:
            self.config = HarveyTransformerConfig.from_pretrained(ckpt_dir)
            self.backbone = HarveyMoE(self.config)
            self.is_seq2seq = False
        elif self.arch_type == "baseline":
            self.config = BaselineConfig.from_pretrained(ckpt_dir)
            self.backbone = BaselineLM(self.config)
            self.is_seq2seq = False
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

        # Load weights
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safetensor_path):
            st = load_file(safetensor_path)
            self.backbone.load_state_dict(st, strict=False)
            print(f"Loaded weights from {safetensor_path}")
        elif os.path.exists(bin_path):
            st = torch.load(bin_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(st, strict=False)
            print(f"Loaded weights from {bin_path}")

        # Apply Adaptation Strategy
        targets = ARCH_TARGETS.get(self.arch_type, ["wq", "wk", "wv", "wo"])
        if self.adaptation == "lorma":
            count = apply_lorma_to_model(self.backbone, r=r, lorma_alpha=lorma_alpha, target_modules=targets)
            mark_only_lorma_as_trainable(self.backbone)
            print(f"Applied LORMA (r={r}, alpha={lorma_alpha}) to {count} modules.")
        elif self.adaptation == "lora":
            count = apply_lora_to_model(self.backbone, r=r, lora_alpha=lorma_alpha, target_modules=targets)
            mark_only_lora_as_trainable(self.backbone)
            print(f"Applied LoRA (r={r}, alpha={lorma_alpha}) to {count} modules.")
        elif self.adaptation == "qlora":
            count = apply_qlora_to_model(self.backbone, r=r, lora_alpha=lorma_alpha, target_modules=targets)
            mark_only_qlora_as_trainable(self.backbone)
            print(f"Applied QLoRA (r={r}, alpha={lorma_alpha}) to {count} modules.")
        elif self.adaptation in ["kd", "distillation"]:
            # Student model in KD: full student parameters trainable
            for p in self.backbone.parameters():
                p.requires_grad = True
            print("Configured for Knowledge Distillation training (full student parameters trainable).")
        elif self.adaptation == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
            print("Configured for Full Model Fine-Tuning.")

        self.kd_loss_fn = GenerationKDLoss(temperature=kd_temperature, alpha=kd_alpha)

        # Precompute extended positional representations (4096 tokens) for long context modeling
        if hasattr(self.backbone, "freqs_cis"):
            dim = self.config.dim // self.config.n_heads
            self.backbone.register_buffer("freqs_cis", precompute_freqs_cis(dim, 4096), persistent=False)
        elif hasattr(self.backbone, "alibi"):
            self.backbone.register_buffer("alibi", build_alibi_tensor(4096, self.config.n_heads), persistent=False)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes document text through LegalEncoderDecoder encoder stack."""
        assert self.is_seq2seq, "encode() is only valid for Seq2Seq models."
        bsz, seqlen = input_ids.shape
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        emb = self.backbone.embedding(input_ids) + self.backbone.pos_encoder(positions)
        enc_mask = None
        if attention_mask is not None:
            enc_mask = (1.0 - attention_mask.float()) * -10000.0
            enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, seqlen, seqlen)
        h = emb
        for layer in self.backbone.encoders:
            h = layer(h, enc_mask)
        return self.backbone.norm(h)

    def decode_step(
        self,
        decoder_input_ids: torch.Tensor,
        enc_out: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Runs decoder layers over encoder memory for current decoder tokens."""
        assert self.is_seq2seq, "decode_step() is only valid for Seq2Seq models."
        bsz, dec_seqlen = decoder_input_ids.shape
        _, enc_seqlen, _ = enc_out.shape
        dec_positions = torch.arange(dec_seqlen, device=decoder_input_ids.device).unsqueeze(0).expand(bsz, dec_seqlen)
        dec_emb = self.backbone.embedding(decoder_input_ids) + self.backbone.pos_encoder(dec_positions)

        causal_mask = torch.triu(torch.full((dec_seqlen, dec_seqlen), float('-inf'), device=decoder_input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, dec_seqlen, dec_seqlen)

        cross_mask = None
        if attention_mask is not None:
            cross_mask = (1.0 - attention_mask.float()) * -10000.0
            cross_mask = cross_mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, dec_seqlen, enc_seqlen)

        h = dec_emb
        for layer in self.backbone.decoders:
            h = layer(h, enc_out, causal_mask, cross_mask)
        h = self.backbone.norm(h)
        return self.backbone.output(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if self.is_seq2seq:
            out = self.backbone(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = out.loss
            logits = out.logits

            if teacher_logits is not None and labels is not None:
                loss = self.kd_loss_fn(logits, teacher_logits, labels)

            return {"loss": loss, "logits": logits}

        else:
            # Causal LLM forward
            out = self.backbone(input_ids)
            logits = out.logits if hasattr(out, "logits") else out

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

                if teacher_logits is not None:
                    loss = self.kd_loss_fn(logits, teacher_logits, labels)

            return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 768,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 3,
        dynamic_holding_injection: bool = False,
        holding_prompt_ids: Optional[List[int]] = None,
        holding_injection_step: int = 620
    ) -> List[List[int]]:
        """
        Fast autoregressive generation with trigram blocking and repetition penalty.
        """
        self.eval()
        bsz = input_ids.size(0)
        device = input_ids.device

        if self.is_seq2seq:
            enc_out = self.encode(input_ids, attention_mask)
            cur_tokens = torch.full((bsz, 1), bos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(bsz, dtype=torch.bool, device=device)

            for step in range(max_new_tokens):
                logits = self.decode_step(cur_tokens, enc_out, attention_mask)
                next_token_logits = logits[:, -1, :].clone()

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(bsz):
                        prev_tokens = cur_tokens[b].tolist()
                        for token in set(prev_tokens):
                            if next_token_logits[b, token] > 0:
                                next_token_logits[b, token] /= repetition_penalty
                            else:
                                next_token_logits[b, token] *= repetition_penalty

                # No-repeat n-gram blocking
                if no_repeat_ngram_size > 0 and cur_tokens.size(1) >= no_repeat_ngram_size:
                    for b in range(bsz):
                        seq = cur_tokens[b].tolist()
                        ngram_prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                        banned = set()
                        for i in range(len(seq) - no_repeat_ngram_size + 1):
                            if tuple(seq[i: i + no_repeat_ngram_size - 1]) == ngram_prefix:
                                banned.add(seq[i + no_repeat_ngram_size - 1])
                        for token in banned:
                            next_token_logits[b, token] = float('-inf')

                # Greedy selection
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # [bsz]

                # Update finished status
                finished = finished | (next_tokens == eos_token_id)
                next_tokens = torch.where(finished, torch.tensor(pad_token_id, device=device), next_tokens)

                cur_tokens = torch.cat([cur_tokens, next_tokens.unsqueeze(1)], dim=1)

                if finished.all():
                    break

            # Convert to list of token IDs (excluding BOS and trailing PAD/EOS)
            results = []
            for b in range(bsz):
                tokens = cur_tokens[b].tolist()[1:]  # skip BOS
                clean = []
                for t in tokens:
                    if t in [eos_token_id, pad_token_id]:
                        break
                    clean.append(t)
                results.append(clean)
            return results

        else:
            # Causal LLM generation with KV Caching
            cur_tokens = input_ids.clone()
            orig_len = cur_tokens.size(1)
            finished = torch.zeros(bsz, dtype=torch.bool, device=device)

            # 1. Prefill pass over prompt context
            out = self.backbone(cur_tokens, use_cache=True)
            past_key_values = getattr(out, "past_key_values", None)
            logits = out.logits if hasattr(out, "logits") else out
            next_token_logits = logits[:, -1, :].clone()

            generated_tokens = [[] for _ in range(bsz)]

            for step in range(max_new_tokens):
                # Repetition penalty over generated tokens
                if repetition_penalty != 1.0:
                    for b in range(bsz):
                        for token in set(generated_tokens[b]):
                            if next_token_logits[b, token] > 0:
                                next_token_logits[b, token] /= repetition_penalty
                            else:
                                next_token_logits[b, token] *= repetition_penalty

                # Trigram blocking over generated tokens
                if no_repeat_ngram_size > 0:
                    for b in range(bsz):
                        gen_seq = generated_tokens[b]
                        if len(gen_seq) >= no_repeat_ngram_size:
                            ngram_prefix = tuple(gen_seq[-(no_repeat_ngram_size - 1):])
                            banned = set()
                            for i in range(len(gen_seq) - no_repeat_ngram_size + 1):
                                if tuple(gen_seq[i: i + no_repeat_ngram_size - 1]) == ngram_prefix:
                                    banned.add(gen_seq[i + no_repeat_ngram_size - 1])
                            for token in banned:
                                next_token_logits[b, token] = float('-inf')

                next_tokens = torch.argmax(next_token_logits, dim=-1)
                finished = finished | (next_tokens == eos_token_id)
                if finished.all():
                    break

                for b in range(bsz):
                    if not finished[b]:
                        generated_tokens[b].append(next_tokens[b].item())

                # 2. Decode step: 1 token pass using past_key_values
                if past_key_values is not None:
                    out = self.backbone(next_tokens.unsqueeze(1), past_key_values=past_key_values, use_cache=True)
                    past_key_values = getattr(out, "past_key_values", None)
                    logits = out.logits if hasattr(out, "logits") else out
                else:
                    cur_tokens = torch.cat([cur_tokens, next_tokens.unsqueeze(1)], dim=1)
                    out = self.backbone(cur_tokens)
                    logits = out.logits if hasattr(out, "logits") else out

                # Dynamic holding injection at targeted step to ensure complete ratio decidendi & disposition
                if dynamic_holding_injection and holding_prompt_ids and step == holding_injection_step:
                    for b in range(bsz):
                        if not finished[b]:
                            for inj_id in holding_prompt_ids:
                                generated_tokens[b].append(inj_id)
                                inj_tensor = torch.tensor([[inj_id]], dtype=torch.long, device=device)
                                if past_key_values is not None:
                                    out = self.backbone(inj_tensor, past_key_values=past_key_values, use_cache=True)
                                    past_key_values = getattr(out, "past_key_values", None)
                                    logits = out.logits if hasattr(out, "logits") else out
                                else:
                                    cur_tokens = torch.cat([cur_tokens, inj_tensor], dim=1)
                                    out = self.backbone(cur_tokens)
                                    logits = out.logits if hasattr(out, "logits") else out

                next_token_logits = logits[:, -1, :].clone()

            results = []
            for b in range(bsz):
                clean = []
                for t in generated_tokens[b]:
                    if t in [eos_token_id, pad_token_id]:
                        break
                    clean.append(t)
                results.append(clean)
            return results
