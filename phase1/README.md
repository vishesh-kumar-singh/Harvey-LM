# Phase 1 — Legal Language Model Pre-training

Train compact legal language models **from scratch** on the official Phase-1 corpus (Acts, High
Courts, Supreme Court, Tribunals). For each 512-word test prompt, generate a 128-word continuation.
Metric: BERTScore F1 (secondary ROUGE-L, BLEU-4).

## Architectures

All four models are implemented in `src/model/` without any library model classes. They share one
32k-vocabulary BPE tokenizer (whitespace pre-tokenisation) (`scripts/train_tokenizer.py` →
`models/tokenizer/legal_bpe_tokenizer.json`).

| # | Model | File | Layers × d | Heads (KV) | Positional | Distinctive design | Params |
|:-:|---|---|:-:|:-:|---|---|:-:|
| 1 | **BaselineLM** | `src/model/baseline.py` | 16 × 1024 | 16 (4) GQA | RoPE | pre-norm RMSNorm, SwiGLU (2752), separate LM head | 242.8 M |
| 2 | **HarveyLM** | `src/model/architecture.py` | 16 × 1024 | 16 (4) GQA | ALiBi | QK-RMSNorm, parallel attention + FFN block, tied input/output embeddings, z-loss | 210.0 M |
| 3 | **HarveyMoE** | `src/model/moe.py` | 12 × 768 | 12 (4) | RoPE (`use_alibi=false`) | FFN replaced by 4 SwiGLU experts, softmax top-2 router | 270.0 M |
| 4 | **LegalEncoderDecoder** | `src/model/enc_dec.py` | 6 enc + 6 dec × 768 | 12 | learned absolute | bidirectional encoder, causal decoder with cross-attention, tied embeddings | 125.3 M |

An additional ablation, `checkpoints/alibi_experiment` (BaselineLM with ALiBi in place of RoPE,
242.8 M, 512 context), was used to isolate the effect of the positional scheme.

Optimiser: Muon (`src/model/optim_muon.py`) for HarveyLM, with bf16 mixed precision via `accelerate`.

## Data pipeline

```bash
python src/data/download.py                 # official corpus -> data/raw/
python scripts/train_tokenizer.py           # BPE tokenizer -> models/tokenizer/
python src/data/process_data.py             # tokenise + chunk -> data/processed/
python scripts/rechunk_data.py              # re-chunk to longer contexts (2048) for later stages
```

`process_data.py` writes the corpus in **curriculum order** (Supreme Court → High Courts →
Tribunals → Acts). The last 1% of chunks is held out as a validation split; it is not shuffled, so the
ordering is preserved.

## Training

| Model | Script | Notes |
|---|---|---|
| BaselineLM / ALiBi ablation | `scripts/pretrain.py` (set `EXPERIMENT_NAME`, `USE_ALIBI`) | 5 epochs |
| HarveyLM | `scripts/pretrain.py` then `scripts/archive/pretrain_epoch{12,13,20,30_balanced,50}.py` | staged run, 40 epochs total: 512-token chunks, then 2048-token context extension, then a domain-balanced annealing stage |
| HarveyMoE | `scripts/transfer_weights_moe.py` (initialises the MoE from HarveyLM epoch 40), then `scripts/pretrain_moe.py` | 10 epochs, 2048 context |
| LegalEncoderDecoder | `scripts/pretrain_enc_dec.py` | 10 epochs, 2048 context |

Each script's hyper-parameters are module-level constants at the top of the file (batch size,
gradient accumulation, sequence length, learning rate, resume checkpoint). Training resumes from
`RESUME_FROM_CHECKPOINT` when that is set.

**Checkpoint averaging.** `scripts/archive/average_checkpoints.py` averages the weights of the
HarveyLM checkpoints from **epochs 28, 30 and 40** and saves the result as
`checkpoints/harveylm_experiment3_improved/epoch_99`. That averaged model, **not a 99th epoch**, is the
HarveyLM backbone used in Phases 2–7.
`scripts/slerp_merge.py` and `scripts/dare_merge.py` are alternative merging experiments.

The four backbones used downstream are symlinked in `models/final_architectures/`:
`1_Baseline` (epoch 5), `2_HarveyLM` (epoch_99 = averaged), `3_HarveyMoE` (epoch 10) and
`4_EncoderDecoder` (epoch 10).

**Retained checkpoints** (`models/checkpoints/`):

| Run | Kept | Reason |
|---|---|---|
| `baseline/` | `epoch_5` | final BaselineLM |
| `harveylm_experiment3_improved/` | `epoch_28`, `epoch_30`, `epoch_40`, `epoch_99` | sources of the average, and the average itself; `epoch_40` also initialises HarveyMoE |
| `moe_experiment/` | `epoch_10` | final HarveyMoE |
| `enc_dec_experiment/` | `epoch_10` | final LegalEncoderDecoder |
| `alibi_experiment/` | `epoch_5` | final checkpoint of the RoPE→ALiBi ablation |

Intermediate epochs and an early HarveyLM bf16 trial run were removed from the working tree.

## Generation and submission

```bash
python scripts/generate_submission_cli.py --epoch 99 --temperature <T> --top_p <p> --penalty <repetition_penalty>
# per-architecture generators: scripts/generate_{baseline,harveylm,moe,enc_dec}.py
# checkpoint ensemble:          scripts/generate_ensemble.py
# decoding grid search:         scripts/archive/run_grid.py (temperature × top-p × repetition penalty)
```

These write `prediction.json` (`{"id": {"text": "<128-word continuation>"}}`) and `submission.zip`.

## Submissions

| Folder | System | Leaderboard BERTScore F1 |
|---|---|:-:|
| `submissions/final/` | HarveyLM, averaged checkpoint | _fill in from Codabench_ |

Only the final `prediction.json` was kept locally. Scores for the other Phase-1 submissions are on
the Codabench submission history.

## Logs

`logs/` holds the pre-training logs (`pretrain_enc_dec_new.log`, `pretrain_moe_resume.log`) and the
per-epoch training losses (`training_losses.txt`).
