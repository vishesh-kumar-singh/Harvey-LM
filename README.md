# In-House Legal Language Models — CS781 Capstone

A compact legal language foundation trained **from scratch** on the official Indian legal corpus, then
adapted to six downstream legal NLP tasks. Every model in this repository starts from one of four
Phase-1 architectures that were implemented and pre-trained here; no external pretrained weights,
external legal data, or LLM APIs are used for training or inference.

| Phase | Task | Metric | Best leaderboard score | Best system |
|:-:|---|---|:-:|---|
| 1 | Legal LM pre-training (128-word continuation) | BERTScore F1 | _see [phase1](phase1/README.md)_ | HarveyLM (weight-averaged checkpoint) |
| 2 | Legal NER (character spans) | Strict Macro-F1 | **0.5137** | HarveyLM · LORMA + CRF + boundary snapping + context ensemble |
| 3 | Rhetorical role prediction | Macro-F1 | **0.4785** | HarveyLM ensemble (5-fold LORMA + unanimous override) |
| 4 | Court judgment prediction | Macro-F1 | **0.7712** | HarveyLM · LORMA, head–tail 2,048 tokens |
| 5 | Legal statute identification (100 labels) | Macro-F1 | **0.3922** | Multi-architecture probability ensemble |
| 6 | Abstractive summarisation | BERTScore F1 | **0.8407** | BaselineLM · LoRA r=64, 3,072-token input |
| 7 | Prior case retrieval | Micro-F1@7 | **0.3066** | HarveyLM bi-encoder ensemble |

All scores come from the per-phase `results.txt` files; each phase README lists every submission.

---

## Repository layout

```
code/
├── README.md               ← you are here
├── requirements.txt
├── evaluation/             official local scorers: evaluate_phase{1..7}.py
├── phase1/                 tokenizer + four from-scratch LM architectures + pre-training
├── phase2/  … phase7/      one folder per downstream task
│   ├── README.md           task, method, how to run, every submission with its score
│   ├── src/                library code (data loading, models/heads, adapters, metrics)
│   ├── scripts/            entry points (train / predict / ensemble / package)
│   ├── submissions/        <backbone>/<method>/{prediction.json, submission.zip}
│   ├── results.txt         raw experiment log (dev + leaderboard scores)
│   ├── data/               (git-ignored) official data, downloaded by scripts/download_data.py
│   └── models/             (git-ignored) checkpoints and training logs
```

Submission folders use the same backbone names in every phase:

| Folder | Phase-1 model | Parameters |
|---|---|:-:|
| `baseline/` | **BaselineLM** — LLaMA-style decoder (RoPE, GQA, SwiGLU, RMSNorm) | 242.8 M |
| `harveylm/` | **HarveyLM** — decoder with ALiBi, QK-norm, parallel attention/FFN, tied embeddings | 210.0 M |
| `moe/` | **HarveyMoE** — sparse mixture-of-experts decoder (4 experts, top-2 routing) | 270.0 M |
| `enc_dec/` | **LegalEncoderDecoder** — 6+6-layer encoder–decoder, tied embeddings | 125.3 M |
| `ensemble/` | combinations of the above | — |

Parameter counts are read from the saved checkpoints.

### Adaptation methods (names used in folders and scripts)

| Name | What it is |
|---|---|
| `lora` | LoRA (Hu et al., 2022), implemented from scratch: frozen W, trainable zero-initialised `B·A` |
| `lorma` | **LORMA (Low-Rank Matrix Adaptation)** (Bihany et al., 2025; course reference [12]): `W = W_res + s·B·A`, with `B, A` initialized from the top-r singular components of W and `W_res` frozen. Fulfills the compulsory LORMA fine-tuning requirement across all downstream phases. |
| `qlora` | LoRA on a 4-bit NormalFloat-quantised frozen base (Dettmers et al., 2023), NF4 quantiser written from scratch |
| `kd` | Knowledge distillation (Hinton et al., 2015): student trained on teacher soft targets + labels |
| `full_ft` | full fine-tuning (Phase 3 only) |

---

## Setup

```bash
# 1. clone so that the repository folder is called `code` (scripts use paths relative to its parent)
git clone <repo-url> code && cd code

# 2. environment (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. data — official CS-781 dataset from Hugging Face (needs an HF token with access)
echo "HF_TOKEN=hf_..." > ../.env
python phase1/src/data/download.py          # Phase 1 corpus
python phase3/scripts/download_data.py      # same pattern for phases 3–7
```

All experiments, pre-training, and fine-tuning scripts run on a **single local GPU workstation**
powered by an NVIDIA GeForce RTX 5080 GPU (16 GB GDDR7 VRAM, CUDA 13.0). To pick a different device, set
`CUDA_VISIBLE_DEVICES`. To use a specific interpreter, set `PYTHON_BIN`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=.venv/bin/python bash phase4/scripts/run_all_cjpe_matrix.sh
```

Phases 2–7 load the Phase-1 checkpoints from `phase1/models/final_architectures/`
(`1_Baseline`, `2_HarveyLM`, `3_HarveyMoE`, `4_EncoderDecoder`). Train them first (see
[phase1/README.md](phase1/README.md)), or place the released checkpoints there.

## Local evaluation

Every phase uses the official scorer on a held-out split before anything is submitted:

```bash
python evaluation/evaluate_phase4.py --truth path/to/dev_gold.json --prediction phase4/submissions/harveylm/lorma_headtail/prediction.json
```

## Submission format

Each `submission.zip` contains only `prediction.json` at the archive root, which is the Codabench format.

## Phase documentation

[Phase 1](phase1/README.md) · [Phase 2](phase2/README.md) · [Phase 3](phase3/README.md) ·
[Phase 4](phase4/README.md) · [Phase 5](phase5/README.md) · [Phase 6](phase6/README.md) ·
[Phase 7](phase7/README.md)

The full write-up (methods, experiments, analysis, references) is in the project report.
