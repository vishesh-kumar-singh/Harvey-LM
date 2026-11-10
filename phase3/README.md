# Phase 3 — Rhetorical Role Prediction (RR)

Assign one of 13 rhetorical roles to every sentence of a judgment (Corporate Law **CL** and
Information Technology **IT** domains). Metric: 13-class Macro-F1. The test set has 2,197 sentences
in 10 documents.

## Method

* **Sentence-in-context encoding** (`src/data/rr_context_dataset_v3.py`). Each target sentence is
  encoded together with its neighbours (256-token window). The target-sentence tokens are
  mean-pooled, or attention-pooled in the "attentive" variant (`src/model/attentive_token_head.py`).
* **Extra features** (`src/model/rr_universal_classifier.py`). A Fourier encoding of the sentence's
  relative position in the document, and a learned CL/IT domain embedding, both concatenated before
  the classification MLP.
* **Loss.** Class-balanced focal loss (γ = 2) with clipped inverse-frequency weights.
* **Decoding.** Viterbi over a bigram label-transition matrix estimated on the training documents
  (`src/evaluation/viterbi_decoder.py`).
* **Adapters.** LoRA, LORMA (SVD-initialised, `src/model/lorma_from_scratch.py`) and QLoRA with a
  from-scratch NF4 quantiser (`src/model/qlora_from_scratch.py`), all r = 32, α = 64 in the
  cross-architecture runs. The HarveyLM LORMA runs use r = 64, α = 128.
* **Document-level variants.** A BiLSTM + CRF over sentence vectors (`src/model/hierarchical_crf.py`),
  and 5-fold stratified cross-validation of LORMA with fold-averaged test probabilities (`v45`).
* **Knowledge distillation** (`--mode kd`). Students are trained on the labels plus soft targets that
  `scripts/prepare_teacher_soft_targets.py` computes for the **training** documents from the HarveyLM
  LORMA-v3 model (T = 1.5); KD loss uses τ = 2.5. The students are full Phase-1 backbones, so this
  run tests transfer of the teacher's label distribution across architectures; it is not a
  compression to a smaller model.

## How to run

```bash
# cross-architecture matrix (LoRA / LORMA / KD for baseline, moe, enc_dec, harveylm)
bash phase3/scripts/run_all_experiments.sh
bash phase3/scripts/run_remaining_qlora.sh                 # QLoRA for baseline, moe, enc_dec
python code/phase3/scripts/train_adaptation_and_kd.py --arch harveylm --mode qlora \
       --output_dir code/phase3/models/qlora/qlora_harveylm --sub_dir code/phase3/submissions/harveylm/qlora

# HarveyLM-specific experiments (run from the folder containing code/)
python code/phase3/scripts/train_lora_v3_rr.py             # -> harveylm/lora_v2
python code/phase3/scripts/train_lorma_v3_rr.py            # -> harveylm/lorma_v3
python code/phase3/scripts/train_full_v3_rr.py             # full fine-tuning
python code/phase3/scripts/train_attentive_lorma_rr.py     # -> harveylm/attentive_lorma
python code/phase3/scripts/train_hierarchical_crf.py       # -> harveylm/hierarchical_lorma
python code/phase3/scripts/train_lorma_v45_cv.py           # -> harveylm/lorma_v45_cv (5 folds)

# ensembles
python code/phase3/scripts/generate_elite_consensus.py     # -> ensemble/elite_consensus
python code/phase3/scripts/generate_grand_consensus_v45.py # -> ensemble/grand_consensus_v45
```

KD needs the soft-target cache, so run `scripts/prepare_teacher_soft_targets.py` before `--mode kd`.

## Submissions and scores

Dev = official dev split (CL + IT), LB = Codabench test Macro-F1.

| Folder | Backbone · method | Dev Macro-F1 | LB Macro-F1 |
|---|---|:-:|:-:|
| `baseline/lora` | BaselineLM · LoRA | 0.5074 | **0.4591** |
| `baseline/lorma` | BaselineLM · LORMA | 0.5094 | 0.4492 |
| `baseline/qlora` | BaselineLM · QLoRA | 0.5017 | not recorded |
| `baseline/kd` | BaselineLM · KD | 0.4992 | 0.2863 |
| `moe/lora` | HarveyMoE · LoRA | 0.4153 | 0.3798 |
| `moe/lorma` | HarveyMoE · LORMA | 0.3965 | 0.3829 |
| `moe/qlora` | HarveyMoE · QLoRA | 0.4329 | not recorded |
| `moe/kd` | HarveyMoE · KD | 0.3750 | 0.3238 |
| `enc_dec/lora` | EncDec · LoRA | 0.3757 | 0.3119 |
| `enc_dec/lorma` | EncDec · LORMA | 0.3775 | 0.3078 |
| `enc_dec/qlora` | EncDec · QLoRA | 0.3851 | not recorded |
| `enc_dec/kd` | EncDec · KD | 0.3002 | 0.2863 |
| `harveylm/lora_v2` | HarveyLM · LoRA | 0.3948 | 0.4436 |
| `harveylm/lorma_v3` | HarveyLM · LORMA | 0.3974 | 0.4599 |
| `harveylm/qlora` | HarveyLM · QLoRA | 0.5084 | 0.4527 |
| `harveylm/kd` | HarveyLM · KD | 0.5010 | 0.4489 |
| `harveylm/attentive_lorma` | HarveyLM · LORMA, attention pooling | 0.3980 | 0.4620 |
| `harveylm/hierarchical_lorma` | HarveyLM · LORMA + BiLSTM-CRF | 0.4002 | 0.4583 |
| `harveylm/lorma_v45_cv` | HarveyLM · LORMA, 5-fold CV average | 0.7418 (OOF)\* | 0.4660 |
| `ensemble/elite_consensus` | vote of attentive + v3 + hierarchical | 0.3983 | 0.4634 |
| `ensemble/grand_consensus_v45` | CV ensemble, overridden where all 3 elite models agree | — | **0.4785** |

Submitted, with folders not kept locally: HarveyLM full fine-tuning (LB 0.3412), fused hierarchical
CRF (0.4305), calibrated sequence ensembles v1/v2 (0.4452 / 0.4467), 4- and 5-model majority votes
(0.4524 / 0.4530), precision-routed consensus (0.4554), apex consensus (0.4741). See `results.txt`.

\* Out-of-fold score from the cross-validation folds, not the official dev split, so it is **not
comparable** to the other dev numbers.

**Dev/LB caveat.** The HarveyLM `*_v3`/`attentive`/`hierarchical` runs come from separate training
scripts and score about 0.40 on dev. The `train_adaptation_and_kd.py` runs score about 0.50 on dev,
yet on LB the ordering is reversed. Compare dev numbers only within the same script, and use LB
across scripts.
