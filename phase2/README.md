# Phase 2 — Legal Named Entity Recognition (L-NER)

Predict character-level spans `[start, end, label]` for 12 legal entity types (APP, RESP, A.COUNSEL,
R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO). Metric: **strict** Macro-F1, where a
span counts only if start, end and label all match exactly.

## Method

* **Token classification.** BIO tagging (25 tags) over the 32k BPE tokens. Character offsets are recovered
  from the tokenizer's offset mapping. Documents are split into non-overlapping chunks: 2048 tokens
  for the LoRA/LORMA softmax models, and 512 tokens for the CRF models and the KD student.
* **Class imbalance.** Inverse-frequency class weights, with the `O` weight fixed at 0.5.
* **Heads.** A linear softmax head, or a linear-chain CRF with Viterbi decoding (`pytorch-crf`).
* **Boundary snapping** (`scripts/snap_boundaries.py`, `snap_crf_boundaries.py`). A post-processing
  step that expands spans to whole alphanumeric words and trims stray edge punctuation. It fixes
  off-by-one character boundaries caused by BPE.
* **Context ensemble** (`scripts/generate_all_architecture_submissions.py`). Runs overlapping
  sliding-window inference (stride 256) and merges it with the chunked CRF predictions. When spans
  overlap, the longer, complete span is kept, which repairs entities cut at chunk boundaries.
* **Dense merge.** Folds the trained low-rank factors back into the base weights, giving one dense
  model with the same outputs (`9_dense_exact_merged_snapped`).

Code: `src/model/lora_from_scratch.py`, `src/model/lorma_from_scratch.py`, `src/model/` (heads),
`src/data/` (span ↔ token alignment).

## How to run

All scripts take their settings from constants at the top of the file. Run them from the folder
that contains `code/`.

```bash
# HarveyLM backbone
python code/phase2/scripts/train_harvey_lora_weighted_full.py      # -> 1_lora
python code/phase2/scripts/train_harvey_lorma_weighted_full.py     # -> 2_lorma
python code/phase2/scripts/generate_lorma_predictions.py
python code/phase2/scripts/snap_boundaries.py                      # -> 3_lorma_boundary_snapping
python code/phase2/scripts/train_harvey_lorma_crf_full.py          # -> 4_lorma_crf
python code/phase2/scripts/generate_lorma_crf_predictions.py
python code/phase2/scripts/snap_crf_boundaries.py                  # -> 5_lorma_crf_boundary_snapping
python code/phase2/scripts/train_harvey_distillation_full.py       # -> 6_kd_student (8-layer, 121 M student)
# other backbones (CRF head) + long-context fusion for all four
python code/phase2/scripts/train_baseline_ner.py
python code/phase2/scripts/train_moe_ner.py
python code/phase2/scripts/train_enc_dec_ner.py
python code/phase2/scripts/generate_all_architecture_submissions.py
```

## Submissions and leaderboard scores (strict Macro-F1)

| Method | `harveylm/` | `baseline/` | `moe/` | `enc_dec/` |
|---|:-:|:-:|:-:|:-:|
| 1 LoRA | 0.4286 | 0.2767 | 0.0280 | 0.0555 |
| 2 LORMA | 0.4379 | † | † | † |
| 3 LORMA + snapping | 0.4687 | 0.3215 | 0.0479 | 0.0559 |
| 4 LORMA + CRF | 0.4983 | — | — | — |
| 5 LORMA + CRF + snapping | 0.5036 | 0.3857 | 0.0729 | 0.0950 |
| 6 KD student (8 layers, 121 M) | 0.3096 | † | — | † |
| 7 KD student + snapping | 0.3485 | ‡ | ‡ | ‡ |
| 8 Context ensemble (CRF + snapping) | **0.5137** | 0.3851 | 0.0742 | 0.0934 |
| 9 Dense merged (= 5) | 0.5036 | | | |

**Important: duplicate prediction files.** For the BaselineLM, HarveyMoE and LegalEncoderDecoder
folders, several "different" submissions are byte-identical:

* † `2_lorma` is identical to `1_lora` (for EncDec, `6_kd_student` is also identical to `1_lora`).
* ‡ `7_kd_student_boundary_snapping` is identical to `3_lorma_boundary_snapping`.

Separate LORMA and KD models were therefore **not** produced for those three backbones, and the
repeated leaderboard scores in `results.txt` are re-submissions of the same file. Only the HarveyLM
column contains eight distinct experiments. "—" means not run.
