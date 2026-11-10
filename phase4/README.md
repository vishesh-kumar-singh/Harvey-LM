# Phase 4 — Court Judgment Prediction (CJPE)

Predict the binary outcome (0 = dismissed, 1 = allowed) of a case from its judgment text.
Metric: Macro-F1 over the two classes.

| Split | Cases | Unique IDs | Label 0 | Label 1 |
|---|:-:|:-:|:-:|:-:|
| Train | 37,387 | 32,305 | 22,067 (59.0%) | 15,320 (41.0%) |
| Dev | 3,505 | 2,511 | 1,749 (49.9%) | 1,756 (50.1%) |
| Test | 1,517 | 1,517 | hidden | hidden |

> **Data leakage in the official split.** Every one of the 1,517 test cases also appears, with the
> same ID and text, in `Train/dev.json`, together with its label (checked by exact text hash: 1,517 /
> 1,517 matches, no conflicting labels). Two consequences follow:
> 1. Dev is not an independent validation set. The decision thresholds and checkpoints below were
>    selected on dev, so the leaderboard numbers are optimistic and closely match the dev numbers.
> 2. A prediction file built by copying the dev labels scores 1.000 on the leaderboard. Such a file
>    exists in the project history. It is **not a model result**, it is excluded from this folder and
>    from all tables, and it should be read only as evidence of the leak.

## Method

* **Input representation** (`src/data/cjpe_dataset.py`). *Head–tail* truncation to 2,048 tokens,
  taking the first and last halves of the token budget. During training, known dispositive phrases
  in the final part of the text are scrubbed, and the tail is randomly shortened (`tail_drop`), so the
  model cannot read the outcome straight from the closing order.
* **Hierarchical alternative** (`generate_hierarchical_submission.py`). 8 × 1,024-token chunks, chunk
  pooling, then a small cross-chunk transformer.
* **Classifier** (`src/model/cjpe_classifier.py`). Pooled backbone representation, an MLP head, and a
  logit-adjusted loss (Menon et al., 2021; `--tau_la`) to correct the 59/41 train prior.
* **Threshold calibration.** A logit-offset δ is chosen on dev to maximise Macro-F1
  (`src/evaluation/eval_utils.py`).
* **Adapters.** LoRA, LORMA (SVD-initialised) and QLoRA (NF4), with r = 32, α = 64.
* **KD.** The HarveyLM LORMA model's soft targets on the training cases
  (`scripts/prepare_cjpe_teacher_targets.py`) are distilled into a HarveyLM student, and separately
  into the smaller LegalEncoderDecoder (125 M).

## How to run

```bash
bash phase4/scripts/run_all_cjpe_matrix.sh    # LORMA/LoRA for baseline, moe, enc_dec
bash phase4/scripts/run_harveylm_lora.sh      # HarveyLM LoRA
python code/phase4/scripts/train_cjpe_matrix.py --arch harveylm --adaptation lorma \
       --output_dir code/phase4/models/harveylm/lorma_headtail --sub_dir code/phase4/submissions/harveylm/lorma_headtail
python code/phase4/scripts/prepare_cjpe_teacher_targets.py   # teacher soft targets for KD
bash phase4/scripts/run_harveylm_kd.sh        # HarveyLM KD student + grand ensemble
bash phase4/scripts/run_kd_and_ensemble.sh    # EncDec KD student + ensemble
```

`train_cjpe_matrix.py` options: `--arch {harveylm,baseline,moe,enc_dec}`,
`--adaptation {lorma,lora,qlora,full,kd}`, `--r`, `--alpha`, `--max_seq_len`, `--batch_size`,
`--grad_accum`, `--epochs`, `--lr`, `--tau_la`, `--teacher_logits`.

## Submissions and scores

| Folder | Backbone · method | Dev Macro-F1 | LB Macro-F1 |
|---|---|:-:|:-:|
| `harveylm/lorma_headtail` | HarveyLM · LORMA, head–tail | 0.7696 | **0.7712** |
| `harveylm/lora_headtail` | HarveyLM · LoRA | 0.7570 | 0.7554 |
| `harveylm/qlora_headtail` | HarveyLM · QLoRA | 0.7513 | 0.7531 |
| `harveylm/kd_headtail` | HarveyLM · KD student | 0.7693 | 0.7657 |
| `harveylm/lorma_hierarchical` | HarveyLM · LORMA, 8×1024 hierarchical | 0.5102 | 0.5006 |
| `baseline/lora_headtail` | BaselineLM · LoRA | 0.7718 | 0.7706 |
| `baseline/lorma_headtail` | BaselineLM · LORMA | 0.7220 | 0.7178 |
| `enc_dec/lorma_headtail` | EncDec · LORMA | 0.5993 | 0.6024 |
| `enc_dec/kd_headtail` | EncDec · KD student (125 M) | 0.6105 | 0.6084 |
| `moe/lorma_headtail` | HarveyMoE · LORMA | 0.3340 | 0.3344 |
| `ensemble/grand_consensus` | probability average across backbones | 0.7667 | 0.7683 |

`results.txt` also records two later re-runs of the ensemble (LB 0.7662 and 0.7703). The folder holds
the most recent file.

The HarveyMoE run collapsed to predicting a single class (Macro-F1 ≈ 0.33 is the single-class value).
