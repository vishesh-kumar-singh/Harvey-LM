# Phase 5 — Legal Statute Identification (LSI)

Multi-label classification: assign every applicable statute (100 classes, IDs 0–99) to a case.
Metric: Macro-F1 over the 100 classes; secondary metrics are Micro-F1 and subset accuracy.

| Split | Cases | Label assignments | Mean labels / case |
|---|:-:|:-:|:-:|
| Train | 42,750 | 158,398 | 3.71 |
| Dev | 10,181 | 39,870 | 3.92 |
| Test | 13,019 | hidden | — |

The label distribution is long-tailed: the most frequent statute (#5) appears in 27.7% of training
cases. 188 test texts (1.4%) also occur verbatim in train/dev, which is a minor overlap
(`scripts/explore_phase5_data.py` produces the statistics).

## Method

* **Input.** Head–tail 2,048 tokens. `scripts/pretokenize_lsi_data.py` caches the tokenised inputs.
* **Head** (`src/model/lsi_classifier.py`). Multi-query attention pooling (4 learned queries) over
  the backbone states, followed by an MLP that outputs 100 sigmoid logits.
* **Loss.** Asymmetric loss (Ridnik et al., 2021) for the positive/negative imbalance.
* **Per-class thresholds**, tuned on dev to maximise each class's F1.
* **Adapters.** LoRA, LORMA (SVD-initialised) and QLoRA (NF4), with r = 32, α = 64. For HarveyMoE,
  the router is trainable.

## How to run

```bash
python code/phase5/scripts/download_data.py && python code/phase5/scripts/pretokenize_lsi_data.py
bash phase5/scripts/run_harveylm_lorma.sh
bash phase5/scripts/run_baseline_lora.sh
bash phase5/scripts/run_enc_dec_lorma.sh
bash phase5/scripts/run_moe_lora.sh
bash phase5/scripts/run_harveylm_qlora.sh
python code/phase5/scripts/prepare_lsi_teacher_targets.py && bash phase5/scripts/run_enc_dec_kd.sh
python code/phase5/scripts/generate_grand_ensemble.py
# re-create a submission from a saved checkpoint:
python code/phase5/scripts/generate_submission_from_ckpt.py --help
```

## Submissions and scores

| Folder | Backbone · method | Dev Macro-F1 | Dev Micro-F1 | LB Macro-F1 |
|---|---|:-:|:-:|:-:|
| `baseline/lora_headtail` | BaselineLM · LoRA | 0.3483 | 0.4287 | 0.3859 |
| `harveylm/lorma_headtail` | HarveyLM · LORMA | 0.3539 | 0.4394 | 0.3851 |
| `harveylm/qlora_headtail` | HarveyLM · QLoRA | 0.3988 | 0.4922 | submitted, not recorded |
| `moe/lora_headtail` | HarveyMoE · LoRA | 0.3201 | 0.4038 | 0.3058 |
| `enc_dec/lorma_headtail` | EncDec · LORMA | 0.2827 | 0.3826 | 0.2800 |
| `ensemble/grand_consensus` | sigmoid average across backbones | 0.4108 | 0.5070 | **0.3922** |

**Knowledge distillation was not completed in this phase.** The EncDec KD run was queued but no
prediction file was produced (`results.txt`: "QUEUED").

For HarveyMoE, `results.txt` holds two dev entries (0.3015 in the summary table, 0.3201 in the later
log). The table above uses the later one.
