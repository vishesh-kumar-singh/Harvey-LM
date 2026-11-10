# Phase 7 — Prior Case Retrieval (PCR)

For each query case, rank its candidate pool and return the 7 most relevant prior cases.
Metric: Micro-F1@7; secondary metrics are Micro-Precision@7 and Micro-Recall@7.

| Split | Queries | Candidate pool |
|---|:-:|:-:|
| Train | 827 | 4,320 |
| Dev | 118 | 1,023 |
| Test | 237 | 1,727 |

`scripts/explore_phase7_data.py` reports relevance-set sizes and lexical overlap statistics.
`scripts/prepare_eval_files.py` builds the local gold files for `evaluation/evaluate_phase7.py`.

## Method

* **Lexical baseline.** BM25 (`src/baselines/`).
* **Dense bi-encoder** (`src/model/pcr_retriever.py`, `scripts/run_pcr_pipeline.py`). A Phase-1
  backbone encodes the query and the candidate into pooled vectors. It is trained with an in-batch
  contrastive (InfoNCE) loss at temperature 0.05, and candidates are ranked by cosine similarity.
  Adapters: LoRA, LORMA (SVD-initialised) and QLoRA (NF4), r = 32, α = 64.
* **KD.** HarveyLM LORMA (210 M) is the teacher and LegalEncoderDecoder (125 M) the student. The
  student matches the teacher's similarity distribution (`--kd_temperature 2.0`, `--kd_alpha 0.5`).
* **Hybrid.** Reciprocal-rank fusion of BM25 with a dense run (`--w_dense`).
* **Improved retriever** (`scripts/run_peak_solution.py`). Head/middle/tail input selection, mined
  hard negatives (`scripts/mine_hard_negatives.py`), a multi-positive InfoNCE loss with a margin
  term, CSLS anti-hubness re-scoring, and score fusion across the HarveyLM LORMA, HarveyLM QLoRA and
  BaselineLM LoRA retrievers.
* **Single-model variants** (`scripts/run_harvey_mglg_standalone.py`,
  `scripts/run_standalone_cross_encoder.py`). A multi-granularity late-interaction model and a
  cross-encoder re-ranker, both on HarveyLM and both initialised from the improved retriever.

## How to run

```bash
python code/phase7/scripts/download_data.py && python code/phase7/scripts/prepare_eval_files.py
python code/phase7/scripts/run_pcr_pipeline.py --arch bm25 --adaptation bm25 --sub_dir code/phase7/submissions/bm25/bm25_lexical
bash phase7/scripts/run_baseline_lora.sh
bash phase7/scripts/run_harveylm_lorma.sh
bash phase7/scripts/run_harveylm_qlora.sh
bash phase7/scripts/run_moe_lora.sh
bash phase7/scripts/run_enc_dec_kd.sh            # needs the HarveyLM LORMA checkpoint (teacher)
bash phase7/scripts/run_hybrid_rrf.sh            # BM25 + dense fusion
bash phase7/scripts/launch_peak_solution.sh      # improved retriever + fusion
bash phase7/scripts/launch_harvey_mglg.sh        # single-model late-interaction variant
bash phase7/scripts/launch_standalone_cross_encoder.sh
```

Checkpoints are written to `models/<backbone>/<method>/best_model.pt`, and predictions and
`submission.zip` to `submissions/<backbone>/<method>/`. Each submission folder also contains
`dev_prediction.json`.

## Submissions and scores

| Folder | System | Dev F1@7 | Dev P@7 | Dev R@7 | LB F1@7 |
|---|---|:-:|:-:|:-:|:-:|
| `bm25/bm25_lexical` | BM25 | 0.1078 | 0.1174 | 0.0997 | 0.1282 |
| `moe/lora` | HarveyMoE · LoRA | 0.1890 | 0.2058 | 0.1747 | 0.1595 |
| `baseline/lora` | BaselineLM · LoRA | 0.2379 | 0.2591 | 0.2199 | 0.2215 |
| `harveylm/qlora` | HarveyLM · QLoRA | 0.2735 | 0.2978 | 0.2528 | 0.2516 |
| `harveylm/lorma` | HarveyLM · LORMA | 0.2802 | 0.3051 | 0.2590 | 0.2540 |
| `enc_dec/kd` | EncDec student ← HarveyLM LORMA | 0.1190 | 0.1295 | 0.1100 | 0.0992 |
| `ensemble/hybrid_rrf` | BM25 + BaselineLM LoRA (RRF) | 0.2379 | 0.2591 | 0.2199 | not recorded |
| `ensemble/peak_solution` | improved retriever, 3-model fusion | 0.3324 | 0.3620 | 0.3073 | **0.3066** |
| `harveylm/mglg_standalone` | late-interaction, single model | 0.3558 | 0.3874 | 0.3289 | not recorded |
| `harveylm/cross_encoder` | cross-encoder re-ranker | not recorded | | | not recorded |

The hybrid run's dev scores are identical to the dense BaselineLM run, so the fusion did not change
the dev ranking.
