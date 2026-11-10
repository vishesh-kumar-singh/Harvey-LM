# Phase 6 — Abstractive Legal Summarisation

Generate an abstractive summary of a judgment. Metric: BERTScore F1; secondary metrics are ROUGE-1 F1
and ROUGE-L F1.

| Split | Cases |
|---|:-:|
| Train (after removing 2 empty summaries) | 6,326 |
| Dev (held-out 10%, `scripts/create_splits.py`) | 702 |
| Test | 100 |

Document length: median 3,128 words (max 139,645). Summary length: median 638 words.
Median compression is 4.6× (`scripts/explore_phase6_data.py`).

## Method

* **Conditional generation.** Decoder-only backbones are fine-tuned on `document ⟶ summary`, with the
  loss applied to summary tokens only. The encoder–decoder is trained seq2seq.
* **Input budget.** First runs: head–tail 2,048 tokens (1,024 + 1,024). Scaled runs: a 3,072-token
  *triple-anchor* input, 40% head + 20% from the middle third + 40% tail (`--use_triple_anchor`).
* **Adapters.** LoRA, LORMA (SVD-initialised) and QLoRA (NF4). r = 32, α = 64 at first, then
  r = 64, α = 128 for the scaled runs.
* **Decoding.** 500 generated tokens at first, then 768, with trigram blocking and a repetition
  penalty. *Holding-prompt injection*: if no dispositive cue has appeared by generated token 620, the
  text `"Decision and Holding:"` is appended and decoding continues
  (`scripts/generate_causal_ensemble.py`).
* **Ensembles.** (a) `grand_consensus`: picks one of the six models' output texts per document by
  maximum mean ROUGE-L agreement (minimum Bayes risk). (b) **Causal logit ensemble (CLE)**: at every
  decoding step, averages the next-token logits of decoder-only models that share the tokenizer
  (weights 0.45 / 0.35 / 0.20).
* **KD.** The student (EncDec) is trained on the labels plus the soft targets of the EncDec LORMA
  model, which is self-distillation within one architecture.

## How to run

```bash
python code/phase6/scripts/download_data.py && python code/phase6/scripts/create_splits.py
# first round (r=32, 2,048 tokens)
bash phase6/scripts/run_enc_dec_lorma.sh
bash phase6/scripts/run_harveylm_lorma.sh
bash phase6/scripts/run_harveylm_qlora.sh
bash phase6/scripts/run_moe_lora.sh
bash phase6/scripts/run_baseline_lora.sh
bash phase6/scripts/run_enc_dec_kd.sh
python code/phase6/scripts/generate_grand_ensemble.py   # MBR text ensemble
bash phase6/scripts/run_causal_ensemble.sh              # CLE v1
# scaled round (r=64, 3,072 tokens, 768-token generation) + CLE v2, sequentially
bash phase6/scripts/run_master_pipeline.sh
```

`train_summarizer.py --help` lists every option (`--arch`, `--adaptation`, `--r`, `--alpha`,
`--max_doc_len`, `--use_triple_anchor`, `--gen_tokens`, `--teacher_ckpt`, …).

## Submissions and scores

| Folder | Backbone · method | Dev BERTScore | Dev R-1 | Dev R-L | LB BERTScore |
|---|---|:-:|:-:|:-:|:-:|
| `enc_dec/lorma` | EncDec · LORMA | 0.7440 | 0.3814 | 0.2112 | 0.7441 |
| `enc_dec/kd` | EncDec · KD (self-distilled) | 0.7582 | 0.3825 | 0.2057 | 0.7441 |
| `harveylm/lorma` | HarveyLM · LORMA | 0.7966 | 0.3185 | 0.1541 | 0.8249 |
| `harveylm/qlora` | HarveyLM · QLoRA | 0.8278 | 0.4362 | 0.2027 | 0.8254 |
| `moe/lora` | HarveyMoE · LoRA | 0.8236 | 0.3953 | 0.1933 | 0.8220 |
| `baseline/lora` | BaselineLM · LoRA | 0.8324 | 0.4367 | 0.2175 | 0.8291 |
| `ensemble/grand_consensus` | MBR text selection (6 models) | 0.8214 | 0.4313 | 0.2126 | 0.7866 |
| `ensemble/causal_logit_ensemble` | CLE v1 (r=32 models) | 0.8316 | 0.4444 | 0.2075 | 0.8316 |
| `baseline/lora_scaled` | BaselineLM · LoRA r=64, 3,072 ctx | 0.8445 | 0.4403 | 0.2320 | **0.8407** |
| `harveylm/lora_scaled` | HarveyLM · LoRA r=64, 3,072 ctx | 0.8376 | 0.4284 | 0.2057 | 0.8332 |
| `harveylm/qlora_scaled` | HarveyLM · QLoRA r=64, 3,072 ctx | 0.8374 | 0.4311 | 0.2049 | 0.8344 |
| `ensemble/causal_logit_ensemble_v2` | CLE v2 (scaled models) | 0.8395 | 0.4361 | 0.2176 | 0.8397 |

Dev scores for generation are computed on a subset of the dev split (`--final_dev_samples`). An
earlier HarveyLM LORMA run scored dev BERTScore 0.7809 / R-1 0.1413; it was retrained, and the folder
holds the retrained run.
