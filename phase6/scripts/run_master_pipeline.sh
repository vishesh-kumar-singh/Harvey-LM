#!/bin/bash
# ==============================================================================
# Phase 6: end-to-end pipeline for the scaled summarisation runs + CLE v2 ensemble
#   1. Train the three scaled models (r=64, alpha=128, ctx=3072) one after another:
#        BaselineLM LoRA (scaled) -> HarveyLM LoRA (scaled) -> HarveyLM QLoRA (scaled)
#   2. Causal Logit-space Ensemble (CLE v2) over the three checkpoints
#        - 3,072-token head/middle/tail input budget
#        - holding-prompt injection at generated token 620, 768-token budget
#        - evaluates on the dev split and packages submission.zip
# Runs on a single GPU (first visible device; set CUDA_VISIBLE_DEVICES to choose).
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"
LOG_DIR="${CODE_DIR}/phase6/models/logs"
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "PHASE 6 scaled pipeline started: $(date) | logs: ${LOG_DIR}"

# 1. Scaled training runs (sequential)
for run in baseline_lora_scaled harveylm_lora_scaled harveylm_qlora_scaled; do
    echo "[1/2] Training ${run} ..."
    bash "${SCRIPT_DIR}/run_${run}.sh" > "${LOG_DIR}/run_${run}.log" 2>&1
    echo "      ${run} finished at $(date)"
done

# 2. Causal Logit Ensemble (CLE v2)
CKPT_BASE="${CODE_DIR}/phase6/models/baseline/lora_scaled/best_model.pt"
CKPT_QLORA="${CODE_DIR}/phase6/models/harveylm/qlora_scaled/best_model.pt"
CKPT_LORMA="${CODE_DIR}/phase6/models/harveylm/lora_scaled/best_model.pt"
SUB_DIR="${CODE_DIR}/phase6/submissions/ensemble/causal_logit_ensemble_v2"

# Fall back to the unscaled checkpoints if a scaled one is missing
[ -f "${CKPT_BASE}" ]  || CKPT_BASE="${CODE_DIR}/phase6/models/baseline/lora/best_model.pt"
[ -f "${CKPT_QLORA}" ] || CKPT_QLORA="${CODE_DIR}/phase6/models/harveylm/qlora/best_model.pt"
[ -f "${CKPT_LORMA}" ] || CKPT_LORMA="${CODE_DIR}/phase6/models/harveylm/lorma/best_model.pt"

echo "[2/2] CLE v2 ensemble"
echo "      BaselineLM:     ${CKPT_BASE}"
echo "      HarveyLM QLoRA: ${CKPT_QLORA}"
echo "      HarveyLM LoRA:  ${CKPT_LORMA}"

${VENV_PYTHON} -u "${SCRIPT_DIR}/generate_causal_ensemble.py" \
    --device "cuda:0" \
    --max_doc_len 3072 \
    --max_new_tokens 768 \
    --eval_dev_samples 50 \
    --w_base 0.45 \
    --w_qlora 0.35 \
    --w_lorma 0.20 \
    --ckpt_base "${CKPT_BASE}" \
    --ckpt_qlora "${CKPT_QLORA}" \
    --ckpt_lorma "${CKPT_LORMA}" \
    --sub_dir "${SUB_DIR}" \
    > "${LOG_DIR}/master_causal_ensemble_v2.log" 2>&1

echo "Done at $(date). Submission: ${SUB_DIR}/submission.zip"
