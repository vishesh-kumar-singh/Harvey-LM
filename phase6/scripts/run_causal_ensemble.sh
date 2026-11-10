#!/bin/bash
# ==============================================================================
# Phase 6: Causal Logit Ensemble Runner
# Blends logits from BaselineLM LoRA, HarveyLM QLoRA, and HarveyLM LORMA
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

LOG_DIR="${CODE_DIR}/phase6/models/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 6 CAUSAL LOGIT ENSEMBLE GENERATION"
echo "Timestamp: $(date)"
echo "Log: ${LOG_DIR}/run_causal_ensemble.log"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase6/scripts/generate_causal_ensemble.py" \
    --device "cuda:0" \
    --max_doc_len 2048 \
    --max_new_tokens 768 \
    --eval_dev_samples 50 \
    --w_base 0.45 \
    --w_qlora 0.35 \
    --w_lorma 0.20

echo "=========================================================================="
echo "PHASE 6 CAUSAL LOGIT ENSEMBLE COMPLETED AT: $(date)"
echo "=========================================================================="
