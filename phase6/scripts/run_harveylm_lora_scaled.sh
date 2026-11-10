#!/bin/bash
# ==============================================================================
# Phase 6 Scaled Run 2: HarveyLM (219M ALiBi) + High-Rank LoRA (r=64, alpha=128)
# Context: 3,072 Tokens (Triple-Anchor Salience Budgeting)
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

OUTPUT_DIR="${CODE_DIR}/phase6/models/harveylm/lora_scaled"
SUB_DIR="${CODE_DIR}/phase6/submissions/harveylm/lora_scaled"
LOG_DIR="${CODE_DIR}/phase6/models/logs"

mkdir -p "${OUTPUT_DIR}" "${SUB_DIR}" "${LOG_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 6 SCALED RUN 2: HARVEYLM + LORA (r=64, alpha=128, ctx=3072)"
echo "Timestamp: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase6/scripts/train_summarizer.py" \
    --arch harveylm \
    --adaptation lora \
    --r 64 \
    --alpha 128 \
    --lr 1.5e-4 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --max_doc_len 3072 \
    --max_summary_len 768 \
    --gen_tokens 768 \
    --use_triple_anchor \
    --eval_samples 20 \
    --val_tokens 128 \
    --final_dev_samples 50 \
    --output_dir "${OUTPUT_DIR}" \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 6 SCALED RUN 2 (HARVEYLM LORA) COMPLETED AT: $(date)"
echo "=========================================================================="
