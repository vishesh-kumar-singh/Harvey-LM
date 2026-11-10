#!/bin/bash
# ==============================================================================
# Phase 6 Experiment 3: HarveyLM + QLoRA (4-bit NF4) Adaptation
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

OUTPUT_DIR="${CODE_DIR}/phase6/models/harveylm/qlora"
SUB_DIR="${CODE_DIR}/phase6/submissions/harveylm/qlora"
LOG_DIR="${CODE_DIR}/phase6/models/logs"

mkdir -p "${OUTPUT_DIR}" "${SUB_DIR}" "${LOG_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 6 EXPERIMENT 3: HARVEYLM + QLORA (4-bit NF4)"
echo "Timestamp: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase6/scripts/train_summarizer.py" \
    --arch harveylm \
    --adaptation qlora \
    --r 32 \
    --alpha 64 \
    --lr 2.0e-4 \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 2 \
    --max_doc_len 2048 \
    --head_tokens 1024 \
    --tail_tokens 1024 \
    --max_summary_len 768 \
    --gen_tokens 500 \
    --eval_samples 100 \
    --output_dir "${OUTPUT_DIR}" \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 6 EXPERIMENT 3 (HARVEYLM QLORA) COMPLETED SUCCESSFULLY AT: $(date)"
echo "=========================================================================="
