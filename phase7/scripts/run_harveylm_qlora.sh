#!/bin/bash
# ==============================================================================
# Phase 7 Submission 2: HarveyLM (210M ALiBi) + QLoRA (4-bit NF4 from scratch)
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

SUB_DIR="${CODE_DIR}/phase7/submissions/harveylm/qlora"
mkdir -p "${SUB_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 7 SUBMISSION 2: HARVEYLM + QLORA"
echo "Timestamp: $(date)"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase7/scripts/run_pcr_pipeline.py" \
    --arch harveylm \
    --adaptation qlora \
    --r 32 \
    --alpha 64 \
    --lr 1e-4 \
    --epochs 2 \
    --batch_size 16 \
    --eval_batch_size 32 \
    --max_seq_len 768 \
    --device cuda:0 \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 7 SUBMISSION 2 COMPLETED AT: $(date)"
echo "=========================================================================="
