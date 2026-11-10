#!/bin/bash
# ==============================================================================
# Phase 6 Experiment 1: LegalEncoderDecoder + LORMA Adaptation (Compulsory)
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

OUTPUT_DIR="${CODE_DIR}/phase6/models/enc_dec/lorma"
SUB_DIR="${CODE_DIR}/phase6/submissions/enc_dec/lorma"
LOG_DIR="${CODE_DIR}/phase6/models/logs"

mkdir -p "${OUTPUT_DIR}" "${SUB_DIR}" "${LOG_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 6 EXPERIMENT 1: LEGAL ENCODER-DECODER + LORMA"
echo "Timestamp: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase6/scripts/train_summarizer.py" \
    --arch enc_dec \
    --adaptation lorma \
    --r 32 \
    --alpha 64 \
    --lr 1.5e-4 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_doc_len 2048 \
    --head_tokens 1024 \
    --tail_tokens 1024 \
    --max_summary_len 768 \
    --gen_tokens 500 \
    --eval_samples 100 \
    --output_dir "${OUTPUT_DIR}" \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 6 EXPERIMENT 1 (ENC_DEC LORMA) COMPLETED SUCCESSFULLY AT: $(date)"
echo "=========================================================================="
