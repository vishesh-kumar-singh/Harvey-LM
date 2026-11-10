#!/bin/bash
# ==============================================================================
# Phase 6 Experiment 6: LegalEncoderDecoder Knowledge Distillation (Student, 125M)
# Distilled from peak HarveyLM LORMA teacher
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

TEACHER_CKPT="${CODE_DIR}/phase6/models/enc_dec/lorma/best_model.pt"
OUTPUT_DIR="${CODE_DIR}/phase6/models/kd/kd_enc_dec"
SUB_DIR="${CODE_DIR}/phase6/submissions/enc_dec/kd"
LOG_DIR="${CODE_DIR}/phase6/models/logs"

mkdir -p "${OUTPUT_DIR}" "${SUB_DIR}" "${LOG_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 6 EXPERIMENT 6: KNOWLEDGE DISTILLATION (ENC_DEC STUDENT)"
echo "Timestamp: $(date)"
echo "Teacher Checkpoint: ${TEACHER_CKPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase6/scripts/train_summarizer.py" \
    --arch enc_dec \
    --adaptation kd \
    --teacher_arch enc_dec \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --lr 2.0e-4 \
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
echo "PHASE 6 EXPERIMENT 6 (ENC_DEC KD) COMPLETED SUCCESSFULLY AT: $(date)"
echo "=========================================================================="
