#!/bin/bash
# ==============================================================================
# Phase 7 Submission 5: LegalEncoderDecoder (125M Seq2Seq) + Knowledge Distillation (KD)
# Teacher: HarveyLM LORMA
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

TEACHER_CKPT="${CODE_DIR}/phase7/models/harveylm/lorma/best_model.pt"
SUB_DIR="${CODE_DIR}/phase7/submissions/enc_dec/kd"
mkdir -p "${SUB_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 7 SUBMISSION 5: LEGAL ENCODER-DECODER + KNOWLEDGE DISTILLATION"
echo "Timestamp: $(date)"
echo "Teacher Checkpoint: ${TEACHER_CKPT}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase7/scripts/run_pcr_pipeline.py" \
    --arch enc_dec \
    --adaptation kd \
    --lr 2e-4 \
    --epochs 2 \
    --batch_size 16 \
    --eval_batch_size 32 \
    --max_seq_len 768 \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --device cuda:0 \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 7 SUBMISSION 5 COMPLETED AT: $(date)"
echo "=========================================================================="
