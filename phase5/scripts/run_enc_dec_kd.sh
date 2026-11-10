#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON_BIN:-/home/vishesh/others/.venv/bin/python3} -u"
SCRIPT="code/phase5/scripts/train_lsi_matrix.py"
TEACHER_LOGITS="code/phase5/data/cache/teacher_soft_targets.pt"

echo "=========================================================================="
echo "Phase 5 LSI: LegalEncoderDecoder Knowledge Distillation (Student, 125M params)"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase5/models/logs
mkdir -p code/phase5/models/kd/kd_enc_dec
mkdir -p code/phase5/submissions/enc_dec/kd_headtail

$PYTHON $SCRIPT \
    --arch enc_dec \
    --adaptation kd \
    --teacher_logits $TEACHER_LOGITS \
    --r 32 \
    --alpha 64 \
    --epochs 1 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 2.5e-4 \
    --output_dir code/phase5/models/kd/kd_enc_dec \
    --sub_dir code/phase5/submissions/enc_dec/kd_headtail

echo ""
echo "=========================================================================="
echo "KNOWLEDGE DISTILLATION STUDENT COMPLETED SUCCESSFULLY AT: $(date)"
echo "=========================================================================="
