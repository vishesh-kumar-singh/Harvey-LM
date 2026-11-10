#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"

echo "=========================================================================="
echo "Phase 4: Knowledge Distillation & Grand Multi-Architecture Ensemble"
echo "Started at: $(date)"
echo "=========================================================================="

# 1. Extract Teacher Soft Targets from HarveyLM LORMA
echo ""
echo ">>> [1/3] Extracting Teacher Soft Targets from Peak HarveyLM LORMA..."
$PYTHON code/phase4/scripts/prepare_cjpe_teacher_targets.py

# 2. Train Knowledge Distillation Student (LegalEncoderDecoder, 125M params)
echo ""
echo ">>> [2/3] Training Knowledge Distillation Student (LegalEncoderDecoder)..."
$PYTHON code/phase4/scripts/train_cjpe_matrix.py \
    --arch enc_dec \
    --adaptation kd \
    --teacher_logits code/phase4/data/cache/teacher_soft_targets.pt \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.0e-4 \
    --output_dir code/phase4/models/kd/kd_enc_dec \
    --sub_dir code/phase4/submissions/enc_dec/kd_headtail

# 3. Generate Grand Multi-Architecture Ensemble
echo ""
echo ">>> [3/3] Generating Grand Multi-Architecture Consensus Ensemble..."
$PYTHON code/phase4/scripts/generate_grand_ensemble.py

echo ""
echo "=========================================================================="
echo "KNOWLEDGE DISTILLATION & ENSEMBLE PIPELINE COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
