#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase4/scripts/train_cjpe_matrix.py"

echo "=========================================================================="
echo "Phase 4 CJPE: HarveyLM Knowledge Distillation (Student Head-Tail)"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase4/models/logs
mkdir -p code/phase4/models/harveylm/kd_headtail
mkdir -p code/phase4/submissions/harveylm/kd_headtail

# 1. Train HarveyLM KD Student
echo ">>> Training HarveyLM Knowledge Distillation Student (r=32, alpha=64)..."
$PYTHON $SCRIPT \
    --arch harveylm \
    --adaptation kd \
    --teacher_logits code/phase4/data/cache/teacher_soft_targets.pt \
    --r 32 \
    --alpha 64 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.5e-4 \
    --output_dir code/phase4/models/harveylm/kd_headtail \
    --sub_dir code/phase4/submissions/harveylm/kd_headtail

# 2. Re-generate Grand Multi-Architecture Consensus Ensemble with HarveyLM KD
echo ""
echo ">>> Re-generating Grand Multi-Architecture Consensus Ensemble..."
$PYTHON code/phase4/scripts/generate_grand_ensemble.py

echo ""
echo "=========================================================================="
echo "HARVEYLM KNOWLEDGE DISTILLATION & ENSEMBLE COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
