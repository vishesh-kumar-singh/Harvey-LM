#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase4/scripts/train_cjpe_matrix.py"

echo "=========================================================================="
echo "Phase 4 CJPE: HarveyLM LoRA (Standard Low-Rank Adaptation, 2,048 tokens)"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase4/models/logs
mkdir -p code/phase4/models/harveylm/lora_headtail
mkdir -p code/phase4/submissions/harveylm/lora_headtail

# 1. Train HarveyLM LoRA
echo ">>> Training HarveyLM LoRA (Standard Low-Rank Adaptation, r=32, alpha=64)..."
$PYTHON $SCRIPT \
    --arch harveylm \
    --adaptation lora \
    --r 32 \
    --alpha 64 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.5e-4 \
    --output_dir code/phase4/models/harveylm/lora_headtail \
    --sub_dir code/phase4/submissions/harveylm/lora_headtail

# 2. Re-generate Grand Multi-Architecture Consensus Ensemble with HarveyLM LoRA
echo ""
echo ">>> Re-generating Grand Multi-Architecture Consensus Ensemble (with HarveyLM LoRA)..."
$PYTHON code/phase4/scripts/generate_grand_ensemble.py

echo ""
echo "=========================================================================="
echo "HARVEYLM LORA AND ENSEMBLE RE-GENERATION COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
