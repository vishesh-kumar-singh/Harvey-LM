#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase5/scripts/train_lsi_matrix.py"

echo "=========================================================================="
echo "Phase 5 LSI: HarveyLM QLoRA (4-bit NF4 Quantization + LoRA, 2,048 tokens)"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase5/models/logs
mkdir -p code/phase5/models/harveylm/qlora_headtail
mkdir -p code/phase5/submissions/harveylm/qlora_headtail

$PYTHON $SCRIPT \
    --arch harveylm \
    --adaptation qlora \
    --r 32 \
    --alpha 64 \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 2.5e-4 \
    --output_dir code/phase5/models/harveylm/qlora_headtail \
    --sub_dir code/phase5/submissions/harveylm/qlora_headtail

echo ""
echo "=========================================================================="
echo "HARVEYLM QLORA COMPLETED SUCCESSFULLY AT: $(date)"
echo "=========================================================================="
