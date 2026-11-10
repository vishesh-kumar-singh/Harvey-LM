#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase3/scripts/train_adaptation_and_kd.py"

echo "=========================================================================="
echo "Phase 3: Running QLoRA on Remaining 3 Architectures"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase3/models/baseline/qlora code/phase3/submissions/baseline/qlora
mkdir -p code/phase3/models/moe/qlora code/phase3/submissions/moe/qlora
mkdir -p code/phase3/models/enc_dec/qlora code/phase3/submissions/enc_dec/qlora

# 1. BaselineLM QLoRA
echo ""
echo ">>> [1/3] Running QLoRA on BaselineLM (4-bit NF4 Quantization + LoRA)..."
$PYTHON $SCRIPT --arch baseline --mode qlora --epochs 5 --lr 2.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/baseline/qlora \
    --sub_dir code/phase3/submissions/baseline/qlora

# 2. HarveyMoE QLoRA
echo ""
echo ">>> [2/3] Running QLoRA on HarveyMoE (4-bit NF4 Quantization + LoRA)..."
$PYTHON $SCRIPT --arch moe --mode qlora --epochs 5 --lr 2.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/moe/qlora \
    --sub_dir code/phase3/submissions/moe/qlora

# 3. LegalEncoderDecoder QLoRA
echo ""
echo ">>> [3/3] Running QLoRA on LegalEncoderDecoder (4-bit NF4 Quantization + LoRA)..."
$PYTHON $SCRIPT --arch enc_dec --mode qlora --epochs 5 --lr 2.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/enc_dec/qlora \
    --sub_dir code/phase3/submissions/enc_dec/qlora

echo ""
echo "=========================================================================="
echo "ALL REMAINING QLORA EXPERIMENTS & SUBMISSIONS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
