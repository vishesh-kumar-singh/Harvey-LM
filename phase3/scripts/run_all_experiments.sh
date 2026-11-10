#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase3/scripts/train_adaptation_and_kd.py"

echo "=========================================================================="
echo "Phase 3: Comprehensive Adaptation & Knowledge Distillation Matrix"
echo "Started at: $(date)"
echo "=========================================================================="

# 1. QLoRA Adaptation on HarveyLM (COMPLETED - 0.5084 Macro-F1)
# echo ">>> [1/11] Running QLoRA on HarveyLM (4-bit NF4 Quantization + LoRA)..."
# $PYTHON $SCRIPT --arch harveylm --mode qlora --epochs 5 --lr 2.5e-4 --batch_size 32 \
#     --output_dir code/phase3/models/qlora/qlora_harveylm \
#     --sub_dir code/phase3/submissions/harveylm/qlora

# 2. LoRA Adaptation on BaselineLM (COMPLETED - 0.5074 Macro-F1)
# echo ">>> [2/11] Running LoRA on BaselineLM (Custom LLaMA style)..."
# $PYTHON $SCRIPT --arch baseline --mode lora --epochs 5 --lr 2.5e-4 --batch_size 32 \
#     --output_dir code/phase3/models/baseline/lora \
#     --sub_dir code/phase3/submissions/baseline/lora

# 3. LORMA Adaptation on BaselineLM (COMPLETED - 0.5094 Macro-F1)
# echo ">>> [3/11] Running LORMA on BaselineLM (Truncated SVD Matrix Approximation)..."
# $PYTHON $SCRIPT --arch baseline --mode lorma --epochs 5 --lr 1.5e-4 --batch_size 32 \
#     --output_dir code/phase3/models/baseline/lorma \
#     --sub_dir code/phase3/submissions/baseline/lorma

# 4. LoRA Adaptation on HarveyMoE
echo ""
echo ">>> [4/11] Running LoRA on HarveyMoE (Expert Attention Projections)..."
$PYTHON $SCRIPT --arch moe --mode lora --epochs 5 --lr 2.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/moe/lora \
    --sub_dir code/phase3/submissions/moe/lora

# 5. LORMA Adaptation on HarveyMoE
echo ""
echo ">>> [5/11] Running LORMA on HarveyMoE (Truncated SVD Matrix Approximation)..."
$PYTHON $SCRIPT --arch moe --mode lorma --epochs 5 --lr 1.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/moe/lorma \
    --sub_dir code/phase3/submissions/moe/lorma

# 6. LoRA Adaptation on LegalEncoderDecoder
echo ""
echo ">>> [6/11] Running LoRA on LegalEncoderDecoder..."
$PYTHON $SCRIPT --arch enc_dec --mode lora --epochs 5 --lr 2.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/enc_dec/lora \
    --sub_dir code/phase3/submissions/enc_dec/lora

# 7. LORMA Adaptation on LegalEncoderDecoder
echo ""
echo ">>> [7/11] Running LORMA on LegalEncoderDecoder..."
$PYTHON $SCRIPT --arch enc_dec --mode lorma --epochs 5 --lr 1.5e-4 --batch_size 32 \
    --output_dir code/phase3/models/enc_dec/lorma \
    --sub_dir code/phase3/submissions/enc_dec/lorma

# 8. Knowledge Distillation on HarveyLM
echo ""
echo ">>> [8/11] Running Knowledge Distillation on HarveyLM from 0.4785 LB Grand Consensus..."
$PYTHON $SCRIPT --arch harveylm --mode kd --epochs 5 --lr 2.0e-4 --batch_size 32 \
    --output_dir code/phase3/models/kd/kd_harveylm \
    --sub_dir code/phase3/submissions/harveylm/kd

# 9. Knowledge Distillation on BaselineLM
echo ""
echo ">>> [9/11] Running Knowledge Distillation on BaselineLM..."
$PYTHON $SCRIPT --arch baseline --mode kd --epochs 5 --lr 2.0e-4 --batch_size 32 \
    --output_dir code/phase3/models/kd/kd_baseline \
    --sub_dir code/phase3/submissions/baseline/kd

# 10. Knowledge Distillation on HarveyMoE
echo ""
echo ">>> [10/11] Running Knowledge Distillation on HarveyMoE..."
$PYTHON $SCRIPT --arch moe --mode kd --epochs 5 --lr 2.0e-4 --batch_size 32 \
    --output_dir code/phase3/models/kd/kd_moe \
    --sub_dir code/phase3/submissions/moe/kd

# 11. Knowledge Distillation on LegalEncoderDecoder
echo ""
echo ">>> [11/11] Running Knowledge Distillation on LegalEncoderDecoder..."
$PYTHON $SCRIPT --arch enc_dec --mode kd --epochs 5 --lr 2.0e-4 --batch_size 32 \
    --output_dir code/phase3/models/kd/kd_enc_dec \
    --sub_dir code/phase3/submissions/enc_dec/kd

echo ""
echo "=========================================================================="
echo "ALL 11 EXPERIMENTS & SUBMISSIONS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
