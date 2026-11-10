#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."  # paths below are relative to the directory containing code/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
PYTHON="${PYTHON_BIN:-python} -u"
SCRIPT="code/phase4/scripts/train_cjpe_matrix.py"

echo "=========================================================================="
echo "Phase 4 Court Judgment Prediction (CJPE): Full Architecture Matrix"
echo "Started at: $(date)"
echo "=========================================================================="

mkdir -p code/phase4/models/logs

# 1. HarveyLM QLoRA (COMPLETED - 0.753112 Test Macro-F1)
echo ">>> [1/7] HarveyLM QLoRA: ALREADY COMPLETED (0.7531 Test Macro-F1)"

# 2. BaselineLM LORMA (Truncated SVD, 2,048 tokens)
echo ""
echo ">>> [2/7] Training BaselineLM LORMA (Custom LLaMA style + Truncated SVD)..."
$PYTHON $SCRIPT \
    --arch baseline \
    --adaptation lorma \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.0e-4 \
    --output_dir code/phase4/models/baseline/lorma_headtail \
    --sub_dir code/phase4/submissions/baseline/lorma_headtail

# 3. HarveyMoE LORMA (Sparse Mixture of Experts, 2,048 tokens)
echo ""
echo ">>> [3/7] Training HarveyMoE LORMA (Top-2 Sparse MoE + Truncated SVD)..."
$PYTHON $SCRIPT \
    --arch moe \
    --adaptation lorma \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.0e-4 \
    --output_dir code/phase4/models/moe/lorma_headtail \
    --sub_dir code/phase4/submissions/moe/lorma_headtail

# 4. LegalEncoderDecoder LORMA (Encoder-Decoder, 2,048 tokens)
echo ""
echo ">>> [4/7] Training LegalEncoderDecoder LORMA (6 Enc + 6 Dec + Truncated SVD)..."
$PYTHON $SCRIPT \
    --arch enc_dec \
    --adaptation lorma \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.0e-4 \
    --output_dir code/phase4/models/enc_dec/lorma_headtail \
    --sub_dir code/phase4/submissions/enc_dec/lorma_headtail

# 5. BaselineLM LoRA (Standard Low-Rank Adaptation, 2,048 tokens)
echo ""
echo ">>> [5/7] Training BaselineLM LoRA (Low-Rank Parameter Adaptation)..."
$PYTHON $SCRIPT \
    --arch baseline \
    --adaptation lora \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.5e-4 \
    --output_dir code/phase4/models/baseline/lora_headtail \
    --sub_dir code/phase4/submissions/baseline/lora_headtail

# 6. Extract Teacher Soft Targets from Peak HarveyLM LORMA
echo ""
echo ">>> [6/7] Extracting Teacher Soft Targets from Peak HarveyLM LORMA..."
$PYTHON code/phase4/scripts/prepare_cjpe_teacher_targets.py

# 7. Knowledge Distillation Student (LegalEncoderDecoder, 125M params)
echo ""
echo ">>> [7/7] Training Knowledge Distillation Student (LegalEncoderDecoder)..."
$PYTHON $SCRIPT \
    --arch enc_dec \
    --adaptation kd \
    --teacher_logits code/phase4/data/cache/teacher_soft_targets.pt \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2.0e-4 \
    --output_dir code/phase4/models/kd/kd_enc_dec \
    --sub_dir code/phase4/submissions/enc_dec/kd_headtail

# 8. Grand Multi-Architecture Consensus Ensemble
echo ""
echo ">>> [BONUS] Generating Grand Multi-Architecture Consensus Ensemble..."
$PYTHON code/phase4/scripts/generate_grand_ensemble.py

echo ""
echo "=========================================================================="
echo "ALL ARCHITECTURAL MATRIX EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=========================================================================="
