#!/bin/bash
# ==============================================================================
# Phase 7 Submission 6: Dense-Lexical Hybrid Consensus (Reciprocal Rank Fusion)
# Combines: Top Dense Retriever (BaselineLM LoRA) + Lexical BM25
# Hardware: CPU / Instant Execution
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

DENSE_PRED="${1:-${CODE_DIR}/phase7/submissions/baseline/lora/prediction.json}"
SUB_DIR="${CODE_DIR}/phase7/submissions/ensemble/hybrid_rrf"
mkdir -p "${SUB_DIR}"

echo "=========================================================================="
echo "LAUNCHING PHASE 7 SUBMISSION 6: DENSE-LEXICAL HYBRID RRF"
echo "Dense Prediction Source: ${DENSE_PRED}"
echo "Submissions Dir: ${SUB_DIR}"
echo "=========================================================================="

${VENV_PYTHON} -u "${CODE_DIR}/phase7/scripts/run_pcr_pipeline.py" \
    --arch hybrid \
    --adaptation hybrid \
    --dense_pred_file "${DENSE_PRED}" \
    --w_dense 0.65 \
    --sub_dir "${SUB_DIR}"

echo "=========================================================================="
echo "PHASE 7 SUBMISSION 6 COMPLETED AT: $(date)"
echo "=========================================================================="
