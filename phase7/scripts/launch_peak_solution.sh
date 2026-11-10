#!/bin/bash
# ==============================================================================
# Uses nohup with unbuffered stdout/stderr for live logging
# ==============================================================================
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU; override to pick another device
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${PYTHON_BIN:-python}"

LOG_DIR="${CODE_DIR}/phase7/models/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/peak_solution.log"

echo "=========================================================================="
echo "LAUNCHING PHASE 7 PEAK SOLUTION VIA NOHUP"
echo "Timestamp: $(date)"
echo "Log File: ${LOG_FILE}"
echo "=========================================================================="

nohup ${VENV_PYTHON} -u "${CODE_DIR}/phase7/scripts/run_peak_solution.py" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Process launched with PID: ${PID}"
echo "Monitor live logs with: tail -f \"${LOG_FILE}\""
