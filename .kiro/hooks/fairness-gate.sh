#!/usr/bin/env bash
# Kiro Hook: fairness-gate
# Trigger: onFileChange — models/*.py
# Purpose: Block model promotion if fairness metrics fail threshold
# Spec: CREDIT-001 NFR-004, CREDIT-002 NFR-003

set -euo pipefail

MODEL_FILE="${1:-models/fraud_detector.py}"
LOG_FILE=".kiro/hooks/fairness-gate.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [fairness-gate] Running fairness check on ${MODEL_FILE}" | tee -a "$LOG_FILE"

# Run fairness evaluation
python governance/fairness_gate.py \
  --model-file "$MODEL_FILE" \
  --threshold-demographic-parity 0.05 \
  --threshold-equal-opportunity 0.05 \
  --log-file "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [fairness-gate] BLOCKED: Fairness thresholds not met. See ${LOG_FILE} for details." | tee -a "$LOG_FILE"
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  ⚠  FAIRNESS GATE FAILED                               │"
    echo "│                                                         │"
    echo "│  Model cannot be promoted until fairness metrics pass.  │"
    echo "│  Review: .kiro/hooks/fairness-gate.log                  │"
    echo "│  Override: python governance/fairness_gate.py --hitl    │"
    echo "└─────────────────────────────────────────────────────────┘"
    exit 1
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [fairness-gate] PASSED: All fairness metrics within threshold." | tee -a "$LOG_FILE"
