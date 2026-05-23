#!/usr/bin/env bash
# Kiro Hook: psi-check
# Trigger: pre-commit (runs before any git commit)
# Purpose: Warn if feature distribution drift (PSI) exceeds threshold
# Spec: CREDIT-002 FR-004

set -euo pipefail

LOG_FILE=".kiro/hooks/psi-check.log"
PSI_THRESHOLD=0.20  # Industry standard: PSI > 0.20 = significant drift

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [psi-check] Checking feature drift (PSI)..." | tee -a "$LOG_FILE"

python governance/drift_monitor.py \
  --mode psi \
  --threshold "$PSI_THRESHOLD" \
  --output-format json \
  --log-file "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│  ⚠  PSI DRIFT DETECTED                                 │"
    echo "│                                                         │"
    echo "│  Feature distribution has drifted significantly.        │"
    echo "│  Consider retraining before deploying.                  │"
    echo "│  Details: .kiro/hooks/psi-check.log                     │"
    echo "│  Override: git commit --no-verify (with justification)  │"
    echo "└─────────────────────────────────────────────────────────┘"
    exit 1  # Block the commit
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [psi-check] PASSED: PSI within acceptable range." | tee -a "$LOG_FILE"
