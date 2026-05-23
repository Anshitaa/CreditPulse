#!/usr/bin/env bash
# Kiro Hook: auto-test
# Trigger: onFileChange — models/*.py, features/*.py
# Purpose: Auto-run unit tests for modified ML components

set -euo pipefail

CHANGED_FILE="${1:-}"
LOG_FILE=".kiro/hooks/auto-test.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [auto-test] Running tests for: ${CHANGED_FILE}" | tee -a "$LOG_FILE"

# Determine which test suite to run based on changed file
if [[ "$CHANGED_FILE" == models/* ]]; then
    TEST_PATH="tests/unit/test_models.py"
elif [[ "$CHANGED_FILE" == features/* ]]; then
    TEST_PATH="tests/unit/test_features.py"
elif [[ "$CHANGED_FILE" == governance/* ]]; then
    TEST_PATH="tests/unit/test_governance.py"
else
    TEST_PATH="tests/unit/"
fi

python -m pytest "$TEST_PATH" -v --tb=short 2>&1 | tee -a "$LOG_FILE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [auto-test] Done." | tee -a "$LOG_FILE"
