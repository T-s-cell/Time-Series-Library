#!/usr/bin/env bash
# Stage driver for the TimeMixer TimesX T1 experiment (spec v1).
# Usage: bash repro_timemixer_timesx_v1/run_all.sh <stage>
#   stage in {prepare, preflight, cuda_tests, train, freeze_selection, test,
#   aggregate, audit}
# cuda_tests: remediation evidence on the real CUDA path (run after prepare,
# before the final preflight; sandboxed, produces no test metrics).
# Fail-fast; every stage's output is teed INSIDE this experiment dir
# (logs/stage_<stage>_<ts>.log). Env:
#   PY           python executable (default python3; on eta use the
#                /dev_data/wlt/conda/envs/timemixer/bin/python clone)
#   TM_GPU_UUID  target GPU uuid (train/preflight section I); unset = require
#                exactly one visible GPU
#   TM_PAUSE file (PAUSE sentinel) is honored by the queue's GPU gate.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
STAGE="${1:?usage: run_all.sh <prepare|preflight|cuda_tests|train|freeze_selection|test|aggregate|audit>}"

cd "$HERE"
mkdir -p logs
LOG="logs/stage_${STAGE}_$(date +%Y%m%d_%H%M%S).log"

case "$STAGE" in
  prepare)          CMD="prepare.py" ;;
  preflight)        CMD="preflight.py" ;;
  cuda_tests)       CMD="cuda_path_tests.py" ;;
  train)            CMD="queue_train.py" ;;
  freeze_selection) CMD="freeze_selection.py" ;;
  test)             CMD="evaluate_test.py" ;;
  aggregate)        CMD="aggregate.py" ;;
  audit)            CMD="audit.py" ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "[run_all] stage=$STAGE py=$PY log=$LOG"
set -o pipefail
"$PY" -u "$CMD" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
if [ "$rc" -ne 0 ]; then
  echo "[run_all] stage $STAGE FAILED (rc=$rc); log: $LOG" >&2
  exit "$rc"
fi
echo "[run_all] stage $STAGE OK"
