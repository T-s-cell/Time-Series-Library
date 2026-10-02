#!/usr/bin/env bash
# Stage driver for the TimeMixer TimesX T0 experiment (native pool, spec v1).
# Usage: bash repro_timemixer_t0_timesx_v1/run_all.sh <stage> [extra args...]
#   stage in {prepare, preflight, cuda_tests, train, freeze_selection, test,
#   aggregate, audit}; extra args are passed through to the stage script
#   (e.g. aggregate --ln-dir <dir> --t1-dir <dir>, audit --tar).
# cuda_tests: remediation evidence on the real CUDA path (run after prepare,
# before the final preflight; sandboxed, produces no test metrics).
# Fail-fast; every stage's output is teed INSIDE this experiment dir
# (logs/stage_<stage>_<ts>.log). Env (set ONCE for the whole campaign):
#   PY                    python executable (default python3; on theta use the
#                         /dev_data/anaconda3/envs/timemixer/bin/python env)
#   TM_LN_DIR             LN run dir (prepare + preflight section D probes)
#   CUDA_VISIBLE_DEVICES  pin compute to one theta GPU (same UUID as below,
#                         identical across preflight/cuda_tests/train/test)
#   TM_GPU_UUID           same GPU uuid, for the queue's nvidia-smi gate
#   TM_PAUSE file (PAUSE sentinel) is honored by the queue's GPU gate.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
STAGE="${1:?usage: run_all.sh <prepare|preflight|cuda_tests|train|freeze_selection|test|aggregate|audit> [-- extra args]}"
shift || true

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

echo "[run_all] stage=$STAGE py=$PY log=$LOG args=$*"
set -o pipefail
"$PY" -u "$CMD" "$@" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
if [ "$rc" -ne 0 ]; then
  echo "[run_all] stage $STAGE FAILED (rc=$rc); log: $LOG" >&2
  exit "$rc"
fi
echo "[run_all] stage $STAGE OK"
