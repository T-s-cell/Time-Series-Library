#!/usr/bin/env bash
# Stage driver for the PatchTST TimesX P0 experiment (native pool, spec v1).
# Usage: bash repro_patchtst_p0_timesx_v1/run_all.sh <stage> [extra args...]
#   stage in {prepare, vendor_check, cuda_tests, preflight, pretrain_audit,
#   train, freeze_selection, test, aggregate, recheck, audit}; extra args are
#   passed through to the stage script (e.g. aggregate --ln-dir <dir>
#   --t0-dir <dir>, audit --tar).
# Order matters: prepare -> vendor_check -> cuda_tests -> preflight ->
# pretrain_audit -> train -> ... ; every later stage re-verifies the bindings
# frozen earlier (launch_gate refuses stale/drifted evidence).
# Fail-fast; every stage's output is teed INSIDE this experiment dir
# (logs/stage_<stage>_<ts>.log). Env (set ONCE for the whole campaign):
#   PY                    python executable (default python3; on theta use
#                         /dev_data/wlt/conda/envs/timemixer/bin/python)
#   P0_LN_DIR             LN run dir (prepare + preflight section D probes)
#   CUDA_VISIBLE_DEVICES  pin compute to one theta GPU (same UUID as below,
#                         identical across preflight/cuda_tests/train/test)
#   P0_GPU_UUID           same GPU uuid, for the queue's nvidia-smi gate
#   P0_PREFLIGHT_ALLOW_CPU  dev-only CPU fallback for preflight (never train)
#   PAUSE file (PAUSE sentinel) is honored by the queue's GPU gate.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
STAGE="${1:?usage: run_all.sh <prepare|vendor_check|cuda_tests|preflight|pretrain_audit|train|freeze_selection|test|aggregate|recheck|audit> [-- extra args]}"
shift || true

cd "$HERE"
mkdir -p logs
LOG="logs/stage_${STAGE}_$(date +%Y%m%d_%H%M%S).log"

case "$STAGE" in
  prepare)          CMD="prepare.py" ;;
  vendor_check)     CMD="vendor_check.py" ;;
  cuda_tests)       CMD="cuda_path_tests.py" ;;
  preflight)        CMD="preflight.py" ;;
  pretrain_audit)   CMD="pretrain_audit.py" ;;
  train)            CMD="queue_train.py" ;;
  freeze_selection) CMD="freeze_selection.py" ;;
  test)             CMD="evaluate_test.py" ;;
  aggregate)        CMD="aggregate.py" ;;
  recheck)          CMD="recheck.py" ;;
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
