# repro_timemixer_timesx_v1

TimeMixer-dense (T1) supervised baseline on TimesX, run under the exact
data/sampling/scoring protocol of the VisionTS LN run
(`VisionTS_Experiments/repro_ln_timesx_v1`). Spec:
`/home/wlt/MMTS/TimeMixer_TimesX_Experiment_Plan_v1_20260929.md`.

- 171 candidates: 19 domains x lr {1e-4, 1e-3, 1e-2} x seeds {2021, 2022, 2023};
  shared TimeMixer (63,531 params, fp32, TF32 off, deterministic), single
  variable [B,96,1] -> [B,12].
- Dense-only training pool (36,763 windows, variable-balanced slots, LN's
  build_slots RNG families, epochs 1..10 reproduce LN F1 sampling); native
  val 895 / test 2,474 windows from the frozen LN snapshot.
- Loss/metrics standardized by per-window input pstdev (fallback_std from the
  manifest, no gradient); val = var-internal mean -> var-equal mean.
- Per-domain lr = mean of 3 seeds' best val std-MSE (any diverged seed ->
  +inf; tie -> smaller lr); freeze -> test 57 models; T1 vs F1 comparison.

## Stages

```bash
PY=/dev_data/wlt/conda/envs/timemixer/bin/python   # dedicated clone of visionts
cd /dev_data/wlt/Time-Series-Library
env PY=$PY TM_GPU_UUID=GPU-534dc363-5a70-bc9a-824a-309a9a2a3911 \
  bash repro_timemixer_timesx_v1/run_all.sh prepare
env PY=$PY ... bash repro_timemixer_timesx_v1/run_all.sh preflight
env PY=$PY ... nohup bash repro_timemixer_timesx_v1/run_all.sh train \
  > repro_timemixer_timesx_v1/logs/train_nohup.log 2>&1 &
# after train completes:
... run_all.sh freeze_selection && ... run_all.sh test
... run_all.sh aggregate && ... run_all.sh audit --tar   # via PY -u audit.py --tar
```

Stage logs land in `logs/` (inside this dir, never the repo root). Resume =
re-run the same stage: SUCCESS/DIVERGED runs are skipped, interrupted runs
continue from `resume.pt` (model+optimizer+epoch+best+patience+4 RNG states;
dropout stream replayed bitwise). Fingerprint drift refuses with an error.

- `touch repro_timemixer_timesx_v1/PAUSE` pauses before the next candidate.
- GPU gate: free VRAM on TM_GPU_UUID >= preflight peak + 3 GiB; co-tenant
  processes are never touched; no CPU fallback.
- Divergence (NaN/Inf in loss/grad/params/val) -> `DIVERGED.json` evidence,
  queue continues, selection score +inf; a domain whose every lr has >=1
  diverged seed -> `results/MUST_PARK.json` + exit 4 (must-park, never
  waived; a restart refuses while the file exists); OOM/data/code errors
  stop the queue with the scene preserved (exit 3).
- Launch gate (train/freeze/test): `audit/preflight_report.json` must be
  PASS and its `bindings` (snapshot materials / TSLib state / experiment
  code md5s / env versions / CUDA identity) must match the live tree, or the
  stage aborts with exit 2.
- A resume whose replayed patience is already exhausted finalizes the run
  immediately - early stop is never followed by extra epochs.

## Files

| file | role |
|---|---|
| `configs/v1.yaml` | frozen protocol: pins, budgets, rules (JSON, no yaml dep) |
| `common.py` | paths, hashes, fingerprints, DivergenceError, constants |
| `data.py` | DenseData over the exported LN snapshot (torch-free) |
| `model_adapter.py` | repo TimeMixer, read-only; numerics + shape guards |
| `trainer.py` | sampling, loss, eval, resume, per-run training loop |
| `ln_probe.py` | slot probe hashes from LN's real build_slots (ast) |
| `prepare.py` | LN material verification + read-only export + frozen projection |
| `preflight.py` | A-M evidence gates (H/F on the real device when CUDA) + launch bindings |
| `cuda_path_tests.py` | remediation evidence on the real CUDA path (divergence wiring, kill+resume, early-stop finalize, must-park, LN-missing aggregate, launch gate) |
| `queue_train.py` | serial 171-run queue, launch+GPU gates, divergence state machine, must-park |
| `freeze_selection.py` | terminal-state audit + per-domain lr selection |
| `evaluate_test.py` | 57 models x native test windows, npz per run |
| `aggregate.py` | T1 tables + T1-vs-F1 (LN CSVs parsed, never recomputed) |
| `audit.py` | baseline diff, hash chain, 171-run + coverage review, bundle |
| `run_all.sh` | stage driver (fail-fast, logs inside this dir) |

Isolation rules: TSLib tracked files untouched; new content only inside this
dir; LN dir read-only (file hashes re-verified at prepare/audit); the VisionTS
conda env is never modified (timemixer env is a clone).
