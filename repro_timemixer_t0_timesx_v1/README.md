# repro_timemixer_t0_timesx_v1

T0: TimeMixer-native (random init, full-param) on TimesX, trained on the
4,041 native train windows of the frozen LN snapshot
(`VisionTS_Experiments/repro_ln_timesx_v1`). Derived from the audited
T1 experiment `repro_timemixer_timesx_v1` (commit add6d65) — 16 files
copied verbatim, then minimally patched; `diff -ru` against T1 is the
code-diff deliverable. Targets: T0 vs F0 (VisionTS LN-native main result,
no zero-shot fallback) and T0 vs T1 (training-pool effect, dense MSE
3.842157±0.043400).

- 171 candidates: 19 domains x lr {1e-4, 1e-3, 1e-2} x seeds {2021, 2022,
  2023}; shared TimeMixer (63,531 params, fp32, TF32 off, deterministic),
  single variable [B,96,1] -> [B,12].
- Native training pool: slots hold native sample_ids, drawn WITH
  replacement per variable (LN build_slots F0 branch, RNG families and
  call order cloned line-by-line; epochs 1..10 reproduce LN F0 sampling
  for the same (domain, seed) — verified 570/570 by native_slot_equality).
  Budget stays dense-derived: U_d = ceil(D_dense_d/32), S_d = 32*U_d,
  sum(U_d) = 1157 — never shrunk to native pool size. Native val 895 /
  test 2,474 windows from the frozen LN snapshot.
- Loss/metrics standardized by per-window input pstdev (fallback_std from
  the manifest, no gradient); val = var-internal mean -> var-equal mean.
- Per-domain lr = mean of 3 seeds' best val std-MSE (any diverged seed ->
  +inf; tie -> smaller lr); freeze -> test 57 models; T0-vs-F0 and
  T0-vs-T1 comparisons (reference metrics parsed from the runs' results/,
  never recomputed).

## Stages (theta)

```bash
PY=/dev_data/anaconda3/envs/timemixer/bin/python
GPU=GPU-eed27426-e2dc-589b-c763-81251e9563d8   # theta GPU0; same for CVD+UUID
LN=$HOME/VisionTS_Experiments/repro_ln_timesx_v1
cd $HOME/Time-Series-Library
env PY=$PY TM_LN_DIR=$LN bash repro_timemixer_t0_timesx_v1/run_all.sh prepare
env PY=$PY TM_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TM_GPU_UUID=$GPU \
  bash repro_timemixer_t0_timesx_v1/run_all.sh cuda_tests
env PY=$PY TM_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TM_GPU_UUID=$GPU \
  bash repro_timemixer_t0_timesx_v1/run_all.sh preflight
# after preflight PASS, the four env vars stay identical for train/test:
env PY=$PY TM_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TM_GPU_UUID=$GPU \
  nohup bash -c 'set -e
    bash repro_timemixer_t0_timesx_v1/run_all.sh train
    bash repro_timemixer_t0_timesx_v1/run_all.sh freeze_selection
    bash repro_timemixer_t0_timesx_v1/run_all.sh test
    bash repro_timemixer_t0_timesx_v1/run_all.sh aggregate \
      --ln-dir $LN --t1-dir $HOME/t1_results_20261001
    "$PY" -u repro_timemixer_t0_timesx_v1/audit.py --tar' \
  > repro_timemixer_t0_timesx_v1/logs/train_nohup.log 2>&1 &
```

Stage logs land in `logs/` (inside this dir, never the repo root). Resume =
re-run the same stage: SUCCESS/DIVERGED runs are skipped, interrupted runs
continue from `resume.pt` (model+optimizer+epoch+best+patience+4 RNG states;
dropout stream replayed bitwise). Fingerprint drift refuses with an error.

- `touch repro_timemixer_t0_timesx_v1/PAUSE` pauses before the next
  candidate.
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
| `trainer.py` | native-pool sampling (F0 clone), loss, eval, resume, training loop |
| `ln_probe.py` | slot probe hashes from LN's real build_slots (ast) + native slot equality |
| `prepare.py` | LN material verification + read-only export + frozen projection |
| `preflight.py` | A-M evidence gates (D native pools + 570 F0 alignment; H/F on the real device) + launch bindings |
| `cuda_path_tests.py` | remediation evidence on the real CUDA path (divergence wiring, kill+resume, early-stop finalize, must-park, T0 aggregate, launch gate) |
| `queue_train.py` | serial 171-run queue, launch+GPU gates, divergence state machine, must-park |
| `freeze_selection.py` | terminal-state audit + per-domain lr selection |
| `evaluate_test.py` | 57 models x native test windows, npz per run |
| `aggregate.py` | T0 tables + T0-vs-F0 and T0-vs-T1 (reference CSVs parsed, never recomputed) |
| `audit.py` | baseline diff, hash chain, 171-run + coverage review, bundle |
| `run_all.sh` | stage driver (fail-fast, arg passthrough, logs inside this dir) |

Isolation rules: TSLib tracked files untouched; new content only inside this
dir; LN dir read-only (file hashes re-verified at prepare/audit); the T1
experiment dir and its runtime artifacts are never touched; the theta
`timemixer` conda env is dedicated to this campaign (template envs
`pytorch_init`/`visionts` never modified).
