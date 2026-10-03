# repro_patchtst_p0_timesx_v1

P0: PatchTST-native (random init, full-param) on TimesX, trained on the
4,041 native train windows of the frozen LN snapshot
(`VisionTS_Experiments/repro_ln_timesx_v1`). Third controlled baseline of
the VisionTS_Exp campaign: the model is the official `yuqinie98/PatchTST`
`PatchTST_supervised` implementation (paper appendix A.1.4 small-dataset
structure), **vendored read-only** into `vendor/PatchTST/` (upstream commit
`204c21e`, file sha256s + 4-line import diff pinned in configs/v1.yaml;
`vendor_check` proves upstream and adapted are bitwise-equal on identical
weights+input). Data, sampling, budget, selection and scoring are inherited
verbatim from the accepted T0 campaign (`repro_timemixer_t0_timesx_v1`).
Targets: P0 vs F0 (VisionTS LN-native, MSE 3.546316±0.004496) and P0 vs T0
(TimeMixer-native, MSE 3.973380±0.118713); acceptance is protocol
correctness and evidence completeness, NOT P0 winning.

- 171 candidates: 19 domains x lr {1e-4, 1e-3, 1e-2} x seeds {2021, 2022,
  2023}; PatchTST (19,555-param-class A.1.4 structure, measured pins fill
  configs/v1.yaml before freeze; fp32, TF32 off, deterministic), single
  variable [B,96,1] -> permute -> [B,12,1]; RevIN (eps 1e-5, affine off);
  patch 16/8, padding end, patch_num 12, e_layers 3, n_heads 4, d_model 16,
  d_ff 128, GELU, flatten head, BatchNorm; the 3 fixed attention scales are
  `requires_grad=False` by upstream design (never unfrozen, single-list
  pinned); fc_dropout=0.2 sits on an unused flatten-head path (disclosed).
- Native training pool: slots hold native sample_ids, drawn WITH
  replacement per variable (LN build_slots F0 branch, RNG families and
  call order cloned line-by-line; epochs 1..10 reproduce LN F0 sampling
  for the same (domain, seed) — verified 570/570 by ln_probe).
  Budget stays dense-derived: U_d = ceil(D_dense_d/32), S_d = 32*U_d,
  sum(U_d) = 1157 — never shrunk to native pool size. Native val 895 /
  test 2,474 windows from the frozen LN snapshot.
- Loss/metrics standardized by per-window input pstdev (fallback_std from
  the manifest, no gradient); val = var-internal mean -> var-equal mean.
- Per-domain lr = mean of 3 seeds' best val std-MSE (any diverged seed ->
  +inf; tie -> smaller lr); freeze -> each of the 57 selected models
  predicts ONLY its own domain's native test windows; per seed the union
  over the 19 domain models is exactly the 2,474 unique test windows
  (3 x 2,474 = 7,422 window predictions). P0-vs-F0 and P0-vs-T0 reference
  metrics are parsed at 6-dp precision (pre-registered values in
  configs/v1.yaml), never recomputed.

## Stages (theta)

```bash
PY=/dev_data/wlt/conda/envs/timemixer/bin/python
GPU=GPU-eed27426-e2dc-589b-c763-81251e9563d8   # theta GPU0; same for CVD+UUID
LN=/dev_data/wlt/VisionTS_Experiments/repro_ln_timesx_v1
T0=/dev_data/wlt/Time-Series-Library/repro_timemixer_t0_timesx_v1/results
cd /dev_data/wlt/Time-Series-Library
# 0) one-off, before prepare: measure the model pins on GPU, fill configs/v1.yaml
env PY=$PY CUDA_VISIBLE_DEVICES=$GPU $PY -u repro_patchtst_p0_timesx_v1/vendor_probe.py
# 1) freeze
env PY=$PY P0_LN_DIR=$LN bash repro_patchtst_p0_timesx_v1/run_all.sh prepare
env PY=$PY bash repro_patchtst_p0_timesx_v1/run_all.sh vendor_check
env PY=$PY P0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU P0_GPU_UUID=$GPU \
  bash repro_patchtst_p0_timesx_v1/run_all.sh cuda_tests
env PY=$PY P0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU P0_GPU_UUID=$GPU \
  bash repro_patchtst_p0_timesx_v1/run_all.sh preflight
env PY=$PY P0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU P0_GPU_UUID=$GPU \
  bash repro_patchtst_p0_timesx_v1/run_all.sh pretrain_audit
# 2) train + everything after (pretrain_audit must be PASS first)
env PY=$PY P0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU P0_GPU_UUID=$GPU \
  nohup bash -c 'set -e
    bash repro_patchtst_p0_timesx_v1/run_all.sh train
    bash repro_patchtst_p0_timesx_v1/run_all.sh freeze_selection
    bash repro_patchtst_p0_timesx_v1/run_all.sh test
    bash repro_patchtst_p0_timesx_v1/run_all.sh aggregate \
      --ln-dir '"$LN"' --t0-dir '"$T0"'
    bash repro_patchtst_p0_timesx_v1/run_all.sh recheck
    bash repro_patchtst_p0_timesx_v1/run_all.sh audit --tar' \
  > repro_patchtst_p0_timesx_v1/logs/train_nohup.log 2>&1 &
```

Stage logs land in `logs/` (inside this dir, never the repo root). Resume =
re-run the same stage: SUCCESS/DIVERGED runs are skipped, interrupted runs
continue from `resume.pt` (model+optimizer+epoch+best+patience+4 RNG states;
dropout stream replayed bitwise). Fingerprint drift refuses with an error.

- `touch repro_patchtst_p0_timesx_v1/PAUSE` pauses before the next
  candidate.
- GPU gate: free VRAM on P0_GPU_UUID >= preflight peak + 3 GiB; co-tenant
  processes are never touched; no CPU fallback.
- Divergence (NaN/Inf in loss/grad/params/val) -> `DIVERGED.json` evidence,
  queue continues, selection score +inf; a domain whose every lr has >=1
  diverged seed -> `results/MUST_PARK.json` + exit 4 (must-park, never
  waived; a restart refuses while the file exists); OOM/data/code errors
  stop the queue with the scene preserved (exit 3).
- Launch gate (train/freeze/test/recheck): `audit/preflight_report.json`
  AND `audit/vendor_check_report.json` must be PASS and their `bindings`
  (snapshot materials / TSLib state / vendored-model md5s / experiment code
  md5s / env versions / CUDA identity) must match the live tree, or the
  stage aborts with exit 2.
- recheck: two evidence classes — (A) metric recompute from the saved npz +
  raw snapshot series must reproduce stored d_i/MSE/MAE and the full
  aggregation bitwise (frozen 1e-12 sanity bound); (B) inference replay of
  the 57 selected checkpoints, primary assertion bitwise vs saved
  predictions (pre-frozen grace rel<=1e-6 -> PASS_WITH_DISCLOSURE only;
  a mismatch is investigated, never absorbed by loosening tolerances).
- A resume whose replayed patience is already exhausted finalizes the run
  immediately - early stop is never followed by extra epochs.

## Files

| file | role |
|---|---|
| `configs/v1.yaml` | frozen protocol: pins, budgets, rules, comparison + recheck tolerances (JSON, no yaml dep) |
| `common.py` | paths, hashes, fingerprints (incl. vendor md5 state), DivergenceError, launch gate |
| `data.py` | DenseData over the exported LN snapshot (torch-free) |
| `model_adapter.py` | vendored PatchTST only (ptst_models/ptst_layers); numerics + pin guards; single-arg forward |
| `vendor/PatchTST/` | upstream pristine files, adapted runtime copies, import_diff.patch, UPSTREAM.md |
| `vendor_probe.py` | GPU measurement of the model pins -> audit/vendor_probe.json (run BEFORE prepare) |
| `vendor_check.py` | upstream-vs-adapted bitwise equivalence in isolated processes -> audit/vendor_check_report.json (hard launch gate) |
| `trainer.py` | native-pool sampling (F0 clone), loss, eval, resume, training loop |
| `ln_probe.py` | slot probe hashes from LN's real build_slots (ast) + native slot equality (570 combos) |
| `prepare.py` | vendor + LN material verification, read-only export, frozen projection |
| `preflight.py` | A-M evidence gates (B = PatchTST structure/behavior pins; D native pools + 570 F0 alignment; F CUDA bitwise numerics) + launch bindings |
| `cuda_path_tests.py` | T1-T6 on the real CUDA path (divergence wiring, kill+resume, early-stop finalize, must-park, P0 aggregate degradation, launch gate) |
| `pretrain_audit.py` | training-gate deliverable: pretrain_audit.md + evidence bundle; run_all refuses training unless PASS |
| `queue_train.py` | serial 171-run queue, launch+GPU gates, divergence state machine, must-park |
| `freeze_selection.py` | terminal-state audit + per-domain lr selection |
| `evaluate_test.py` | 57 models, each on its own domain's native test windows (npz per run, incl. start_idx) |
| `aggregate.py` | P0 tables + P0-vs-F0 and P0-vs-T0 (6-dp import validation; reference CSVs parsed, never recomputed) |
| `recheck.py` | independent metric recompute + 57-checkpoint inference replay |
| `audit.py` | baseline diff, hash chain, 171-run + coverage + recheck review, evidence bundle |
| `run_all.sh` | stage driver (fail-fast, arg passthrough, logs inside this dir) |

Isolation rules: TSLib tracked files untouched (HEAD stays add6d65, P0 stays
untracked); new content only inside this dir and `vendor/PatchTST/`; LN dir
read-only (file hashes re-verified at prepare/audit); the T0 experiment dir
is never touched (disclosed side effect: while P0 exists untracked, T0's own
gates would see P0 as an extra violation, so the frozen T0 protocol is not
re-run on this machine); the theta `timemixer` conda env is dedicated to
this campaign (template envs `pytorch_init`/`visionts` never modified).
