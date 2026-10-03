# repro_timesnet_tn0_timesx_v1

TN0: TimesNet-native (random init, full-param) on TimesX, trained on the
4,041 native train windows of the frozen LN snapshot
(`VisionTS_Experiments/repro_ln_timesx_v1`). Fourth controlled baseline of
the VisionTS_Exp campaign (after T1/T0/P0): the model is the official thuml
`Time-Series-Library` forecasting path (`models/TimesNet.py` +
`layers/{Embed,Conv_Blocks}.py`), **vendored read-only** into
`vendor/TimesNet/` (upstream commit `74e58ebf` of
`T-s-cell/Time-Series-Library` — the 3 model files there are byte-identical
to theta's pinned working-tree commit `1d086314`; file sha256s + 2-line
import diff pinned in configs/v1.yaml; `model_check` proves upstream and
adapted are bitwise-equal on identical weights+inputs in isolated
processes). Data, sampling, budget, selection and scoring are inherited
verbatim from the accepted T0/P0 campaigns. Targets: TN0 vs F0 (VisionTS
LN-native, MSE 3.546316±0.004496), T0 (TimeMixer-native, MSE
3.973380±0.118713) and P0 (PatchTST-native, MSE 3.941538±0.028879);
acceptance is protocol correctness and evidence completeness, NOT TN0
winning.

- 171 candidates: 19 domains x lr {1e-4, 1e-3, 1e-2} x seeds {2021, 2022,
  2023}; TimesNet fixed structure (measured pins fill configs/v1.yaml
  before freeze; fp32, TF32 off, deterministic): 96->12, e_layers 2,
  d_model 16, d_ff 32, top_k 5, num_kernels 6, dropout 0.1, embed timeF,
  freq h; single variable [B,96,1] -> 4-arg forecast call
  `model(x_enc, None, None, None)` -> tail 12 of [B,108,1]; internal
  normalization mean/var(unbiased=False)+1e-5 (detached, non-adaptive);
  no decoder (x_dec/x_mark_dec/label_len unused - disclosed); every
  parameter trains (non-trainable list is EMPTY; the only buffer is the
  fixed positional pe, proven never to move).
- **B=1 evaluation (frozen rule)**: TimesNet's `FFT_for_Period` selects
  top-k periods from the batch-mean amplitude spectrum, so with B>1 the
  windows' period selections couple. Every evaluation forward
  (epoch-0 diagnostic, per-epoch validation, final test, recheck replay)
  runs through `forward_pred`, which asserts B==1 (`test.eval_batch=1`).
  Training keeps the protocol batch of 32 via `forward_train` (any B).
  A batch of identical copies of one window provably reproduces that
  window's own selection (preflight B); real-window batch sensitivity is a
  DIAGNOSTIC only — distinct windows may coincidentally agree. The hard
  batch rules are: forward_pred B==1 assertion, eval-order invariance, and
  checkpoint save/reload/re-inference bitwise agreement.
- With x_mark=None the timeF temporal embedding (Linear 4->16, bias=False)
  never enters the forward: its grad stays None after every real backward
  (positive perturbation proof in preflight B) and Adam skips it; pinned
  and disclosed, vendor code unchanged.
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
  (3 x 2,474 = 7,422 window predictions). TN0-vs-F0/T0/P0 reference
  metrics are parsed at 6-dp precision (pre-registered values in
  configs/v1.yaml), never recomputed.

## Stages (theta)

```bash
PY=/dev_data/wlt/conda/envs/timemixer/bin/python
GPU=GPU-eed27426-e2dc-589b-c763-81251e9563d8   # theta GPU0; same for CVD+UUID
LN=/dev_data/wlt/VisionTS_Experiments/repro_ln_timesx_v1
T0=/dev_data/wlt/Time-Series-Library/repro_timemixer_t0_timesx_v1/results
P0=/dev_data/wlt/Time-Series-Library/repro_patchtst_p0_timesx_v1/results
cd /dev_data/wlt/Time-Series-Library
# 0) one-off, before prepare: measure the model pins on GPU, fill configs/v1.yaml
env PY=$PY CUDA_VISIBLE_DEVICES=$GPU $PY -u repro_timesnet_tn0_timesx_v1/vendor_probe.py
# 1) freeze + pre-flight evidence
env PY=$PY TN0_LN_DIR=$LN bash repro_timesnet_tn0_timesx_v1/run_all.sh prepare
# model_check runs in the SAME pinned CUDA env as every other stage: its
# launch bindings (device_count etc.) must agree with the other reports or
# pretrain_audit/launch_gate refuse the chain
env PY=$PY TN0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TN0_GPU_UUID=$GPU \
  bash repro_timesnet_tn0_timesx_v1/run_all.sh model_check
env PY=$PY TN0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TN0_GPU_UUID=$GPU \
  bash repro_timesnet_tn0_timesx_v1/run_all.sh cuda_tests
env PY=$PY TN0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TN0_GPU_UUID=$GPU \
  bash repro_timesnet_tn0_timesx_v1/run_all.sh preflight
env PY=$PY TN0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TN0_GPU_UUID=$GPU \
  bash repro_timesnet_tn0_timesx_v1/run_all.sh pretrain_audit
#    -> pulls together audit/TimesNet_TN0_pretrain_audit_with_source_<UTC>.tar.gz
#       *** HARD STOP: fetch to the reviewer; training starts ONLY after
#       *** the user approves that bundle.
# 2) AFTER APPROVAL: train + everything after (pretrain_audit must be PASS)
env PY=$PY TN0_LN_DIR=$LN CUDA_VISIBLE_DEVICES=$GPU TN0_GPU_UUID=$GPU \
  nohup bash -c 'set -e
    bash repro_timesnet_tn0_timesx_v1/run_all.sh train
    bash repro_timesnet_tn0_timesx_v1/run_all.sh freeze_selection
    bash repro_timesnet_tn0_timesx_v1/run_all.sh test
    bash repro_timesnet_tn0_timesx_v1/run_all.sh aggregate \
      --ln-dir '"$LN"' --t0-dir '"$T0"' --p0-dir '"$P0"'
    bash repro_timesnet_tn0_timesx_v1/run_all.sh recheck
    bash repro_timesnet_tn0_timesx_v1/run_all.sh audit --tar' \
  > repro_timesnet_tn0_timesx_v1/logs/train_nohup.log 2>&1 &
```

Stage logs land in `logs/` (inside this dir, never the repo root). Resume =
re-run the same stage: SUCCESS/DIVERGED runs are skipped, interrupted runs
continue from `resume.pt` (model+optimizer+epoch+best+patience+4 RNG states;
dropout stream replayed bitwise). Fingerprint drift refuses with an error.

- `touch repro_timesnet_tn0_timesx_v1/PAUSE` pauses before the next
  candidate.
- GPU gate: free VRAM on TN0_GPU_UUID >= preflight peak + 3 GiB; co-tenant
  processes are never touched; no CPU fallback.
- Divergence (NaN/Inf in loss/grad/params/val) -> `DIVERGED.json` evidence,
  queue continues, selection score +inf; a domain whose every lr has >=1
  diverged seed -> `results/MUST_PARK.json` + exit 4 (must-park, never
  waived; a restart refuses while the file exists); OOM/data/code errors
  stop the queue with the scene preserved (exit 3).
- Launch gate (train/freeze/test/recheck): four reports must be PASS with
  bindings matching the live tree — `audit/preflight_report.json`,
  `audit/model_check_report.json`, `audit/cuda_path_tests_report.json`,
  `audit/pretrain_audit_report.json` — else the stage aborts with exit 2.
- recheck: two evidence classes — (A) metric recompute from the saved npz +
  raw snapshot series must reproduce stored d_i/MSE/MAE and the full
  aggregation bitwise (frozen 1e-12 sanity bound); (B) inference replay of
  the 57 selected checkpoints at B=1, primary assertion bitwise vs saved
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
| `model_adapter.py` | vendored TimesNet only (tn_models/tn_layers); numerics + pin guards; `forward_pred` (B==1, eval) / `forward_train` (any B, training) |
| `vendor/TimesNet/` | upstream pristine files, adapted runtime copies (2-line import diff), import_diff.patch, UPSTREAM.md |
| `vendor_probe.py` | GPU measurement of the model pins -> audit/vendor_probe.json (run BEFORE prepare) |
| `model_check.py` | upstream-vs-adapted bitwise equivalence in isolated processes -> audit/model_check_report.json (hard launch gate) |
| `trainer.py` | native-pool sampling (F0 clone), loss, eval, resume, training loop (update via forward_train) |
| `ln_probe.py` | slot probe hashes from LN's real build_slots (ast) + native slot equality (570 combos) |
| `prepare.py` | vendor + LN material verification, read-only export, frozen projection |
| `preflight.py` | A-M evidence gates (B = TimesNet structure/grad/FFT pins + B=1 basis; D native pools + 570 F0 alignment; F CUDA bitwise numerics incl. circular-padding backward; H resume + checkpoint reload replay) + launch bindings |
| `cuda_path_tests.py` | T1-T6 on the real CUDA path (divergence wiring, kill+resume, early-stop finalize, must-park, TN0 aggregate degradation + 3-reference render, launch gate) |
| `pretrain_audit.py` | training-gate deliverable: report json written BEFORE the tar; bundle includes ALL sources + vendor tree + all pre-train reports; run_all refuses training unless PASS and user-approved |
| `queue_train.py` | serial 171-run queue, launch+GPU gates, divergence state machine, must-park |
| `freeze_selection.py` | terminal-state audit + per-domain lr selection |
| `evaluate_test.py` | 57 models, each on its own domain's native test windows at B=1 (npz per run, incl. start_idx) |
| `aggregate.py` | TN0 tables + TN0-vs-F0/T0/P0 (6-dp import validation on all three references; reference CSVs parsed, never recomputed); 4 new disclosures |
| `recheck.py` | independent metric recompute + 57-checkpoint B=1 inference replay |
| `audit.py` | baseline diff, hash chain, 171-run + coverage + recheck review; `--tar` builds ONE deduplicated final bundle with MANIFEST.sha256 (generated + member-verified) |
| `run_all.sh` | stage driver (fail-fast, arg passthrough, logs inside this dir) |

Isolation rules: TSLib tracked files untouched (theta HEAD stays
`1d086314`, TN0 stays untracked); new content only inside this dir and
`vendor/TimesNet/`; LN dir read-only (file hashes re-verified at
prepare/audit); the T0 and P0 experiment dirs are never touched (disclosed
side effect: while TN0 exists untracked alongside them, the frozen T0/P0
protocols would each see the others as extra violations, so those frozen
protocols are not re-run on this machine); the theta `timemixer` conda env
is dedicated to this campaign (template envs `pytorch_init`/`visionts`
never modified).
