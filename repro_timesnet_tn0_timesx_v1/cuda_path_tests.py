#!/usr/bin/env python3
"""cuda_path_tests: remediation evidence on the REAL CUDA training path.

Runs after code fixes and prepare (v2 code fingerprint), BEFORE the final
preflight. Exercises on the actual device the full run will use:

  T1  divergence -> DIVERGED.json written by the queue, queue CONTINUES to
      the next candidate; re-invocation skips terminal states, never retrains
  T2  mid-flight kill (SIGKILL of a real child process on CUDA) -> resume
      completes with a bitwise-identical val history vs an uninterrupted
      reference (4-RNG replay incl. the CUDA dropout stream)
  T3  resume after the early-stop condition was already reached (SUCCESS.json
      removed, resume.pt kept) finalizes immediately - zero extra epochs
  T4  every lr of a domain with >=1 diverged seed -> MUST_PARK.json + exit 4,
      the next candidate never starts, restart refuses while the file exists
  T5  LN/T0/P0 reference results missing -> aggregate degrades to TN0-only
      (comparison_status=ln_results_unavailable,
      t0ref_comparison_status=t0_results_unavailable,
      p0ref_comparison_status=p0_results_unavailable) instead of crashing;
      all 171 candidate terminal states must be present and are counted
  T5b LN/T0/P0 reference CSVs present (overall pinned to the pre-registered
      6-dp values) -> import validation passes, comparison/improvement
      branches execute and main_table.md renders all reference cells
  T6  launch gate: for EVERY gated report (preflight / model_check /
      cuda_path_tests / pretrain_audit), missing / FAIL / stale /
      tampered-bindings evidence refuses to start the stage
  T7  recheck end-to-end path on an ISOLATED synthetic fixture: recheck's
      DenseData is replaced by _FakeRecheckData (real domain names, 190
      synthetic vars x 2 windows, real snapshot never opened); spies on the
      real test-split accessors must record ZERO calls; arts runs carry
      real checkpoint inference + bitwise replay on the synthetic windows,
      the other 54 use zero-pred stubs; metric recompute covers all 19
      domains and must reproduce methods_overall.TN0 bitwise; report PASS

All artifacts stay under audit/cuda_tmp/ (wiped on success, kept on failure
for scene inspection). Queue/aggregate outputs are redirected into the
sandbox; the frozen snapshot is only read. Divergence is injected by wrapping
trainer.forward_train (the real training forward/backward path runs until the
injection point) - no production code is modified for testing.
"""
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

import aggregate
import common
import queue_train
import recheck
import trainer
from common import (CFG, DivergenceError, DOMAIN_IDX, DOMAIN_NAMES, LRS,
                    SEEDS, atomic_torch_save, atomic_write_json, cuda_state,
                    launch_bindings, launch_gate, sha256_file)
from data import DenseData
from model_adapter import load_model
from trainer import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "audit", "cuda_tmp")
REPORT = os.path.join(HERE, "audit", "cuda_path_tests_report.json")

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append({"name": name, "ok": bool(cond), "detail": str(detail)[:400]})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {str(detail)[:160]}",
          flush=True)
    return bool(cond)


def load_json(p):
    with open(p) as f:
        return json.load(f)


class Patch:
    def __init__(self):
        self.saved = []

    def set(self, obj, attr, val):
        self.saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    def restore(self):
        for obj, attr, old in reversed(self.saved):
            setattr(obj, attr, old)
        self.saved = []


def sandbox(name):
    root = os.path.join(SB, name)
    for sub in ("checkpoints", "logs", "results", "predictions"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def aim_queue_at(p, root):
    p.set(trainer, "CHECKPOINTS", os.path.join(root, "checkpoints"))
    p.set(trainer, "LOGS", os.path.join(root, "logs"))
    p.set(queue_train, "CHECKPOINTS", os.path.join(root, "checkpoints"))
    p.set(queue_train, "LOGS", os.path.join(root, "logs"))
    p.set(queue_train, "RESULTS", os.path.join(root, "results"))
    p.set(queue_train, "MUST_PARK_JSON",
          os.path.join(root, "results", "MUST_PARK.json"))


def install_gate_report(p, root):
    """All four launch-gate reports, PASS with live bindings, redirected
    into the sandbox so launch_gate's full four-report matrix is exercised
    without touching real evidence."""
    live = launch_bindings()
    for name, attr in (("preflight", "PREFLIGHT_JSON"),
                       ("model_check", "MODEL_CHECK_JSON"),
                       ("cuda_path_tests", "CUDA_TESTS_JSON"),
                       ("pretrain_audit", "PRETRAIN_AUDIT_JSON")):
        path = os.path.join(root, f"gate_{name}.json")
        p.set(common, attr, path)
        rep = {"status": "PASS", "bindings": dict(live)}
        if name == "preflight":
            rep["sections"] = {"I_throughput": {"peak_vram_mb": 200.0}}
        atomic_write_json(rep, path)


def force_diverges(p):
    """Every run's 3rd forward call raises DivergenceError (counter resets on
    each raise so each candidate diverges exactly once, in its epoch 1)."""
    state = {"n": 0}
    real = trainer.forward_train

    def forced(model, xt):
        state["n"] += 1
        if state["n"] == 3:
            state["n"] = 0
            raise DivergenceError("forced divergence (cuda_path_tests)")
        return real(model, xt)

    p.set(trainer, "forward_train", forced)


def count_rows(root, rid, row_type):
    n = 0
    path = os.path.join(root, "logs", f"{rid}.jsonl")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    if json.loads(line).get("type") == row_type:
                        n += 1
                except json.JSONDecodeError:
                    continue
    return n


def ckpt_dirs(root):
    base = os.path.join(root, "checkpoints")
    return sorted(os.listdir(base)) if os.path.isdir(base) else []


# ---------------------------------------------------------------- T1
def t1_divergence_continues(data):
    print("[T1] divergence -> DIVERGED.json + queue continues + skip on rerun",
          flush=True)
    p = Patch()
    try:
        root = sandbox("t1")
        aim_queue_at(p, root)
        install_gate_report(p, root)
        force_diverges(p)
        p.set(queue_train, "queue_order",
              lambda d: [("arts", 1e-4, 2021), ("arts", 1e-4, 2022)])
        check("t1_queue_exit0", queue_train.main() == 0, "rc")
        rid1, rid2 = "arts__lr0.0001__s2021", "arts__lr0.0001__s2022"
        d1 = load_json(os.path.join(root, "checkpoints", rid1,
                                    "DIVERGED.json"))
        check("t1_diverged_json", d1["status"] == "diverged"
              and d1["epoch"] is not None, d1)
        man = load_json(os.path.join(root, "results", "run_manifest.json"))
        check("t1_queue_continued",
              man["diverged"] == [rid1, rid2] and man["completed"] == [],
              man)
        n_upd = count_rows(root, rid1, "update")
        u_d = -(-data.domain_dense_total("arts") // 32)
        check("t1_partial_trail", 0 < n_upd < u_d, f"{n_upd} updates (U_d={u_d})")
        check("t1_rerun_exit0", queue_train.main() == 0, "rc")
        check("t1_skip_no_retrain",
              count_rows(root, rid1, "update") == n_upd,
              f"update rows {count_rows(root, rid1, 'update')} == {n_upd}")
        man2 = load_json(os.path.join(root, "results", "run_manifest.json"))
        check("t1_skips_listed", man2["diverged"] == [rid1, rid2]
              and man2["completed"] == [], man2)
        check("t1_ckpt_dirs", ckpt_dirs(root) == [rid1, rid2],
              ckpt_dirs(root))
    except Exception:  # noqa: BLE001
        check("t1_uncaught", False, traceback.format_exc()[-400:])
    finally:
        p.restore()


# ---------------------------------------------------------------- T2
def _child(root, epochs, rid):
    """Child mode: run one real candidate on CUDA inside the sandbox until
    the parent kills it."""
    dom, lr_s, seed_s = rid.split("__")
    lr, seed = float(lr_s[2:]), int(seed_s[1:])
    trainer.CHECKPOINTS = os.path.join(root, "checkpoints")
    trainer.LOGS = os.path.join(root, "logs")
    trainer.EPOCHS = epochs
    data = DenseData()
    run_training = trainer.run_training
    run_training(dom, lr, seed, data, "cuda")


def t2_interrupt_resume(data):
    print("[T2] mid-flight kill -> resume continues bitwise-identically",
          flush=True)
    p = Patch()
    try:
        root = sandbox("t2")
        aim_queue_at(p, root)
        install_gate_report(p, root)
        rid = "arts__lr0.0001__s2022"
        out = open(os.path.join(root, "child.log"), "w")
        child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--child", root,
             "6", rid], stdout=out, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        killed = False
        while time.time() < deadline:
            if not os.path.isdir(os.path.join(root, "checkpoints")):
                break
            ep4 = os.path.join(root, "checkpoints", rid, "ep004.val.json")
            if os.path.isfile(ep4):
                child.kill()
                killed = True
                break
            if child.poll() is not None:
                break
            time.sleep(0.05)
        child.wait(timeout=30)
        out.close()
        succ = os.path.join(root, "checkpoints", rid, "SUCCESS.json")
        finished_first = child.poll() is not None and os.path.isfile(succ)
        if finished_first:
            os.remove(succ)
        check("t2_child_interrupted", killed or finished_first,
              f"killed={killed} finished_first={finished_first}")
        trainer.EPOCHS = 6
        info = trainer.run_training("arts", 1e-4, 2022, data, "cuda")
        p.restore()
        check("t2_resume_success", info.get("status") == "success",
              info.get("status"))
        check("t2_history_complete",
              sorted(map(int, info["val_mse_by_epoch"])) == [1, 2, 3, 4, 5, 6],
              sorted(map(int, info["val_mse_by_epoch"])))
        raw_att, best = {}, {}
        with open(os.path.join(root, "logs", f"{rid}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "epoch_end":
                    raw_att.setdefault(r["epoch"], set()).add(r["attempt"])
                k = (r.get("type"), r.get("epoch"), r.get("update"))
                if k not in best or r.get("attempt", -1) >= \
                        best[k].get("attempt", -1):
                    best[k] = r
        att = {e: sorted(a) for e, a in sorted(raw_att.items())}
        check("t2_attempts_consistent",
              len(att) == 6 and att[1][0] == 0
              and (att[6][-1] == 1 or finished_first)
              and all(set(a) <= {0, 1} for a in att.values()),
              f"attempts={att} finished_first={finished_first} "
              "(fresh runs start at attempt 0)")
        updates_by_epoch, end_epochs = {}, set()
        for (t, e, u), r in best.items():
            if t == "update":
                updates_by_epoch[e] = updates_by_epoch.get(e, 0) + 1
            elif t == "epoch_end":
                end_epochs.add(e)
        u_d = -(-data.domain_dense_total("arts") // 32)
        check("t2_budget_per_epoch",
              all(v == u_d for v in updates_by_epoch.values())
              and len(end_epochs) == 6,
              f"updates/epoch={updates_by_epoch} "
              f"end_epochs={sorted(end_epochs)} (U_d={u_d})")
        proot = sandbox("t2ref")
        p2 = Patch()
        p2.set(trainer, "CHECKPOINTS", os.path.join(proot, "checkpoints"))
        p2.set(trainer, "LOGS", os.path.join(proot, "logs"))
        try:
            trainer.EPOCHS = 6
            ref = trainer.run_training("arts", 1e-4, 2022, data, "cuda")
        finally:
            p2.restore()
        check("t2_bitwise_vs_reference",
              info["val_mse_by_epoch"] == ref["val_mse_by_epoch"],
              f"resumed={info['val_mse_by_epoch']} "
              f"ref={ref['val_mse_by_epoch']}")
    except Exception:  # noqa: BLE001
        check("t2_uncaught", False, traceback.format_exc()[-400:])
    finally:
        p.restore()


# ---------------------------------------------------------------- T3
def t3_early_stop_resume_finalizes(data):
    print("[T3] resume after early-stop reached finalizes, no extra epochs",
          flush=True)
    p = Patch()
    try:
        root = sandbox("t3")
        aim_queue_at(p, root)
        install_gate_report(p, root)
        rid = "arts__lr0.001__s2021"
        trainer.EPOCHS = CFG["budget"]["epochs"]
        info1 = trainer.run_training("arts", 1e-3, 2021, data, "cuda")
        check("t3_first_run_early_stop", info1.get("stopped_by")
              == "early_stop", info1.get("stopped_by"))
        succ = os.path.join(root, "checkpoints", rid, "SUCCESS.json")
        os.remove(succ)  # simulate the crash between early stop and SUCCESS
        n_upd = count_rows(root, rid, "update")
        n_end = count_rows(root, rid, "epoch_end")
        info2 = trainer.run_training("arts", 1e-3, 2021, data, "cuda")
        check("t3_finalize_success", info2.get("status") == "success",
              info2.get("status"))
        check("t3_no_extra_epochs",
              count_rows(root, rid, "update") == n_upd
              and count_rows(root, rid, "epoch_end") == n_end,
              f"updates {count_rows(root, rid, 'update')}=={n_upd}, "
              f"epoch_end {count_rows(root, rid, 'epoch_end')}=={n_end}")
        check("t3_same_selection",
              info2["local_best_epoch"] == info1["local_best_epoch"]
              and info2["local_best_val_mse"] == info1["local_best_val_mse"],
              f"{info2['local_best_epoch']}/{info1['local_best_epoch']}")
        check("t3_success_rewritten", os.path.isfile(succ), succ)
        check("t3_stopped_by", info2["stopped_by"] == "early_stop",
              info2["stopped_by"])
    except Exception:  # noqa: BLE001
        check("t3_uncaught", False, traceback.format_exc()[-400:])
    finally:
        p.restore()


# ---------------------------------------------------------------- T4
def t4_all_lrs_dead_parks(data):
    print("[T4] every lr diverged -> MUST_PARK + exit 4 + restart refused",
          flush=True)
    p = Patch()
    try:
        root = sandbox("t4")
        aim_queue_at(p, root)
        install_gate_report(p, root)
        force_diverges(p)
        p.set(queue_train, "queue_order",
              lambda d: [("arts", 1e-4, 2022), ("arts", 1e-3, 2022),
                         ("arts", 1e-2, 2022), ("arts", 1e-4, 2023)])
        code = None
        try:
            queue_train.main()
        except SystemExit as e:
            code = e.code
        check("t4_exit_code_4", code == 4, code)
        mp = os.path.join(root, "results", "MUST_PARK.json")
        check("t4_must_park_json", os.path.isfile(mp)
              and load_json(mp)["domain"] == "arts"
              and set(load_json(mp)["lrs"]) == {"%g" % lr for lr in LRS},
              load_json(mp) if os.path.isfile(mp) else "missing")
        want_dirs = ["arts__lr0.0001__s2022", "arts__lr0.001__s2022",
                     "arts__lr0.01__s2022"]
        check("t4_next_candidate_never_started",
              ckpt_dirs(root) == want_dirs, ckpt_dirs(root))
        code2 = None
        try:
            queue_train.main()
        except SystemExit as e:
            code2 = e.code
        check("t4_restart_refused", code2 == 4
              and ckpt_dirs(root) == want_dirs, f"code={code2}")
    except Exception:  # noqa: BLE001
        check("t4_uncaught", False, traceback.format_exc()[-400:])
    finally:
        p.restore()


# ---------------------------------------------------------------- T5/T5b
def build_tn0_fixtures(root, data):
    """57-run selection (every domain at lr 0.001) + test npz + terminal
    states for ALL 19x3x3 = 171 candidates: 3 diverged (first domain at the
    highest lr), 1 success that hit the 100-epoch cap, 167 early-stopped."""
    res = os.path.join(root, "results")
    pred = os.path.join(root, "predictions")
    doms = sorted(data.domain_vars)
    sel_domains = {}
    for dom in doms:
        for seed in SEEDS:
            rid = f"{dom}__lr0.001__s{seed}"
            vks = data.domain_vars[dom]
            n = len(vks)
            recs = {"var_key": np.array(vks),
                    "sample_id": np.arange(n),
                    "seed": np.full(n, seed, dtype=np.int64),
                    "checkpoint_sha256": np.full(n, "0" * 64),
                    "pred": np.zeros((n, 12)),
                    "target": np.zeros((n, 12)),
                    "d": np.ones(n)}
            err = recs["pred"] - recs["target"]
            recs["mse"] = np.mean(err ** 2, axis=1)
            recs["mae"] = np.mean(np.abs(err), axis=1)
            np.savez_compressed(os.path.join(pred, f"{rid}__test.npz"),
                                **recs)
            sel_domains.setdefault(dom, {"selected_lr": 0.001,
                                         "scores": {}, "diverged": [],
                                         "seeds": {}})["seeds"][str(seed)] \
                = {"run_id": rid}
    atomic_write_json({"frozen": True, "domains": sel_domains},
                      os.path.join(res, "selection.json"))
    for dom in doms:
        for lr in LRS:
            for seed in SEEDS:
                rid = f"{dom}__lr{lr:g}__s{seed}"
                cdir = os.path.join(root, "checkpoints", rid)
                os.makedirs(cdir, exist_ok=True)
                if dom == doms[0] and lr == LRS[-1]:
                    atomic_write_json({"status": "diverged",
                                       "reason": "forced (cuda_path_tests)",
                                       "epoch": 1},
                                      os.path.join(cdir, "DIVERGED.json"))
                elif dom == doms[1] and lr == LRS[1] and seed == SEEDS[0]:
                    atomic_write_json({"status": "success",
                                       "stopped_by": "max_epochs",
                                       "epochs_run": CFG["budget"]["epochs"],
                                       "still_improving_at_max": False},
                                      os.path.join(cdir, "SUCCESS.json"))
                else:
                    atomic_write_json({"status": "success",
                                       "stopped_by": "early_stop"},
                                      os.path.join(cdir, "SUCCESS.json"))
    return res, pred


def aim_aggregate_at(p, root, res, pred, ln_dir=None, t0_dir=None,
                     p0_dir=None):
    p.set(aggregate, "RESULTS", res)
    p.set(aggregate, "PREDICTIONS", pred)
    p.set(aggregate, "CHECKPOINTS", os.path.join(root, "checkpoints"))
    p.set(aggregate, "locate_ref_dir",
          (lambda cli, key: None)
          if ln_dir is None and t0_dir is None and p0_dir is None
          else (lambda cli, key: {"ln_results_dir_default": ln_dir,
                                  "t0_results_dir_default": t0_dir,
                                  "p0_results_dir_default": p0_dir}.get(key)))


def t5_aggregate_without_refs(data):
    print("[T5] missing LN/T0/P0 reference results -> TN0-only aggregate, no "
          "crash", flush=True)
    p = Patch()
    old_argv = sys.argv
    try:
        root = sandbox("t5")
        res, pred = build_tn0_fixtures(root, data)
        aim_aggregate_at(p, root, res, pred)
        sys.argv = ["aggregate.py"]
        rc = aggregate.main()
        check("t5_rc0", rc == 0, rc)
        summ = load_json(os.path.join(res, "aggregate_summary.json"))
        check("t5_status_degraded",
              summ["comparison_status"] == "ln_results_unavailable", summ)
        check("t5_t0_status_degraded",
              summ["t0ref_comparison_status"] == "t0_results_unavailable",
              summ)
        check("t5_p0_status_degraded",
              summ["p0ref_comparison_status"] == "p0_results_unavailable",
              summ)
        check("t5_190_vars", summ["n_vars"] == 190, summ["n_vars"])
        check("t5_methods_tn0_only",
              set(summ["methods_overall"]) == {"TN0"}, summ["methods_overall"])
        cts = summ["campaign_terminal_states"]
        check("t5_terminal_states_171",
              cts["expected_candidates"] == 171 and cts["success"] == 168
              and cts["diverged"] == 3, cts)
        check("t5_diverged_detail",
              len(cts["diverged_detail"]) == 3
              and all("forced" in (d["reason"] or "")
                      for d in cts["diverged_detail"]),
              cts["diverged_detail"])
        check("t5_max_epoch_cap_counted",
              len(summ["max_epoch_candidates"]) == 1
              and summ["max_epoch_candidates"][0]["epochs_run"]
              == CFG["budget"]["epochs"],
              summ["max_epoch_candidates"])
        check("t5_lr_counts_all_domains",
              summ["lr_selection_counts"] == {"0.001": 19},
              summ["lr_selection_counts"])
        with open(os.path.join(res, "overall.csv")) as f:
            rows = [ln.split(",")[0] for ln in f.read().splitlines()[1:]]
        check("t5_overall_rows_tn0_only", rows == ["TN0"], rows)
    except Exception:  # noqa: BLE001
        check("t5_uncaught", False, traceback.format_exc()[-400:])
    finally:
        sys.argv = old_argv
        p.restore()


def t5b_aggregate_with_refs(data):
    print("[T5b] LN/T0/P0 reference CSVs present (pre-registered 6-dp "
          "overall) -> comparison branches execute, main_table renders",
          flush=True)
    p = Patch()
    old_argv = sys.argv
    try:
        root = sandbox("t5b")
        res, pred = build_tn0_fixtures(root, data)
        doms = sorted(data.domain_vars)
        exp = CFG["comparison"]["expected_overall_6dp"]

        def write_refs(kind, method):
            d = os.path.join(root, f"{kind}_ref")
            os.makedirs(d, exist_ok=True)
            e = exp[method]
            with open(os.path.join(d, "overall.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "seed_stat", "MSE", "MAE"])
                w.writerow([method, "as reported",
                            f"{e['mse'][0]:.6f}+/-{e['mse'][1]:.6f}",
                            f"{e['mae'][0]:.6f}+/-{e['mae'][1]:.6f}"])
            with open(os.path.join(d, "domain_summary.csv"), "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "domain", "MSE", "MAE"])
                for i, dom in enumerate(doms):
                    w.writerow([method, dom,
                                f"{0.5 + i * 0.01:.6f}+/-0.000100",
                                f"{0.3 + i * 0.005:.6f}+/-0.000050"])
            with open(os.path.join(d, "test_variable_level.csv"), "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "var_key", "n_windows", "std_MSE",
                            "std_MAE", "raw_MSE", "raw_MAE"])
                for dom in doms:
                    for vk in data.domain_vars[dom]:
                        w.writerow([method, vk, 10,
                                    "0.500000+/-0.010000",
                                    "0.300000+/-0.005000",
                                    "0.600000+/-0.010000",
                                    "0.350000+/-0.005000"])
            return d

        ln_dir = write_refs("ln", "F0")
        t0_dir = write_refs("t0", "T0")
        p0_dir = write_refs("p0", "P0")
        aim_aggregate_at(p, root, res, pred, ln_dir=ln_dir, t0_dir=t0_dir,
                         p0_dir=p0_dir)
        sys.argv = ["aggregate.py"]
        rc = aggregate.main()
        check("t5b_rc0", rc == 0, rc)
        summ = load_json(os.path.join(res, "aggregate_summary.json"))
        check("t5b_comparison_ok", summ["comparison_status"] == "ok",
              summ["comparison_status"])
        check("t5b_t0_status_ok", summ["t0ref_comparison_status"] == "ok",
              summ["t0ref_comparison_status"])
        check("t5b_p0_status_ok", summ["p0ref_comparison_status"] == "ok",
              summ["p0ref_comparison_status"])
        check("t5b_methods_all_four",
              set(summ["methods_overall"]) == {"TN0", "F0", "T0", "P0"},
              sorted(summ["methods_overall"]))
        for meth in ("F0", "T0", "P0"):
            check(f"t5b_{meth.lower()}_import_matches_pin",
                  summ["methods_overall"][meth]["mse"][0]
                  == exp[meth]["mse"][0], summ["methods_overall"][meth])
        with open(os.path.join(res, "overall.csv")) as f:
            rows = [ln.split(",")[0] for ln in f.read().splitlines()[1:]]
        check("t5b_overall_rows",
              rows == ["TN0", "F0 (LN)", "T0", "P0"], rows)
        with open(os.path.join(res, "main_table.md")) as f:
            md = f.read()
        data_rows = [ln for ln in md.splitlines()
                     if ln.startswith("| ")
                     and not ln.startswith("| ---")
                     and not ln.startswith("| domain |")]
        check("t5b_main_table_rows", len(data_rows) == 20, len(data_rows))
        check("t5b_main_table_no_na", "n/a" not in md,
              [ln for ln in md.splitlines() if "n/a" in ln][:2])
        for meth in ("F0", "T0", "P0"):
            cell = (f"{exp[meth]['mse'][0]:.6f}"
                    f"+/-{exp[meth]['mse'][1]:.6f}")
            check(f"t5b_main_table_{meth.lower()}_overall_cell",
                  data_rows[-1].startswith("| Overall") and cell
                  in data_rows[-1], data_rows[-1])
        with open(os.path.join(res, "comparisons.csv")) as f:
            comp = [r[0] for r in list(csv.reader(f))[1:]]
        check("t5b_comparisons_present",
              set(comp) == {"TN0-F0", "TN0-T0", "TN0-P0"} and len(comp) == 6,
              comp)
        with open(os.path.join(res, "improvement_counts.csv")) as f:
            impr = list(csv.reader(f))[1:]
        var_rows = [r for r in impr if r[2] == "var(190)"]
        check("t5b_var_improvement_counts",
              len(var_rows) == 6
              and sorted(r[1] for r in var_rows)
              == ["F0", "F0", "P0", "P0", "T0", "T0"]
              and all(int(r[4]) == 190 and int(r[5]) == 0 for r in var_rows),
              var_rows)
    except Exception:  # noqa: BLE001
        check("t5b_uncaught", False, traceback.format_exc()[-400:])
    finally:
        sys.argv = old_argv
        p.restore()


# ---------------------------------------------------------------- T6
class _FakeRecheckData:
    """Isolated synthetic stand-in for DenseData used ONLY by the T7 recheck
    exercise: real domain NAMES (aggregate's coverage check requires them)
    over fully synthetic variables/windows. Never opens the frozen snapshot
    - the T7 run must execute with zero real test-split access (asserted by
    spies on the real DenseData test accessors)."""

    N_VARS = 10          # per domain -> 19*10 = 190 vars (honest n_vars)
    N_WINDOWS = 2        # synthetic windows per variable

    def __init__(self):
        self.doms = list(DOMAIN_NAMES)
        self.domain_vars = {d: [f"{d}__v{k}" for k in range(self.N_VARS)]
                            for d in self.doms}
        self.fb_std = {}
        for di, d in enumerate(self.doms):
            for k, vk in enumerate(self.domain_vars[d]):
                self.fb_std[vk] = 0.5 + 0.25 * ((di * 7 + k * 3) % 11)
        self._vk_idx = {vk: di * self.N_VARS + k
                        for di, d in enumerate(self.doms)
                        for k, vk in enumerate(self.domain_vars[d])}

    def split_rows(self, dom, split):
        if split != "test":
            raise RuntimeError(f"_FakeRecheckData only replaces the test "
                               f"split, got split={split!r}")
        g = np.random.default_rng(7300 + DOMAIN_NAMES.index(dom))
        rows = []
        for vk in self.domain_vars[dom]:
            for j in range(self.N_WINDOWS):
                rows.append((vk, f"{vk}__w{j}", g.standard_normal(96),
                             g.standard_normal(12)))
        return rows

    def start_of(self, vk, sid):
        return self._vk_idx[vk] * 1000 + int(sid.rsplit("__w", 1)[1])


def t7_recheck_path():
    print("[T7] recheck end-to-end path on an ISOLATED synthetic fixture "
          "(recheck.DenseData replaced; zero real test access asserted)",
          flush=True)
    from data import DenseData as RealDenseData
    from data import window_d
    p = Patch()
    old_argv = sys.argv
    real_replay = recheck.replay_rows
    try:
        root = sandbox("t7")
        res = os.path.join(root, "results")
        pred = os.path.join(root, "predictions")
        cks = os.path.join(root, "checkpoints")
        os.makedirs(res, exist_ok=True)
        os.makedirs(pred, exist_ok=True)
        device = "cuda"
        fake = _FakeRecheckData()
        arts_vks = set(fake.domain_vars["arts"])

        # zero-real-test-access guard: any live test-split accessor call on
        # the REAL DenseData during T7 is recorded and must come back empty
        test_access = []
        real_sr, real_ni, real_dt = (RealDenseData.split_rows,
                                     RealDenseData.native_ids,
                                     RealDenseData.domain_native_total)

        def spy_sr(self, dom, split):
            if split == "test":
                test_access.append(f"split_rows({dom})")
            return real_sr(self, dom, split)

        def spy_ni(self, vk, split):
            if split == "test":
                test_access.append(f"native_ids({vk})")
            return real_ni(self, vk, split)

        def spy_dt(self, dom, split):
            if split == "test":
                test_access.append(f"domain_native_total({dom})")
            return real_dt(self, dom, split)

        p.set(RealDenseData, "split_rows", spy_sr)
        p.set(RealDenseData, "native_ids", spy_ni)
        p.set(RealDenseData, "domain_native_total", spy_dt)

        sel_domains = {}
        tables = {s: [] for s in SEEDS}
        stub_sd = None
        for dom in fake.doms:
            rows = fake.split_rows(dom, "test")
            y = np.stack([r[3] for r in rows])
            starts = np.array([fake.start_of(r[0], r[1]) for r in rows],
                              dtype=np.int64)
            vks = [r[0] for r in rows]
            sids = [r[1] for r in rows]
            for seed in SEEDS:
                rid = f"{dom}__lr0.001__s{seed}"
                cdir = os.path.join(cks, rid)
                os.makedirs(cdir, exist_ok=True)
                ck = os.path.join(cdir, "best.pt")
                if dom == "arts":
                    seed_everything(4321 + seed, DOMAIN_IDX[dom])
                    model = load_model(device)
                    atomic_torch_save({k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()},
                                      ck)
                    pred_rec, d_rec = real_replay(model, rows, fake, device)
                    del model
                else:
                    if stub_sd is None:
                        stub_sd = {k: v.detach().cpu().clone() for k, v in
                                   load_model(device).state_dict().items()}
                    atomic_torch_save(stub_sd, ck)
                    pred_rec = np.zeros((len(rows), 12), dtype=np.float64)
                    d_rec = np.array([window_d(x, fake.fb_std[vk])
                                      for vk, _, x, _ in rows],
                                     dtype=np.float64)
                err = pred_rec - y
                mse = np.mean((err / d_rec[:, None]) ** 2, axis=1)
                mae = np.mean(np.abs(err) / d_rec[:, None], axis=1)
                np.savez_compressed(
                    os.path.join(pred, f"{rid}__test.npz"),
                    var_key=np.array(vks), sample_id=np.array(sids),
                    start_idx=starts, pred=pred_rec, target=y, d=d_rec,
                    mse=mse, mae=mae)
                sel_domains.setdefault(
                    dom, {"selected_lr": 0.001, "scores": {}, "diverged": [],
                          "seeds": {}})["seeds"][str(seed)] = {
                    "run_id": rid, "ckpt": "best.pt",
                    "ckpt_sha256": sha256_file(ck)}
                tables[seed].append(aggregate.var_table_of({
                    "var": vks, "mse": mse, "mae": mae,
                    "raw_mse": np.mean(err ** 2, axis=1),
                    "raw_mae": np.mean(np.abs(err), axis=1)}))
        atomic_write_json({"frozen": True, "domains": sel_domains},
                          os.path.join(res, "selection.json"))
        seed_ov = {}
        for seed in SEEDS:
            merged = aggregate.merge_var_tables(tables[seed])
            _, seed_ov[seed] = aggregate.dom_overall(merged)
        mo = {"TN0": {
            "mse": list(aggregate.seed_stats([seed_ov[s]["mse"]
                                              for s in SEEDS])),
            "mae": list(aggregate.seed_stats([seed_ov[s]["mae"]
                                              for s in SEEDS]))}}
        atomic_write_json({"methods_overall": mo, "n_vars": 190,
                           "diverged_disclosure": []},
                          os.path.join(res, "aggregate_summary.json"))
        with open(os.path.join(res, "overall.csv"), "w") as f:
            f.write("method,domains,mse,mae\n")
            f.write(f"TN0,19,{aggregate.fmt(*mo['TN0']['mse'])},"
                    f"{aggregate.fmt(*mo['TN0']['mae'])}\n")

        def stub_replay(model, rows_, data_, device_):
            if rows_[0][0] in arts_vks:
                return real_replay(model, rows_, data_, device_)
            return (np.zeros((len(rows_), 12), dtype=np.float64),
                    np.array([window_d(x, data_.fb_std[vk])
                              for vk, _, x, _ in rows_],
                             dtype=np.float64))

        p.set(recheck, "RESULTS", res)
        p.set(recheck, "PREDICTIONS", pred)
        p.set(recheck, "CHECKPOINTS", cks)
        p.set(recheck, "REPORT", os.path.join(res, "recheck_report.json"))
        p.set(recheck, "DenseData", _FakeRecheckData)
        p.set(recheck, "replay_rows", stub_replay)
        p.set(recheck, "launch_gate", lambda stage: None)
        rc = recheck.main()
        check("t7_rc0", rc == 0, rc)
        rep = load_json(os.path.join(res, "recheck_report.json"))
        check("t7_status_pass", rep.get("status") == "PASS",
              rep.get("status"))
        runs = rep.get("inference_replay", {}).get("runs", [])
        check("t7_replay_57_bitwise",
              len(runs) == 57 and all(r["verdict"] == "bitwise" for r in runs),
              f"{len(runs)} runs, verdicts "
              f"{sorted({r.get('verdict') for r in runs})}")
        check("t7_metric_ok",
              rep.get("metric_recompute", {}).get("ok") is True,
              rep.get("metric_recompute", {}).get("ok"))
        check("t7_zero_real_test_access", not test_access,
              test_access[:5] or "0 real test-split accessor calls during "
              "the whole T7 exercise (fixture + recheck.main())")
        arts_runs = [r for r in runs if r["run_id"].startswith("arts__")]
        check("t7_arts_synth_scope",
              len(arts_runs) == 3 and all(r["n_windows"] == fake.N_WINDOWS
                                          * fake.N_VARS for r in arts_runs),
              f"arts replay runs used {fake.N_WINDOWS * fake.N_VARS} "
              f"synthetic windows each: "
              f"{[r['n_windows'] for r in arts_runs]}")
    except SystemExit as exc:
        check("t7_uncaught", False, f"SystemExit: {exc}")
    except Exception:  # noqa: BLE001
        check("t7_uncaught", False, traceback.format_exc()[-400:])
    finally:
        sys.argv = old_argv
        p.restore()


def t6_launch_gate():
    print("[T6] launch gate refuses missing/FAIL/stale/tampered evidence "
          "for every gated report", flush=True)
    p = Patch()
    try:
        specs = (("preflight", "PREFLIGHT_JSON"),
                 ("model_check", "MODEL_CHECK_JSON"),
                 ("cuda_path_tests", "CUDA_TESTS_JSON"),
                 ("pretrain_audit", "PRETRAIN_AUDIT_JSON"))
        paths = {}
        live = launch_bindings()
        for name, attr in specs:
            path = os.path.join(SB, f"gate_t6_{name}.json")
            paths[name] = path
            p.set(common, attr, path)
            atomic_write_json({"status": "PASS", "bindings": dict(live)},
                              path)

        def exits(fn):
            try:
                fn()
                return None
            except SystemExit as e:
                return e.code

        rep = None
        try:
            rep = launch_gate("train")
        except SystemExit as e:
            check("t6_clean_accepts", False, f"unexpected exit {e.code}")
        check("t6_clean_accepts",
              isinstance(rep, dict) and "bindings" in rep,
              "all four reports PASS+bindings -> accepted")

        def fresh(name, obj):
            atomic_write_json(obj, paths[name])

        for name, _attr in specs:
            if os.path.isfile(paths[name]):
                os.remove(paths[name])
            check(f"t6_{name}_missing_refused",
                  exits(lambda: launch_gate("train")) == 2,
                  f"{name} missing")
            fresh(name, {"status": "FAIL", "bindings": dict(live)})
            check(f"t6_{name}_fail_refused",
                  exits(lambda: launch_gate("train")) == 2,
                  f"{name} status FAIL")
            fresh(name, {"status": "PASS"})
            check(f"t6_{name}_stale_refused",
                  exits(lambda: launch_gate("train")) == 2,
                  f"{name} no bindings block")
            b = dict(live)
            b["code"] = dict(live["code"])
            b["code"]["trainer.py"] = "0" * 32
            fresh(name, {"status": "PASS", "bindings": b})
            check(f"t6_{name}_tampered_refused",
                  exits(lambda: launch_gate("train")) == 2,
                  f"{name} bindings tampered")
            fresh(name, {"status": "PASS", "bindings": dict(live)})
    except Exception:  # noqa: BLE001
        check("t6_uncaught", False, traceback.format_exc()[-400:])
    finally:
        p.restore()


def main():
    t0 = time.time()
    cuda = cuda_state()
    if not cuda["available"]:
        print("CUDA_PATH_TESTS ABORT: CUDA unavailable - these tests verify "
              "the real training device", file=sys.stderr)
        return 2
    shutil.rmtree(SB, ignore_errors=True)
    os.makedirs(SB, exist_ok=True)
    data = DenseData()
    print(f"[cuda-tests] device={cuda['device_name']}", flush=True)
    t6_launch_gate()
    t1_divergence_continues(data)
    t2_interrupt_resume(data)
    t3_early_stop_resume_finalizes(data)
    t4_all_lrs_dead_parks(data)
    t5_aggregate_without_refs(data)
    t5b_aggregate_with_refs(data)
    t7_recheck_path()
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "device": cuda, "n_checks": len(CHECKS), "n_failed": n_fail,
              "checks": CHECKS, "wall_s": round(time.time() - t0, 1),
              "bindings": launch_bindings(),
              "status": "PASS" if n_fail == 0 else "FAIL"}
    atomic_write_json(report, REPORT)
    print(f"[cuda-tests] status={report['status']} "
          f"({report['n_checks'] - n_fail}/{report['n_checks']} checks, "
          f"{report['wall_s']}s) -> {os.path.relpath(REPORT, HERE)}",
          flush=True)
    if n_fail == 0:
        shutil.rmtree(SB, ignore_errors=True)
    else:
        print(f"[cuda-tests] failures present - scene kept in "
              f"{os.path.relpath(SB, HERE)}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        _child(sys.argv[i + 1], int(sys.argv[i + 2]), sys.argv[i + 3])
        sys.exit(0)
    sys.exit(main())
