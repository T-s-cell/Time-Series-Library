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
  T5  LN results missing -> aggregate degrades to T1-only
      (comparison_status=ln_results_unavailable) instead of crashing
  T6  launch gate: missing / FAIL / stale / tampered preflight evidence
      refuses to start the stage

All artifacts stay under audit/cuda_tmp/ (wiped on success, kept on failure
for scene inspection). Queue/aggregate outputs are redirected into the
sandbox; the frozen snapshot is only read. Divergence is injected by wrapping
trainer.forward_pred (the real forward/backward path runs until the injection
point) - no production code is modified for testing.
"""
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
import trainer
from common import (CFG, DivergenceError, LRS, SEEDS, atomic_write_json,
                    cuda_state, launch_bindings, launch_gate)
from data import DenseData

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
    rep = {"status": "PASS",
           "created_utc": datetime.datetime.now(datetime.timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sections": {"I_throughput": {"peak_vram_mb": 200.0}},
           "bindings": launch_bindings()}
    p.set(common, "PREFLIGHT_JSON", os.path.join(root, "gate_report.json"))
    atomic_write_json(rep, common.PREFLIGHT_JSON)


def force_diverges(p):
    """Every run's 3rd forward call raises DivergenceError (counter resets on
    each raise so each candidate diverges exactly once, in its epoch 1)."""
    state = {"n": 0}
    real = trainer.forward_pred

    def forced(model, xt):
        state["n"] += 1
        if state["n"] == 3:
            state["n"] = 0
            raise DivergenceError("forced divergence (cuda_path_tests)")
        return real(model, xt)

    p.set(trainer, "forward_pred", forced)


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
        deadline = time.time() + 180
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


# ---------------------------------------------------------------- T5
def t5_aggregate_without_ln(data):
    print("[T5] missing LN/T1 results -> T0-only aggregate, no crash",
          flush=True)
    p = Patch()
    old_argv = sys.argv
    try:
        root = sandbox("t5")
        res = os.path.join(root, "results")
        pred = os.path.join(root, "predictions")
        sel_domains = {}
        for dom in sorted(data.domain_vars):
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
        p.set(aggregate, "RESULTS", res)
        p.set(aggregate, "PREDICTIONS", pred)
        p.set(aggregate, "locate_ln_dir", lambda v: None)
        sys.argv = ["aggregate.py"]
        rc = aggregate.main()
        check("t5_rc0", rc == 0, rc)
        summ = load_json(os.path.join(res, "aggregate_summary.json"))
        check("t5_status_degraded",
              summ["comparison_status"] == "ln_results_unavailable", summ)
        check("t5_t1_status_degraded",
              summ["t1_comparison_status"] == "t1_results_unavailable", summ)
        check("t5_190_vars", summ["n_vars"] == 190, summ["n_vars"])
        check("t5_methods_t0_only",
              set(summ["methods_overall"]) == {"T0"}, summ["methods_overall"])
        with open(os.path.join(res, "overall.csv")) as f:
            rows = [ln.split(",")[0] for ln in f.read().splitlines()[1:]]
        check("t5_overall_rows_t0_only", rows == ["T0"], rows)
    except Exception:  # noqa: BLE001
        check("t5_uncaught", False, traceback.format_exc()[-400:])
    finally:
        sys.argv = old_argv
        p.restore()


# ---------------------------------------------------------------- T6
def t6_launch_gate():
    print("[T6] launch gate refuses missing/FAIL/stale/tampered evidence",
          flush=True)
    p = Patch()
    try:
        gp = os.path.join(SB, "gate_t6.json")
        p.set(common, "PREFLIGHT_JSON", gp)

        def exits(fn):
            try:
                fn()
                return None
            except SystemExit as e:
                return e.code

        if os.path.isfile(gp):
            os.remove(gp)
        check("t6_missing_refused", exits(lambda: launch_gate("train")) == 2,
              "missing report")
        atomic_write_json({"status": "FAIL", "bindings": launch_bindings()},
                          gp)
        check("t6_fail_refused", exits(lambda: launch_gate("train")) == 2,
              "status FAIL")
        atomic_write_json({"status": "PASS"}, gp)
        check("t6_stale_refused", exits(lambda: launch_gate("freeze")) == 2,
              "no bindings block")
        b = launch_bindings()
        b["code"] = dict(b["code"])
        b["code"]["trainer.py"] = "0" * 32
        atomic_write_json({"status": "PASS", "bindings": b}, gp)
        check("t6_tampered_refused", exits(lambda: launch_gate("test")) == 2,
              "code md5 tampered")
        atomic_write_json({"status": "PASS", "bindings": launch_bindings()},
                          gp)
        rep = None
        try:
            rep = launch_gate("train")
        except SystemExit as e:
            check("t6_clean_accepts", False, f"unexpected exit {e.code}")
        check("t6_clean_accepts", isinstance(rep, dict)
              and "bindings" in rep, "gate returns report")
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
    t5_aggregate_without_ln(data)
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "device": cuda, "n_checks": len(CHECKS), "n_failed": n_fail,
              "checks": CHECKS, "wall_s": round(time.time() - t0, 1),
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
