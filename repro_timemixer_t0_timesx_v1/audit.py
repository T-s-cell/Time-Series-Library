#!/usr/bin/env python3
"""audit: end-to-end integrity review after test/aggregate.

1. TSLib tree vs audit/baseline.json: bidirectional diff - zero modifications
   or deletions anywhere, zero additions outside this experiment dir.
2. Read-only proof: the 4 exported LN materials still hash to the pinned
   values (the LN dir was never written).
3. Hash chain: snapshot/protocol.json bindings + protocol_self recomputed;
   core md5s; tslib commit.
4. 171 runs: terminal states complete; successful runs' trails re-audited
   (per-epoch update counts == U_d, select_best_epoch recomputed); the 57
   selected checkpoints' sha256 re-verified against selection.json.
5. Predictions: 57 npz files present, per-seed coverage exactly the manifest
   test enumeration (3 x 2,474 unique windows).
6. Aggregate: overall/domain/variable tables cover T0 (+LN methods and T1
   when present); 190 vars; diverged combos disclosed in selection.json AND
   aggregate_summary.json.
Writes audit/audit_report.json; exit 0 only on full PASS. --tar additionally
builds the evidence bundle audit/TimeMixer_T0_TimesX_audit_<date>.tar.gz
(training jsonl, per-epoch val records, stage logs, terminal states, 57 test
npz; model .pt checkpoints stay on the server).
"""
import argparse
import datetime
import glob
import json
import os
import sys
import tarfile

import numpy as np

from common import (AUDIT, CFG, CHECKPOINTS, EXP_NAME, HERE, LOGS, PREDICTIONS,
                    RESULTS, SEEDS, REPO_ROOT, SNAPSHOT, SNAPSHOT_PROTOCOL,
                    actual_fingerprints, atomic_write_json, binding_mismatches,
                    environment_snapshot, protocol_self_hash, sha256_file,
                    tslib_state)
from data import DenseData
from freeze_selection import jsonl_budget_ok


def fail(msg):
    print(f"AUDIT ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="store_true",
                    help="also build the evidence bundle in audit/")
    args = ap.parse_args()
    checks = {}
    overall_ok = True

    def rec(name, ok, detail=""):
        nonlocal overall_ok
        checks[name] = {"ok": bool(ok), "detail": str(detail)[:500]}
        if not ok:
            overall_ok = False
            print(f"  [audit] FAIL {name}: {str(detail)[:250]}", flush=True)

    # 1. TSLib tree vs baseline
    baseline = load_json(os.path.join(AUDIT, "baseline.json"))
    base = baseline["files"]
    now = {}
    here_name = EXP_NAME
    for root, dirs, fns in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT)
        if rel_root == here_name or rel_root.startswith(here_name + os.sep):
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs
                         if d not in (".git", "__pycache__"))
        for fn in sorted(fns):
            now[os.path.relpath(os.path.join(root, fn), REPO_ROOT)] = \
                sha256_file(os.path.join(root, fn))
    modified = sorted(k for k in now if k in base and now[k] != base[k])
    deleted = sorted(k for k in base if k not in now)
    added_outside = sorted(k for k in now if k not in base
                           and not (k == here_name
                                    or k.startswith(here_name + os.sep)))
    rec("tslib_zero_drift", not modified and not deleted and not added_outside,
        f"mod={modified[:5]} del={deleted[:5]} add_outside={added_outside[:5]}")

    # 2. LN materials untouched
    pins = CFG["source_snapshot"]
    pairs = [("split_manifest.json", "ln_split_manifest_sha256"),
             ("data_cache.npz", "ln_data_cache_sha256"),
             ("protocol.json", "ln_protocol_sha256"),
             ("split_counts.csv", "ln_split_counts_sha256")]
    drift = [fn for fn, key in pairs
             if sha256_file(os.path.join(SNAPSHOT, fn)) != pins[key]]
    rec("ln_materials_unchanged", not drift, drift)

    # 3. hash chain
    proto = load_json(SNAPSHOT_PROTOCOL)
    rec("bindings_recompute", not binding_mismatches(proto["bindings"],
                                                     actual_fingerprints()),
        "frozen protocol vs live")
    rec("protocol_self", proto["bindings"]["protocol_self"]
        == protocol_self_hash(), "two-phase projection hash")
    ts = tslib_state()
    rec("tslib_commit", ts["commit"] == proto["bindings"]["tslib_commit"],
        ts["commit"])
    rec("core_md5", ts["core_md5"] == CFG["model"]["core_md5"], "")
    rec("isolation_violations", not ts["violations"], ts["violations"])

    # 4. runs + selection
    sel_path = os.path.join(RESULTS, "selection.json")
    rec("selection_exists", os.path.isfile(sel_path), sel_path)
    if os.path.isfile(sel_path):
        sel = load_json(sel_path)
        data = DenseData()
        n_terminal = 0
        n_div = 0
        trail_bad = []
        for row in CFG["budget"]["expected_domain_table"]:
            dom = row["domain"]
            u_d = -(-row["dense"] // CFG["optim"]["batch"])
            for lr in CFG["optim"]["lrs"]:
                for seed in SEEDS:
                    rid = f"{dom}__lr{lr:g}__s{seed}"
                    ck = os.path.join(CHECKPOINTS, rid)
                    ok = False
                    if os.path.isfile(os.path.join(ck, "DIVERGED.json")):
                        n_terminal += 1
                        n_div += 1
                        ok = True
                    elif os.path.isfile(os.path.join(ck, "SUCCESS.json")):
                        n_terminal += 1
                        ok, _ = jsonl_budget_ok(rid, u_d)
                        if not ok:
                            trail_bad.append(rid)
                    if not ok and not os.path.isdir(ck):
                        trail_bad.append(f"{rid}(missing)")
        rec("terminal_171", n_terminal == 171, f"{n_terminal}/171")
        rec("trails_audited", not trail_bad, trail_bad[:8])
        ck_bad = []
        for dom, block in sel["domains"].items():
            for seed_s, ent in block["seeds"].items():
                ck = os.path.join(CHECKPOINTS, ent["run_id"],
                                  os.path.basename(ent["ckpt"]))
                if not os.path.isfile(ck) or sha256_file(ck) != \
                        ent["ckpt_sha256"]:
                    ck_bad.append(ent["run_id"])
        rec("selected_ckpts_57", len(sel["domains"]) == 19 and not ck_bad,
            f"domains={len(sel['domains'])} bad={ck_bad[:5]}")
        rec("divergence_disclosed",
            len(sel.get("diverged_disclosure", [])) == n_div,
            f"selection.json has {len(sel.get('diverged_disclosure', []))}, "
            f"checkpoints show {n_div}")

        # 5. predictions coverage
        cov_bad = []
        for seed in SEEDS:
            wins = []
            for dom, block in sel["domains"].items():
                rid = block["seeds"][str(seed)]["run_id"]
                p = os.path.join(PREDICTIONS, f"{rid}__test.npz")
                if not os.path.isfile(p):
                    cov_bad.append(f"{rid}: no npz")
                    continue
                with np.load(p, allow_pickle=False) as z:
                    wins.extend(zip((str(x) for x in z["var_key"]),
                                    (str(x) for x in z["sample_id"])))
            manifest = set()
            for vk in data.manifest["variables"]:
                for sid in data.native_ids(vk, "test"):
                    manifest.add((vk, sid))
            if len(wins) != CFG["test"]["per_seed_coverage"] \
                    or set(wins) != manifest:
                cov_bad.append(f"seed {seed}: {len(wins)} windows / "
                               f"{len(set(wins))} unique vs manifest "
                               f"{len(manifest)}")
        rec("test_coverage_3x2474", not cov_bad, cov_bad[:4])

        # 6. aggregate
        agg_path = os.path.join(RESULTS, "aggregate_summary.json")
        rec("aggregate_exists", os.path.isfile(agg_path), agg_path)
        if os.path.isfile(agg_path):
            agg = load_json(agg_path)
            rec("aggregate_t0_present", "T0" in agg.get("methods_overall", {}),
                list(agg.get("methods_overall", {})))
            with open(os.path.join(RESULTS, "test_variable_level.csv")) as f:
                import csv as _csv
                rows = list(_csv.reader(f))
            t0_vars = {r[1] for r in rows[1:] if r[0] == "T0"}
            rec("aggregate_190_vars", len(t0_vars) == 190, len(t0_vars))
            rec("aggregate_divergence_disclosed",
                len(agg.get("diverged_disclosure", [])) == n_div,
                f"aggregate has {len(agg.get('diverged_disclosure', []))}")
            rec("aggregate_ln_status_declared",
                agg.get("comparison_status") in
                ("ok", "ln_results_unavailable"),
                agg.get("comparison_status"))
            rec("aggregate_t1_status_declared",
                agg.get("t1_comparison_status") in
                ("ok", "t1_results_unavailable"),
                agg.get("t1_comparison_status"))

    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "environment": environment_snapshot(),
              "checks": checks,
              "status": "PASS" if overall_ok else "FAIL"}
    atomic_write_json(report, os.path.join(AUDIT, "audit_report.json"))
    print(f"[audit] status={report['status']} -> audit/audit_report.json",
          flush=True)
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v['ok'] else 'FAIL'} {v['detail'][:80]}",
              flush=True)

    if args.tar and report["status"] == "PASS":
        stamp = datetime.date.today().strftime("%Y%m%d")
        tar_path = os.path.join(AUDIT,
                                f"TimeMixer_T0_TimesX_audit_{stamp}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            for name in ("preflight_report.json", "audit_report.json",
                         "baseline.json", "cuda_path_tests_report.json"):
                p = os.path.join(AUDIT, name)
                if os.path.isfile(p):
                    tf.add(p, arcname=f"audit/{name}")
            for name in ("protocol.json", "SHA256SUMS.txt"):
                p = os.path.join(os.path.dirname(SNAPSHOT), name)
                if os.path.isfile(p):
                    tf.add(p, arcname=f"snapshot/{name}")
            for name in ("selection.json", "run_manifest.json",
                         "aggregate_summary.json", "overall.csv",
                         "domain_summary.csv", "test_variable_level.csv",
                         "comparisons.csv", "improvement_counts.csv"):
                p = os.path.join(RESULTS, name)
                if os.path.isfile(p):
                    tf.add(p, arcname=f"results/{name}")
            # training evidence: per-run trails + terminal states + per-epoch
            # val records (model .pt weights stay on the server)
            for p in sorted(glob.glob(os.path.join(CHECKPOINTS, "*",
                                                   "*.json"))):
                arc = os.path.relpath(p, HERE)
                tf.add(p, arcname=arc)
            for p in sorted(glob.glob(os.path.join(CHECKPOINTS, "*",
                                                   "ep*.val.json"))):
                arc = os.path.relpath(p, HERE)
                tf.add(p, arcname=arc)
            # queue events + per-run jsonl trails + stage logs
            for p in sorted(glob.glob(os.path.join(LOGS, "*"))):
                if os.path.isfile(p):
                    tf.add(p, arcname=f"logs/{os.path.basename(p)}")
            # 57 test prediction arrays
            for p in sorted(glob.glob(os.path.join(PREDICTIONS,
                                                   "*__test.npz"))):
                tf.add(p, arcname=f"predictions/{os.path.basename(p)}")
            print(f"[audit] evidence bundle: {tar_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
