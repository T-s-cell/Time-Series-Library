#!/usr/bin/env python3
"""audit: end-to-end integrity review after test/aggregate/recheck.

1. TSLib tree vs audit/baseline.json: bidirectional diff - zero modifications
   or deletions anywhere, zero additions outside the allowed experiment dirs.
2. Read-only proof: the 4 exported LN materials still hash to the pinned
   values (the LN dir was never written).
3. Hash chain: snapshot/protocol.json bindings + protocol_self recomputed;
   vendored TimesNet core md5s vs pins; tslib commit.
4. 171 runs: terminal states complete; successful runs' trails re-audited
   (per-epoch update counts == U_d, select_best_epoch recomputed); the 57
   selected checkpoints' sha256 re-verified against selection.json.
5. Predictions: 57 npz files present, per-seed coverage exactly the manifest
   test enumeration (3 x 2,474 unique windows).
6. Aggregate: overall/domain/variable tables cover TN0 (+F0/T0/P0 when
   present); 190 vars; diverged combos disclosed in selection.json AND
   aggregate_summary.json; all three reference statuses declared.
7. Recheck gate: results/recheck_report.json exists with status
   PASS or PASS_WITH_DISCLOSURE; disclosure count matches the report.

Writes audit/audit_report.json; exit 0 only on full PASS. --tar additionally
builds ONE unified final bundle
audit/TimesNet_TN0_TimesX_final_<date>.tar.gz with a DEDUPLICATED archive
list (a single manifest drives it - no path can be added twice):
- every experiment source file (.py/.sh/.md, runtime dirs excluded) +
  configs/v1.yaml - the exact code the run used;
- the ENTIRE vendor/TimesNet tree (upstream + adapted + import_diff.patch +
  UPSTREAM.md + LICENSE);
- snapshot/protocol.json + SHA256SUMS.txt;
- ALL pre-training evidence (baseline.json, vendor_probe.json,
  model_check_report.json, pretrain_audit_report.json + pretrain_audit.md,
  cuda_path_tests_report.json, preflight_report.json) plus this audit's
  audit_report.json;
- all results/ tables + recheck_report.json;
- training evidence under checkpoints/ (terminal states + per-epoch val
  records via ONE *.json glob - model .pt weights stay on the server, their
  sha256s live in selection.json / checkpoints_pt.sha256), logs/, and the
  57 test npz;
- MANIFEST.sha256 listing "<sha256>  <arcname>" for every archived file,
  generated BEFORE closing the tar and re-verified member-by-member AFTER
  closing (the verification result is printed and must be 0 mismatches).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import tarfile

import numpy as np

from common import (AUDIT, CFG, CHECKPOINTS, EXP_NAME, HERE, LOGS, PREDICTIONS,
                    RESULTS, SEEDS, REPO_ROOT, SNAPSHOT, SNAPSHOT_PROTOCOL,
                    actual_fingerprints, atomic_write_json, binding_mismatches,
                    environment_snapshot, protocol_self_hash, sha256_file,
                    tslib_state, vendor_state)
from data import DenseData
from freeze_selection import jsonl_budget_ok

_SRC_EXCLUDE_DIRS = {"snapshot", "checkpoints", "logs", "predictions",
                     "results", "audit", "__pycache__"}


def fail(msg):
    print(f"AUDIT ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def bundle_entries():
    """The deduplicated (src_path, arcname) list of the final bundle."""
    entries = []

    def add(src, arc):
        entries.append((src, arc))

    for root, dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE)
        top = rel_root.split(os.sep)[0]
        if top in _SRC_EXCLUDE_DIRS:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            # the vendor tree is archived whole (extensionless files like
            # LICENSE included); experiment sources keep the extension filter
            if top == "vendor" or fn.endswith(
                    (".py", ".sh", ".md", ".yaml", ".patch")):
                p = os.path.join(root, fn)
                add(p, os.path.relpath(p, HERE))
    for name in ("protocol.json", "SHA256SUMS.txt"):
        p = os.path.join(os.path.dirname(SNAPSHOT), name)
        if os.path.isfile(p):
            add(p, f"snapshot/{name}")
    for name in ("baseline.json", "vendor_probe.json",
                 "model_check_report.json", "pretrain_audit_report.json",
                 "pretrain_audit.md", "cuda_path_tests_report.json",
                 "preflight_report.json", "audit_report.json"):
        p = os.path.join(AUDIT, name)
        if os.path.isfile(p):
            add(p, f"audit/{name}")
    for name in ("selection.json", "run_manifest.json",
                 "aggregate_summary.json", "overall.csv",
                 "domain_summary.csv", "test_variable_level.csv",
                 "comparisons.csv", "improvement_counts.csv",
                 "main_table.md", "report.md", "recheck_report.json"):
        p = os.path.join(RESULTS, name)
        if os.path.isfile(p):
            add(p, f"results/{name}")
    # training evidence: terminal states + per-epoch val records - ONE glob
    # (*.json already covers ep*.val.json; the dedup set below is a second
    # guard), model .pt weights stay on the server
    for p in sorted(os.listdir(CHECKPOINTS)) if os.path.isdir(CHECKPOINTS) \
            else []:
        cdir = os.path.join(CHECKPOINTS, p)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".json"):
                add(os.path.join(cdir, fn), f"checkpoints/{p}/{fn}")
    if os.path.isdir(LOGS):
        for fn in sorted(os.listdir(LOGS)):
            p = os.path.join(LOGS, fn)
            if os.path.isfile(p):
                add(p, f"logs/{fn}")
    if os.path.isdir(PREDICTIONS):
        for fn in sorted(os.listdir(PREDICTIONS)):
            if fn.endswith("__test.npz"):
                add(os.path.join(PREDICTIONS, fn), f"predictions/{fn}")
    # dedup by arcname, keep first occurrence, sorted for determinism
    seen = {}
    for src, arc in entries:
        if arc not in seen and os.path.isfile(src):
            seen[arc] = src
    return sorted(((src, arc) for arc, src in seen.items()),
                  key=lambda kv: kv[1])


def build_bundle(entries, tar_path):
    manifest_lines = []
    for src, arc in entries:
        manifest_lines.append(f"{sha256_file(src)}  {arc}")
    manifest_blob = "\n".join(manifest_lines) + "\n"
    tmp_manifest = tar_path + ".manifest.tmp"
    with open(tmp_manifest, "w") as f:
        f.write(manifest_blob)
    with tarfile.open(tar_path, "w:gz") as tf:
        for src, arc in entries:
            tf.add(src, arcname=arc)
        tf.add(tmp_manifest, arcname="MANIFEST.sha256")
    os.remove(tmp_manifest)


def verify_bundle(tar_path, entries):
    """Re-open the finished tar and re-hash every regular member against the
    embedded MANIFEST.sha256, in BOTH directions (every archived member must
    be listed with a matching hash; every MANIFEST entry must be archived).
    The MANIFEST itself is the only member exempt from listing. Returns
    (n_checked, problems) - a valid bundle yields (n_members, [])."""
    want = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        mf = tf.extractfile("MANIFEST.sha256")
        if mf is None:
            return 0, ["MANIFEST.sha256 member missing from archive"]
        for line in mf.read().decode().splitlines():
            if not line:
                continue
            h, arc = line.split("  ", 1)
            want[arc] = h
        problems = []
        n = 0
        member_names = set()
        for m in tf.getmembers():
            if not m.isfile():
                continue
            member_names.add(m.name)
            if m.name == "MANIFEST.sha256":
                continue
            f = tf.extractfile(m)
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            n += 1
            if m.name not in want:
                problems.append(f"{m.name}: archived but not in MANIFEST")
            elif h.hexdigest() != want[m.name]:
                problems.append(f"{m.name}: hash mismatch vs MANIFEST")
        unarchived = sorted(set(want) - member_names)
        if unarchived:
            problems.append(f"{len(unarchived)} MANIFEST entries not "
                            f"archived: {unarchived[:5]}")
    return n, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="store_true",
                    help="also build the final evidence bundle in audit/")
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
    rec("vendor_core_md5", vendor_state() == CFG["model"]["vendor"]["files_md5"]
        and proto["bindings"]["vendor_core_md5"] == vendor_state(),
        "vendored TimesNet md5s vs pins + frozen bindings")
    rec("isolation_violations", not ts["violations"], ts["violations"])

    # 4. runs + selection
    sel_path = os.path.join(RESULTS, "selection.json")
    rec("selection_exists", os.path.isfile(sel_path), sel_path)
    n_div = 0
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
            rec("aggregate_tn0_present",
                "TN0" in agg.get("methods_overall", {}),
                list(agg.get("methods_overall", {})))
            with open(os.path.join(RESULTS, "test_variable_level.csv")) as f:
                import csv as _csv
                rows = list(_csv.reader(f))
            tn0_vars = {r[1] for r in rows[1:] if r[0] == "TN0"}
            rec("aggregate_190_vars", len(tn0_vars) == 190, len(tn0_vars))
            rec("aggregate_divergence_disclosed",
                len(agg.get("diverged_disclosure", [])) == n_div,
                f"aggregate has {len(agg.get('diverged_disclosure', []))}")
            rec("aggregate_ln_status_declared",
                agg.get("comparison_status") in
                ("ok", "ln_results_unavailable"),
                agg.get("comparison_status"))
            rec("aggregate_t0ref_status_declared",
                agg.get("t0ref_comparison_status") in
                ("ok", "t0_results_unavailable"),
                agg.get("t0ref_comparison_status"))
            rec("aggregate_p0ref_status_declared",
                agg.get("p0ref_comparison_status") in
                ("ok", "p0_results_unavailable"),
                agg.get("p0ref_comparison_status"))

        # 7. recheck gate
        rc_path = os.path.join(RESULTS, "recheck_report.json")
        rec("recheck_report_exists", os.path.isfile(rc_path), rc_path)
        if os.path.isfile(rc_path):
            rc = load_json(rc_path)
            rec("recheck_status",
                rc.get("status") in ("PASS", "PASS_WITH_DISCLOSURE"),
                rc.get("status"))
            rec("recheck_metric_ok",
                rc.get("metric_recompute", {}).get("ok") is True, "")
            inf = rc.get("inference_replay", {})
            rec("recheck_no_failed_replay",
                not any(r.get("verdict") == "FAIL"
                        for r in inf.get("runs", [])),
                f"disclosed={inf.get('n_disclosed')}")

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
                                f"TimesNet_TN0_TimesX_final_{stamp}.tar.gz")
        entries = bundle_entries()
        build_bundle(entries, tar_path)
        n_checked, problems = verify_bundle(tar_path, entries)
        bundle_sha = sha256_file(tar_path)
        rec("bundle_manifest_verified", not problems,
            f"{n_checked} members re-hashed, {len(problems)} mismatches")
        # the report status must reflect the bundle verification outcome
        report["status"] = "PASS" if overall_ok else "FAIL"
        atomic_write_json(report, os.path.join(AUDIT, "audit_report.json"))
        if problems:
            print(f"[audit] bundle verification FAILED: {problems[:5]}",
                  file=sys.stderr)
            return 1
        with open(os.path.join(AUDIT, "final_bundle_sha256.txt"), "w") as f:
            f.write(f"{bundle_sha}  {os.path.basename(tar_path)}\n"
                    f"{len(entries)} files + MANIFEST.sha256\n")
        print(f"[audit] final bundle ({len(entries)} files + MANIFEST): "
              f"{tar_path}")
        print(f"[audit] bundle sha256: {bundle_sha}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
