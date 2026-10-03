#!/usr/bin/env python3
"""pretrain_audit: the training-gate deliverable (audited BEFORE any
candidate is trained).

Collects every piece of pre-training evidence into one reviewed package:

- environment + git state (tslib commit / violations) live vs frozen protocol
- frozen protocol.json: bindings recomputed against the live tree (must match
  exactly), budget/optim/model pins summarized
- vendor_probe.json: measured model pins (params/tensors/nontrainable/
  buffers, grad-None set after real backward, derived shapes, FFT record)
  on CPU and CUDA
- model_check_report.json: upstream-vs-adapted bitwise equivalence (PASS is
  a hard launch gate)
- cuda_path_tests_report.json: T1-T6 on the real CUDA path
- preflight_report.json: all sections PASS + bindings drift-free + peak
  training VRAM

WRITE ORDER (fixed): audit/pretrain_audit.md is written first, then audit/
pretrain_audit_report.json (atomic), and ONLY THEN the bundle tar is built -
the bundle itself contains the just-written report, so the reviewed artifact
never lacks its own gate verdict.

The bundle TimesNet_TN0_pretrain_audit_with_source_<UTC>.tar.gz contains:
the audit md + report, snapshot/protocol.json + SHA256SUMS, configs/v1.yaml,
audit/baseline.json + vendor_probe.json + model_check_report.json +
cuda_path_tests_report.json + preflight_report.json, the ENTIRE
vendor/TimesNet tree (upstream + adapted + import_diff.patch + UPSTREAM.md +
LICENSE), and every experiment source file (.py/.sh/.md) in the live tree -
so the reviewer sees exactly the code the pins were measured against.
Exit 0 only when every gate is green; run_all.sh refuses to start training
otherwise, and the user review of this bundle gates the full campaign.
"""
import datetime
import json
import os
import sys
import tarfile

from common import (AUDIT, CFG, HERE, SNAPSHOT, SNAPSHOT_PROTOCOL,
                    actual_fingerprints, atomic_write_json, binding_mismatches,
                    environment_snapshot, launch_bindings, launch_gate_drift,
                    tslib_state, vendor_state)

MD_PATH = os.path.join(AUDIT, "pretrain_audit.md")
REPORT_PATH = os.path.join(AUDIT, "pretrain_audit_report.json")

_SRC_EXCLUDE_DIRS = {"snapshot", "checkpoints", "logs", "predictions",
                     "results", "audit", "__pycache__"}


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def source_files():
    """Every live .py/.sh/.md file under the experiment dir, excluding
    runtime/output dirs and the vendor tree (which is archived whole)."""
    out = []
    for root, dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE)
        top = rel_root.split(os.sep)[0]
        if top in _SRC_EXCLUDE_DIRS or top == "vendor":
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".md")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                out.append((os.path.join(root, fn), rel))
    return sorted(out, key=lambda t: t[1])


def vendor_tree_files():
    out = []
    vroot = os.path.join(HERE, "vendor", "TimesNet")
    for root, dirs, files in os.walk(vroot):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            p = os.path.join(root, fn)
            out.append((p, os.path.relpath(p, HERE)))
    return sorted(out, key=lambda t: t[1])


def main():
    problems = []
    rows = []

    def gate(name, ok, detail=""):
        rows.append((name, "PASS" if ok else "FAIL", str(detail)))
        if not ok:
            problems.append(f"{name}: {detail}")

    env = environment_snapshot()
    git = tslib_state()
    proto = load_json(SNAPSHOT_PROTOCOL)
    gate("protocol_exists", proto is not None, SNAPSHOT_PROTOCOL)
    if proto is None:
        print("PRETRAIN_AUDIT FAIL: no frozen protocol - run prepare first",
              file=sys.stderr)
        return 2
    gate("protocol_frozen_bindings",
         not binding_mismatches(proto["bindings"], actual_fingerprints()),
         "snapshot/protocol.json bindings vs live tree")

    probe = load_json(os.path.join(AUDIT, "vendor_probe.json"))
    gate("vendor_probe_present", probe is not None,
         "audit/vendor_probe.json (measured model pins)")
    cuda_probe_ok = bool(probe and probe.get("devices", {}).get("cuda"))
    gate("vendor_probe_cuda", cuda_probe_ok,
         "CUDA probe present (pins measured on the training device)")
    if probe and cuda_probe_ok:
        dev = probe["devices"]["cuda"]
        pins = CFG["model"]["pins"]
        live = vendor_state()
        gate("vendor_probe_matches_pins",
             dev.get("n_params") == pins["params_expected"]
             and dev.get("n_tensors") == pins["tensors_expected"]
             and dev.get("state_dict_entries")
             == pins["state_dict_entries_expected"]
             and dev.get("nontrainable") == pins["nontrainable_expected"],
             f"probe={dev.get('n_params')}/{dev.get('n_tensors')}/"
             f"{dev.get('state_dict_entries')}/{dev.get('nontrainable')} "
             f"pins={pins['params_expected']}/{pins['tensors_expected']}/"
             f"{pins['state_dict_entries_expected']}/"
             f"{pins['nontrainable_expected']}")
        # grad-None set is measured AFTER a real backward (fresh models have
        # all grads None); it must equal the pinned unused-parameter list -
        # distinct from nontrainable, which is empty for TimesNet.
        gate("vendor_probe_grad_none_matches_pins",
             sorted(dev.get("grad_none_names", []))
             == sorted(pins["grad_none_names_expected"])
             and dev.get("grad_bearing")
             == pins["grad_bearing_expected"],
             f"grad_none={dev.get('grad_none_names')} "
             f"grad_bearing={dev.get('grad_bearing')} pins="
             f"{pins['grad_none_names_expected']}/"
             f"{pins['grad_bearing_expected']}")
        gate("vendor_files_md5_live", live == CFG["model"]["vendor"]["files_md5"],
             f"{len(live)} vendored files")

    mcheck = load_json(os.path.join(AUDIT, "model_check_report.json"))
    gate("model_check_pass",
         bool(mcheck) and mcheck.get("status") == "PASS",
         "upstream vs adapted bitwise equivalence (hard launch gate)")

    cudatest = load_json(os.path.join(AUDIT, "cuda_path_tests_report.json"))
    gate("cuda_path_tests_pass",
         bool(cudatest) and cudatest.get("status") == "PASS"
         and cudatest.get("n_failed") == 0,
         f"{(cudatest or {}).get('n_checks')} checks on the real CUDA path")

    pre = load_json(os.path.join(AUDIT, "preflight_report.json"))
    gate("preflight_pass",
         bool(pre) and pre.get("status") == "PASS",
         "audit/preflight_report.json")
    sec_summary = {}
    peak_vram = None
    if pre:
        for k, v in sorted(pre.get("sections", {}).items()):
            sec_summary[k] = v.get("status")
            if v.get("status") not in ("PASS", "SKIP(dev)"):
                problems.append(f"preflight section {k}: {v.get('status')}")
            if k == "I_throughput":
                peak_vram = v.get("peak_vram_mb")
        drift = launch_gate_drift(pre.get("bindings", {}))
        gate("preflight_bindings_live", not drift, f"drift={drift}")
    gate("preflight_sections_all_pass",
         bool(sec_summary) and all(s in ("PASS", "SKIP(dev)")
                                   for s in sec_summary.values()),
         ", ".join(f"{k}={v}" for k, v in sec_summary.items()))

    # every prerequisite report must describe the SAME environment: a
    # stage run outside the pinned CUDA_VISIBLE_DEVICES env records a
    # different device_count in its launch bindings, which launch_gate
    # would refuse at train time - catch it here, at the bundle gate
    prereq = {}
    for nm, rep in (("preflight", pre), ("model_check", mcheck),
                    ("cuda_path_tests", cudatest)):
        if isinstance(rep, dict) and isinstance(rep.get("bindings"), dict):
            prereq[nm] = rep["bindings"]
    prereq["pretrain_audit"] = launch_bindings()
    if len(prereq) == 4:
        ref = prereq["pretrain_audit"]
        off = {}
        for nm, b in prereq.items():
            if nm == "pretrain_audit":
                continue
            d = [k for k in sorted(set(ref) | set(b))
                 if b.get(k) != ref.get(k)]
            if d:
                off[nm] = d
        gate("prereq_bindings_consistent", not off,
             off or "preflight/model_check/cuda_path_tests/pretrain_audit "
                    "bindings agree (snapshot+code+env+cuda+vendor)")
    else:
        gate("prereq_bindings_consistent", False,
             f"bindings blocks found in {sorted(prereq)}")

    status = "PASS" if not problems else "FAIL"
    now = datetime.datetime.now(datetime.timezone.utc)
    now_s = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# TN0 (TimesNet-native) pre-training audit",
        "",
        f"- created_utc: {now_s}",
        f"- overall: **{status}**",
        "",
        "## Environment",
        "",
        f"| item | value |",
        f"| --- | --- |",
        f"| python / numpy | {env['python']} / {env['numpy']} |",
        f"| torch / cuda / cudnn | {env['torch']} / {env['cuda']} / "
        f"{env['cudnn']} |",
        f"| TSLib commit | `{git['commit']}` |",
        f"| TSLib violations | {git['violations'] or 'none'} |",
        f"| vendored TimesNet | upstream "
        f"`{CFG['model']['vendor']['upstream_commit'][:12]}` "
        f"({len(CFG['model']['vendor']['files_md5'])} adapted files "
        f"md5-pinned; adaptation = 2 import lines only) |",
        f"| protocol_version | {CFG['protocol_version']} |",
        f"| evaluation batch | 1 (frozen: FFT top-k period selection is "
        f"batch-dependent; enforced by forward_pred B==1 assertion) |",
        "",
        "## Gates",
        "",
        "| gate | status | detail |",
        "| --- | --- | --- |",
    ]
    for name, st, detail in rows:
        lines.append(f"| {name} | {st} | {detail[:180]} |")
    lines += [
        "",
        "## Preflight sections",
        "",
        "| section | status |",
        "| --- | --- |",
    ]
    for k, v in sorted(sec_summary.items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Peak training VRAM (preflight I_throughput): "
        f"{peak_vram} MiB" if peak_vram is not None else
        "Peak training VRAM: n/a",
        "",
        "## Frozen budget",
        "",
        f"- seeds {CFG['budget']['seeds']}, lrs {CFG['optim']['lrs']}, "
        f"epochs<= {CFG['budget']['epochs']}, patience "
        f"{CFG['budget']['patience']}, batch {CFG['optim']['batch']}",
        f"- candidates 19 domains x {len(CFG['optim']['lrs'])} lr x "
        f"{len(CFG['budget']['seeds'])} seeds = "
        f"{19 * len(CFG['optim']['lrs']) * len(CFG['budget']['seeds'])}; "
        f"sum(U_d) = {proto['budget']['sum_U_d']}",
        "",
        "## Pre-training test-split access disclosure",
        "",
        "- INCIDENT (disclosed, remediated): pretrain bundle "
        "`TimesNet_TN0_pretrain_audit_with_source_20261003T102124Z.tar.gz` "
        "ran cuda_path_tests T7 over REAL test rows - 3 arts runs x 151 "
        "real test windows inferred and metric-recomputed before any "
        "training. T7 now runs on an isolated synthetic fixture "
        "(recheck.DenseData replaced; zero real test access asserted by "
        "spies). No frozen configuration value was derived from or adjusted "
        "because of that access.",
        "- REMAINING pre-training test touches: preflight section G "
        "(poison-wiring proof, inherited unchanged from the frozen P0/T0 "
        "lineage): test targets are poisoned to NaN in a sandbox and "
        "evaluate_split must abort with DivergenceError - model forwards "
        "run over real test inputs, but no metric is derived and the eval "
        "aborts at the NaN gate; training-time test isolation is proven by "
        "the same section (poisoned test rows leave val bitwise unchanged).",
        "",
        "Verdict: " + ("all gates green - bundle released for user review; "
                       "training starts only after approval."
                       if status == "PASS" else
                       "FAILURES PRESENT - training is refused."),
        "",
    ]
    with open(MD_PATH + ".tmp", "w") as f:
        f.write("\n".join(lines))
    os.replace(MD_PATH + ".tmp", MD_PATH)

    # REPORT BEFORE TAR: the gate verdict is on disk before any packaging,
    # and the tar includes this exact report.
    bundle = os.path.join(
        AUDIT, f"TimesNet_TN0_pretrain_audit_with_source_"
               f"{now.strftime('%Y%m%dT%H%M%SZ')}.tar.gz")
    atomic_write_json({"created_utc": now_s,
                       "status": status, "gates": [
                           {"name": n, "status": s, "detail": d}
                           for n, s, d in rows],
                       "preflight_sections": sec_summary,
                       "peak_vram_mb": peak_vram,
                       "bundle": os.path.basename(bundle),
                       "bindings": launch_bindings()},
                      REPORT_PATH)

    entries = [(MD_PATH, "pretrain_audit.md"), (REPORT_PATH,
                                                "audit/pretrain_audit_report.json")]
    for src, arc in (
            (SNAPSHOT_PROTOCOL, "snapshot/protocol.json"),
            (os.path.join(SNAPSHOT, "SHA256SUMS.txt"),
             "snapshot/SHA256SUMS.txt"),
            (os.path.join(HERE, "configs", "v1.yaml"),
             "configs/v1.yaml"),
            (os.path.join(AUDIT, "baseline.json"),
             "audit/baseline.json"),
            (os.path.join(AUDIT, "vendor_probe.json"),
             "audit/vendor_probe.json"),
            (os.path.join(AUDIT, "model_check_report.json"),
             "audit/model_check_report.json"),
            (os.path.join(AUDIT, "cuda_path_tests_report.json"),
             "audit/cuda_path_tests_report.json"),
            (os.path.join(AUDIT, "preflight_report.json"),
             "audit/preflight_report.json")):
        if os.path.isfile(src):
            entries.append((src, arc))
    entries.extend(source_files())
    entries.extend(vendor_tree_files())
    seen = set()
    with tarfile.open(bundle, "w:gz") as tf:
        for src, arc in entries:
            if arc in seen or not os.path.isfile(src):
                continue
            seen.add(arc)
            tf.add(src, arcname=arc)
    print(f"[pretrain-audit] status={status} -> audit/pretrain_audit.md")
    print(f"[pretrain-audit] bundle ({len(seen)} files): {bundle}")
    for p in problems:
        print(f"  PROBLEM: {p}", file=sys.stderr)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
