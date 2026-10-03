#!/usr/bin/env python3
"""pretrain_audit: the training-gate deliverable (audited BEFORE any
candidate is trained).

Collects every piece of pre-training evidence into one reviewed package:

- environment + git state (tslib commit / violations) live vs frozen protocol
- frozen protocol.json: bindings recomputed against the live tree (must match
  exactly), budget/optim/model pins summarized
- vendor_probe.json: measured model pins (params/tensors/nontrainable/buffers,
  RevIN eps, patch geometry) on CPU and CUDA
- vendor_check_report.json: upstream-vs-adapted bitwise equivalence (PASS is
  a hard launch gate)
- cuda_path_tests_report.json: T1-T6 on the real CUDA path
- preflight_report.json: all A-M sections PASS + bindings drift-free + peak
  training VRAM

Writes audit/pretrain_audit.md (human-readable) and the bundle
audit/pretrain_audit_bundle_<ts>.tar.gz (protocol + configs + vendor
manifests + all four reports + this audit). Exit 0 only when every gate is
green; run_all.sh refuses to start training otherwise.
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


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


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
             and dev.get("nontrainable") == pins["nontrainable_expected"]
             and sorted(dev.get("grad_none_names", []))
             == sorted(pins["nontrainable_expected"]),
             f"probe={dev.get('n_params')}/{dev.get('n_tensors')}/"
             f"{dev.get('nontrainable')} pins={pins['params_expected']}/"
             f"{pins['tensors_expected']}/{pins['nontrainable_expected']}")
        gate("vendor_files_md5_live", live == CFG["model"]["vendor"]["files_md5"],
             f"{len(live)} vendored files")

    vcheck = load_json(os.path.join(AUDIT, "vendor_check_report.json"))
    gate("vendor_check_pass",
         bool(vcheck) and vcheck.get("status") == "PASS",
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

    status = "PASS" if not problems else "FAIL"
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [
        "# P0 (PatchTST-native) pre-training audit",
        "",
        f"- created_utc: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
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
        f"| vendored PatchTST | upstream "
        f"`{CFG['model']['vendor']['upstream_commit'][:12]}` "
        f"({len(CFG['model']['vendor']['files_md5'])} adapted files "
        f"md5-pinned) |",
        f"| protocol_version | {CFG['protocol_version']} |",
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
        "Verdict: " + ("all gates green - training may start."
                       if status == "PASS" else
                       "FAILURES PRESENT - training is refused."),
        "",
    ]
    with open(MD_PATH + ".tmp", "w") as f:
        f.write("\n".join(lines))
    os.replace(MD_PATH + ".tmp", MD_PATH)

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    bundle = os.path.join(AUDIT, f"pretrain_audit_bundle_{stamp}.tar.gz")
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(MD_PATH, arcname="pretrain_audit.md")
        for src, arc in (
                (SNAPSHOT_PROTOCOL, "snapshot/protocol.json"),
                (os.path.join(SNAPSHOT, "SHA256SUMS.txt"),
                 "snapshot/SHA256SUMS.txt"),
                (os.path.join(HERE, "configs", "v1.yaml"),
                 "configs/v1.yaml"),
                (os.path.join(HERE, "vendor", "PatchTST", "UPSTREAM.md"),
                 "vendor/PatchTST/UPSTREAM.md"),
                (os.path.join(HERE, "vendor", "PatchTST",
                              "import_diff.patch"),
                 "vendor/PatchTST/import_diff.patch"),
                (os.path.join(AUDIT, "vendor_probe.json"),
                 "audit/vendor_probe.json"),
                (os.path.join(AUDIT, "vendor_check_report.json"),
                 "audit/vendor_check_report.json"),
                (os.path.join(AUDIT, "cuda_path_tests_report.json"),
                 "audit/cuda_path_tests_report.json"),
                (os.path.join(AUDIT, "preflight_report.json"),
                 "audit/preflight_report.json"),
                (os.path.join(AUDIT, "baseline.json"),
                 "audit/baseline.json")):
            if os.path.isfile(src):
                tf.add(src, arcname=arc)
    atomic_write_json({"created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "status": status, "gates": [
                           {"name": n, "status": s, "detail": d}
                           for n, s, d in rows],
                       "preflight_sections": sec_summary,
                       "peak_vram_mb": peak_vram,
                       "bundle": os.path.basename(bundle),
                       "bindings": launch_bindings()},
                      os.path.join(AUDIT, "pretrain_audit_report.json"))
    print(f"[pretrain-audit] status={status} -> audit/pretrain_audit.md")
    print(f"[pretrain-audit] bundle: {bundle}")
    for p in problems:
        print(f"  PROBLEM: {p}", file=sys.stderr)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
