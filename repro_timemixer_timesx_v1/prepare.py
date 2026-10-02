#!/usr/bin/env python3
"""prepare: read-only export of the LN frozen materials + frozen projection.

1. Disk gate (>= 20 GiB free next to the snapshot).
2. Locate the LN run dir (eta default, else local fallback); NEVER write there.
3. Verify the 4 LN files by file-level sha256 against the pins in
   configs/v1.yaml; ABORT on any mismatch.
4. Verify the LN protocol.json internals: totals, and a RECOMPUTED md5 of
   every file in its code_fingerprint (LN static code must be unchanged since
   ITS freeze; running logs/checkpoints are excluded).
5. Content-level verification: split_manifest hash excluding zip_path_used,
   and per-array sha256(tobytes()) of data_cache.npz.
6. Copy the 4 files read-only into snapshot/manifests/ + SHA256SUMS.txt
   (idempotent: existing identical files are kept, differing files abort).
7. Validate the exported snapshot content: totals, 190 vars / 19 domains,
   dense_count recompute (n_train_obs - 107), window bounds, fallback_std,
   finiteness, sum(U_d) == 1157.
8. Record TSLib state (commit, violations, 4 core md5s) and this experiment's
   code fingerprint; two-phase write of snapshot/protocol.json whose bindings
   pin all of the above.
9. audit/baseline.json: sha256 of every static file in the TSLib tree outside
   .git/ and this experiment dir (audit later diffs against it).
"""
import datetime
import hashlib
import json
import os
import shutil
import sys

import numpy as np

from common import (BASELINE_JSON, CFG, HERE, REPO_ROOT, SNAPSHOT,
                    SNAPSHOT_DIR, SNAPSHOT_PROTOCOL, actual_fingerprints,
                    atomic_write_json, md5_file, protocol_self_hash,
                    sha256_file, tslib_state, u_d_s_d)  # noqa: F401 (paths kept for clarity)
from data import DenseData, SnapshotError

LN_FILES = {"split_manifest.json": "ln_split_manifest_sha256",
            "data_cache.npz": "ln_data_cache_sha256",
            "protocol.json": "ln_protocol_sha256",
            "split_counts.csv": "ln_split_counts_sha256"}


def ln_file_path(ln_dir, fn):
    """LN layout: files under manifests/ (eta + local LN) or run-dir root
    (dev staging)."""
    p = os.path.join(ln_dir, "manifests", fn)
    return p if os.path.isfile(p) else os.path.join(ln_dir, fn)


def fail(msg):
    print(f"PREPARE ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def disk_gate():
    usage = shutil.disk_usage(SNAPSHOT_DIR)
    free_gb = usage.free / 1024 ** 3
    need = CFG["resources"]["min_disk_free_gb"]
    print(f"[prepare] disk free {free_gb:.1f} GiB (gate >= {need} GiB)")
    if free_gb < need:
        fail(f"only {free_gb:.1f} GiB free, need >= {need} GiB")


def locate_ln_dir():
    def has_all(d):
        return all(os.path.isfile(ln_file_path(d, f)) for f in LN_FILES)
    env_dir = os.environ.get("TM_LN_DIR", "").strip()
    if env_dir:
        if not has_all(env_dir):
            fail(f"TM_LN_DIR {env_dir} lacks the 4 LN files (dev override)")
        print(f"[prepare] LN source dir: {env_dir} (TM_LN_DIR dev override)")
        return env_dir
    src = CFG["source_snapshot"]
    for key in ("ln_dir_default", "ln_dir_local_fallback"):
        d = src[key]
        if has_all(d):
            print(f"[prepare] LN source dir: {d}")
            return d
    fail("LN dir not found (neither default nor fallback with all 4 files)")


def verify_ln_files(ln_dir):
    pins = CFG["source_snapshot"]
    hashes = {}
    for fn, pin_key in LN_FILES.items():
        p = ln_file_path(ln_dir, fn)
        h = sha256_file(p)
        if h != pins[pin_key]:
            fail(f"{p} sha256 {h} != pinned {pins[pin_key]}")
        hashes[fn] = h
        print(f"[prepare] OK sha256 {fn} = {h[:16]}...")
    return hashes


def verify_ln_protocol(ln_dir):
    with open(ln_file_path(ln_dir, "protocol.json")) as f:
        proto = json.load(f)
    src = CFG["source_snapshot"]
    exp = CFG["budget"]["expected_totals"]
    tot = proto.get("totals")
    if tot.get("native_train") != exp["native_train"] or \
       tot.get("dense") != exp["dense"] or tot.get("val") != exp["val"] or \
       tot.get("test") != exp["test"] or tot.get("excluded") != exp["excluded"]:
        fail(f"LN protocol totals {tot} != expected {exp}")
    if proto.get("split_manifest_sha256") != src["ln_split_manifest_sha256"] \
            or proto.get("data_cache_sha256") != src["ln_data_cache_sha256"]:
        fail("LN protocol self-declared data hashes != pinned values")
    if proto.get("zip_path_used") != \
            "/dev_data/wlt/VisionTS_Experiments/repro_ln_timesx_v1/dataset/TimesX_Datasets.zip":
        fail(f"unexpected zip_path_used {proto.get('zip_path_used')}")
    bad = {k: v for k, v in proto["code_fingerprint"].items()
           if md5_file(os.path.join(ln_dir, k)) != v}
    if bad:
        fail(f"LN static code drift in {sorted(bad)} - LN materials no longer "
             f"identical to the audited freeze")
    print(f"[prepare] OK LN protocol totals consistent, code_fingerprint "
          f"{len(proto['code_fingerprint'])} files recomputed clean")
    return proto


def verify_content_level(ln_dir):
    src = CFG["source_snapshot"]
    with open(ln_file_path(ln_dir, "split_manifest.json")) as f:
        sm = json.load(f)
    sm2 = {k: v for k, v in sm.items() if k != "zip_path_used"}
    ch = hashlib.sha256(json.dumps(sm2, sort_keys=True).encode()).hexdigest()
    if ch != src["ln_split_manifest_content_sha256_excl_zip_path"]:
        fail(f"split_manifest content hash {ch} != pinned")
    with np.load(ln_file_path(ln_dir, "data_cache.npz"),
                 allow_pickle=False) as z:
        for arr_name, pin in src["ln_data_cache_array_sha256"].items():
            ah = hashlib.sha256(z[arr_name].tobytes()).hexdigest()
            if ah != pin:
                fail(f"data_cache[{arr_name}] array hash {ah} != pinned")
    print("[prepare] OK content-level: split_manifest excl zip_path + 4 "
          "data_cache arrays")


def export_files(ln_dir, hashes):
    os.makedirs(SNAPSHOT, exist_ok=True)
    for fn in LN_FILES:
        dst = os.path.join(SNAPSHOT, fn)
        if os.path.isfile(dst):
            if sha256_file(dst) != hashes[fn]:
                fail(f"existing snapshot/{fn} differs from verified LN source"
                     f" - refusing to overwrite; inspect manually")
            continue
        tmp = dst + ".tmp"
        shutil.copyfile(ln_file_path(ln_dir, fn), tmp)
        if sha256_file(tmp) != hashes[fn]:
            os.remove(tmp)
            fail(f"copy of {fn} failed hash recheck")
        os.replace(tmp, dst)
    lines = [f"{hashes[fn]}  {fn}" for fn in sorted(LN_FILES)]
    with open(os.path.join(SNAPSHOT, "SHA256SUMS.txt.tmp"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(os.path.join(SNAPSHOT, "SHA256SUMS.txt.tmp"),
               os.path.join(SNAPSHOT, "SHA256SUMS.txt"))
    print(f"[prepare] exported {len(LN_FILES)} files -> snapshot/manifests/")


def validate_snapshot_content():
    try:
        data = DenseData()
    except SnapshotError as exc:
        fail(f"snapshot content invalid: {exc}")
    sum_u = 0
    u_d_table = {}
    for row in CFG["budget"]["expected_domain_table"]:
        dm = row["domain"]
        tot = data.domain_dense_total(dm)
        if tot != row["dense"]:
            fail(f"{dm}: dense total {tot} != table {row['dense']}")
        u, _ = u_d_s_d(tot)
        u_d_table[dm] = u
        sum_u += u
        for vk in data.domain_vars[dm]:
            m = data.manifest["variables"][vk]
            want_dc = max(0, m["n_train_obs"] - 107)
            if m["dense_count"] != want_dc:
                fail(f"{vk}: dense_count {m['dense_count']} != "
                     f"n_train_obs-107 = {want_dc}")
            n_obs = len(data.series(vk))
            for split in ("train", "val", "test"):
                for sid in m["native"][split]:
                    st = m["sample_start_idx"][sid]
                    if not (0 <= st and st + 108 <= n_obs):
                        fail(f"{vk}/{sid}: start {st} out of bounds "
                             f"(n_obs={n_obs})")
    if sum_u != CFG["budget"]["sum_U_d_expected"]:
        fail(f"sum(U_d) {sum_u} != {CFG['budget']['sum_U_d_expected']}")
    print(f"[prepare] OK snapshot content: 190 vars / 19 domains, "
          f"dense_count recomputed, bounds+fallback finite, sum(U_d)={sum_u}")
    return u_d_table, sum_u


def build_projection(ln_hashes, ln_proto, u_d_table, sum_u):
    act = actual_fingerprints()
    ts = tslib_state()
    if ts["violations"]:
        fail(f"TSLib working tree has non-allowed changes: {ts['violations']}")
    core_md5 = ts["core_md5"]
    if core_md5 != CFG["model"]["core_md5"]:
        fail(f"TSLib core md5 drift: {core_md5} != {CFG['model']['core_md5']}")
    bindings = {"data_cache": ln_hashes["data_cache.npz"],
                "split_manifest": ln_hashes["split_manifest.json"],
                "protocol_ln": ln_hashes["protocol.json"],
                "tslib_commit": ts["commit"],
                "tslib_violations": ts["violations"],
                "tslib_core_md5": core_md5,
                "code": act["code"],
                "protocol_self": None}
    exported_from = CFG["source_snapshot"]["ln_dir_default"] if os.path.isdir(
        CFG["source_snapshot"]["ln_dir_default"]) else \
        CFG["source_snapshot"]["ln_dir_local_fallback"]
    proto = {
        "protocol_version": CFG["protocol_version"],
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ln_source": {**CFG["source_snapshot"], "exported_from": exported_from},
        "ln_protocol_totals": ln_proto["totals"],
        "bindings": bindings,
        "budget": {"epochs": CFG["budget"]["epochs"],
                   "patience": CFG["budget"]["patience"],
                   "seeds": CFG["budget"]["seeds"],
                   "expected_totals": CFG["budget"]["expected_totals"],
                   "expected_domain_table":
                       CFG["budget"]["expected_domain_table"],
                   "sum_U_d": sum_u, "U_d_table": u_d_table},
        "optim": {"lrs": CFG["optim"]["lrs"], "batch": CFG["optim"]["batch"],
                  "betas": CFG["optim"]["betas"], "eps": CFG["optim"]["eps"],
                  "weight_decay": CFG["optim"]["weight_decay"],
                  "clip_grad": CFG["optim"]["clip_grad"]},
        "data": {"ctx": CFG["data"]["ctx"], "pred": CFG["data"]["pred"],
                 "std_eps": CFG["data"]["std_eps"],
                 "fallback_std_rule": CFG["data"]["fallback_std_rule"]},
        "model_pins": {"params_expected": CFG["model"]["params_expected"],
                       "tensors_expected": CFG["model"]["tensors_expected"],
                       "grad_none_expected": CFG["model"]["grad_none_expected"],
                       "core_md5": CFG["model"]["core_md5"]},
        "selection": {"tie_tol": CFG["selection"]["tie_tol"],
                      "patience": CFG["selection"]["patience"],
                      "lr_rule": CFG["selection"]["lr_rule"]},
        "test": {"eval_batch": CFG["test"]["eval_batch"],
                 "eval_dtype": CFG["test"]["eval_dtype"],
                 "per_seed_coverage": CFG["test"]["per_seed_coverage"]},
        "divergence_protocol": CFG["divergence_protocol"],
        "resources": CFG["resources"],
        "slot_probe_hashes": CFG.get("slot_probe_hashes", {}),
    }
    # phase 1: write with protocol_self=None; phase 2: fill with the sha256 of
    # the phase-1 bytes (reproducible via common.protocol_self_hash)
    atomic_write_json(proto, SNAPSHOT_PROTOCOL)
    self_h = sha256_file(SNAPSHOT_PROTOCOL)
    proto["bindings"]["protocol_self"] = self_h
    atomic_write_json(proto, SNAPSHOT_PROTOCOL)
    if protocol_self_hash() != self_h:
        fail("protocol_self roundtrip mismatch (serializer drift)")
    print(f"[prepare] frozen projection written: snapshot/protocol.json "
          f"(self={self_h[:16]}...)")


def write_baseline():
    files = {}
    for root, dirs, fns in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT)
        if rel_root == HERE_NAME or rel_root.startswith(HERE_NAME + os.sep):
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            files[os.path.relpath(p, REPO_ROOT)] = sha256_file(p)
    atomic_write_json({"created_utc": datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(REPO_ROOT), "n_files": len(files), "files": files},
        BASELINE_JSON)
    print(f"[prepare] baseline.json: {len(files)} static TSLib files hashed")


HERE_NAME = os.path.basename(HERE)


def main():
    disk_gate()
    ln_dir = locate_ln_dir()
    ln_hashes = verify_ln_files(ln_dir)
    ln_proto = verify_ln_protocol(ln_dir)
    verify_content_level(ln_dir)
    export_files(ln_dir, ln_hashes)
    u_d_table, sum_u = validate_snapshot_content()
    build_projection(ln_hashes, ln_proto, u_d_table, sum_u)
    write_baseline()
    print("[prepare] ALL OK")


if __name__ == "__main__":
    main()
