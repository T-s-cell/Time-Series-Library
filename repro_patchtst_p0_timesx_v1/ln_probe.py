#!/usr/bin/env python3
"""Slot probe hashes from the LN run's REAL build_slots, F0 native branch.

LN's trainer.build_slots and common.u_d_s_d are extracted from the LN source
via ast (no import -> no module-name collisions with this experiment's
modules) and executed in an isolated namespace against this experiment's
DenseData with method="F0" (native pools, sample_id slots). The resulting
slot lists are hashed with OUR slot_hash and pinned in configs/v1.yaml
(slot_probe_hashes).

- Locally: `python ln_probe.py --manifests-dir <dir> --print` generates the
  pins from the local LN copy (whose static code is md5-identical to eta's)
  against an already-exported snapshot (T1's, hash-identical to the pins).
- preflight section D re-runs this on theta against the eta-sourced LN copy
  and requires bitwise-identical hashes -> cross-machine proof that our
  sampling clone consumes the same RNG families in the same order as LN's F0.

native_slot_equality additionally compares LN F0's slot lists against OUR
trainer.build_slots element-wise for all 19 domains x 3 seeds x epochs 1..10
(570 combos) - the direct replication proof required by the T0 protocol.

Probes (6): arts 2021 ep1, arts 2022 ep1, arts 2021 ep37 (beyond LN's 10
epochs), Currency 2021 ep1, Currency 2023 ep10, public_policy 2022 ep5.
"""
import argparse
import ast
import json
import os

import numpy as np

from common import CFG, DOMAIN_IDX, DOMAIN_NAMES, SEEDS
from trainer import build_slots as our_build_slots
from trainer import slot_hash

PROBES = [("arts", 2021, 1), ("arts", 2022, 1), ("arts", 2021, 37),
          ("Currency", 2021, 1), ("Currency", 2023, 10),
          ("public_policy", 2022, 5)]

ALIGN_EPOCHS = range(1, 11)

DEFAULT_LN = CFG["source_snapshot"]["ln_dir_default"]
FALLBACK_LN = CFG["source_snapshot"]["ln_dir_local_fallback"]


def extract_funcs(path, names):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    segs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name in names:
            seg = ast.get_source_segment(src, node)
            if seg is None:
                raise RuntimeError(f"cannot extract {node.name} from {path}")
            segs[node.name] = seg
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    segs[t.id] = ast.get_source_segment(src, node)
    missing = set(names) - set(segs)
    if missing:
        raise RuntimeError(f"{path}: definitions not found: {sorted(missing)}")
    return segs


def load_ln_build_slots(ln_dir):
    """Returns LN's real build_slots(data, method, domain, seed, epoch)."""
    segs = extract_funcs(os.path.join(ln_dir, "trainer.py"), ["build_slots"])
    segs.update(extract_funcs(os.path.join(ln_dir, "common.py"), ["u_d_s_d"]))
    proto_p = os.path.join(ln_dir, "protocol.json")
    if not os.path.isfile(proto_p):
        proto_p = os.path.join(ln_dir, "manifests", "protocol.json")
    with open(proto_p) as f:
        methods = set(json.load(f)["config"]["budget"]["methods_train"])
    ns = {"np": np, "DOMAIN_IDX": DOMAIN_IDX, "METHODS_TRAIN": methods}
    for name, seg in segs.items():
        exec(compile(seg, f"<ln:{name}>", "exec"), ns)
    return ns["build_slots"]


def compute_probe_hashes(ln_dir, data):
    build_slots = load_ln_build_slots(ln_dir)
    out = {}
    for dom, seed, epoch in PROBES:
        slots, U, S = build_slots(data, "F0", dom, seed, epoch)
        out[f"{dom}|{seed}|{epoch}"] = {
            "slot_hash": slot_hash(slots), "n_slots": len(slots),
            "U": int(U), "S": int(S)}
    return out


def native_slot_equality(ln_dir, data):
    """Element-wise equality of LN F0 slot lists vs OUR build_slots for all
    19 domains x 3 seeds x epochs 1..10 (570 combos), plus identical (U, S).
    Returns (n_ok, n_total, mismatches[:8])."""
    ln_slots = load_ln_build_slots(ln_dir)
    n_ok = 0
    mism = []
    total = 0
    for dom in DOMAIN_NAMES:
        for seed in SEEDS:
            for epoch in ALIGN_EPOCHS:
                total += 1
                a, Ua, Sa = ln_slots(data, "F0", dom, seed, epoch)
                b, Ub, Sb = our_build_slots(data, dom, seed, epoch)
                if a == b and (int(Ua), int(Sa)) == (int(Ub), int(Sb)):
                    n_ok += 1
                elif len(mism) < 8:
                    why = ("slots" if a != b else
                           f"budget LN=({Ua},{Sa}) ours=({Ub},{Sb})")
                    mism.append(f"{dom}|s{seed}|ep{epoch}: {why}")
    return n_ok, total, mism


def locate_ln_dir(explicit):
    ln_dir = explicit
    if ln_dir is None:
        ln_dir = os.environ.get("P0_LN_DIR")
    if ln_dir is None:
        for cand in (DEFAULT_LN, FALLBACK_LN):
            if os.path.isfile(os.path.join(cand, "trainer.py")):
                ln_dir = cand
                break
    if ln_dir is None or not os.path.isfile(os.path.join(ln_dir, "trainer.py")):
        raise SystemExit("LN dir with trainer.py not found "
                         "(set P0_LN_DIR or pass --ln-dir)")
    return ln_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln-dir", default=None)
    ap.add_argument("--manifests-dir", default=None,
                    help="DenseData manifests dir (default: this exp's "
                         "snapshot/manifests; use T1's exported snapshot to "
                         "generate pins before T0 prepare exists)")
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()
    ln_dir = locate_ln_dir(args.ln_dir)
    from data import DenseData
    data = DenseData(args.manifests_dir)
    hashes = compute_probe_hashes(ln_dir, data)
    for k, v in hashes.items():
        print(f"{k}: {v['slot_hash'][:16]} (n={v['n_slots']} U={v['U']} "
              f"S={v['S']})")
    if args.show:
        print(json.dumps(hashes, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
