#!/usr/bin/env python3
"""evaluate_test: run the 57 frozen checkpoints; each model predicts ONLY its
own domain's native test windows. Per seed, the union over the 19 domain
models is exactly the 2,474 unique native test windows, so the campaign
totals 3 x 2,474 = 7,422 window predictions across 57 npz files.

- Only runs after freeze (results/selection.json required; checkpoint sha256
  re-verified against it before every load).
- Launch gate: preflight AND vendor_check reports must be PASS and their
  bindings (snapshot / TSLib / vendor / code / env / CUDA) must match the
  live tree.
- CUDA is REQUIRED: no CPU fallback per protocol (hard abort if absent).
- Eval mode + no_grad, EVAL_BATCH=64, shuffle off, tail batch included
  (drop_last=False), float64 metrics identical to evaluate_split.
- Per (run) npz: pred/target [N,12] float64, d [N], mse/mae [N], start_idx
  [N] (window start in the variable's series, == manifest sample_start_idx),
  plus var_key/sample_id/checkpoint strings.
- Coverage assertion: for each seed, the union over the 19 domain models is
  exactly the manifest's 2,474 unique native test windows.
"""
import json
import os
import sys

import numpy as np
import torch

from common import (CFG, CHECKPOINTS, DOMAIN_NAMES, PREDICTIONS, RESULTS,
                    SEEDS, launch_gate, sha256_file)
from data import DenseData, window_d
from model_adapter import forward_pred, load_model, setup_numerics

EVAL_BATCH = CFG["test"]["eval_batch"]


def predict_rows(model, rows, data, device):
    preds, ds = [], []
    was = model.training
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(rows), EVAL_BATCH):
            chunk = rows[i0:i0 + EVAL_BATCH]
            xt = torch.tensor(np.stack([r[2] for r in chunk]),
                              dtype=torch.float32, device=device).unsqueeze(-1)
            pred = forward_pred(model, xt).detach().cpu().numpy().astype(
                np.float64)
            preds.append(pred)
            ds.extend([window_d(x, data.fb_std[vk])
                       for vk, _, x, _ in chunk])
    if was:
        model.train()
    return np.concatenate(preds, 0), np.array(ds, dtype=np.float64)


def main():
    launch_gate("test")
    sel_path = os.path.join(RESULTS, "selection.json")
    if not os.path.isfile(sel_path):
        print("TEST ABORT: results/selection.json missing - freeze first",
              file=sys.stderr)
        return 2
    with open(sel_path) as f:
        sel = json.load(f)
    if not sel.get("frozen"):
        print("TEST ABORT: selection.json not frozen", file=sys.stderr)
        return 2

    setup_numerics()
    if not torch.cuda.is_available():
        print("TEST ABORT: CUDA unavailable - no CPU fallback per protocol",
              file=sys.stderr)
        return 2
    device = "cuda"
    data = DenseData()

    per_seed_windows = {s: [] for s in SEEDS}
    for dom in DOMAIN_NAMES:
        block = sel["domains"][dom]
        for seed_s, ent in block["seeds"].items():
            seed = int(seed_s)
            rid = ent["run_id"]
            ck = os.path.join(CHECKPOINTS, rid, os.path.basename(ent["ckpt"]))
            sha = sha256_file(ck)
            if sha != ent["ckpt_sha256"]:
                print(f"TEST ABORT: {ck} sha256 drift vs selection.json",
                      file=sys.stderr)
                return 2
            model = load_model(device)
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=True)
            model.to(device)

            rows = data.split_rows(dom, "test")
            preds, ds = predict_rows(model, rows, data, device)
            recs = {"var_key": np.array([r[0] for r in rows]),
                    "sample_id": np.array([r[1] for r in rows]),
                    "start_idx": np.array(
                        [data.start_of(r[0], r[1]) for r in rows],
                        dtype=np.int64),
                    "seed": np.full(len(rows), seed, dtype=np.int64),
                    "checkpoint_sha256": np.full(len(rows), sha),
                    "pred": preds, "target": np.stack([r[3] for r in rows]),
                    "d": ds}
            err = (recs["pred"] - recs["target"]) / ds[:, None]
            recs["mse"] = np.mean(err ** 2, axis=1)
            recs["mae"] = np.mean(np.abs(err), axis=1)
            if not (np.isfinite(recs["mse"]).all()
                    and np.isfinite(recs["mae"]).all()):
                print(f"TEST ABORT: {rid}: non-finite test metrics",
                      file=sys.stderr)
                return 2
            out = os.path.join(PREDICTIONS, f"{rid}__test.npz")
            tmp = out + ".tmp.npz"
            np.savez_compressed(tmp, **recs)
            os.replace(tmp, out)
            per_seed_windows[seed].extend(
                (vk, sid) for vk, sid, _, _ in rows)
            print(f"[test] {rid}: {len(rows)} windows -> {os.path.basename(out)}"
                  f" (mse={float(np.mean(recs['mse'])):.6f})", flush=True)
            del model

    exp_total = CFG["test"]["per_seed_coverage"]
    manifest_test = set()
    for dom in DOMAIN_NAMES:
        for vk in data.domain_vars[dom]:
            for sid in data.native_ids(vk, "test"):
                manifest_test.add((vk, sid))
    if len(manifest_test) != exp_total:
        print(f"TEST ABORT: manifest has {len(manifest_test)} unique test "
              f"windows != {exp_total}", file=sys.stderr)
        return 2
    for seed, wins in per_seed_windows.items():
        uniq = set(wins)
        if len(wins) != exp_total or uniq != manifest_test:
            print(f"TEST ABORT: seed {seed} coverage {len(wins)} windows / "
                  f"{len(uniq)} unique != manifest {exp_total}", file=sys.stderr)
            return 2
        print(f"[test] seed {seed} coverage OK: {len(uniq)} unique windows "
              f"== manifest")
    print("[test] ALL OK (57 models, 3 x 2,474 window coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
