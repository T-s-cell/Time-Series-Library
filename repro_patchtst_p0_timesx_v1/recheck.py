#!/usr/bin/env python3
"""recheck: independent post-aggregate verification, two evidence classes.

A. metric recompute - from each saved test npz PLUS the snapshot's raw
   float64 series: re-derive start_idx (manifest lookup), the raw target y,
   d_i = window_d(raw 96-step input), and per-window MSE/MAE from the stored
   pred. Then rebuild the full aggregation chain (var -> 19 domains ->
   overall -> 3-seed mean/std) with the same closed-form functions aggregate
   uses and compare against results/aggregate_summary.json + overall.csv.
   Stored values were produced by identical float64 op chains, so equality
   must hold bitwise; the pre-frozen sanity bound
   recheck_tolerances.metric_recompute.abs_tol (1e-12) only makes the
   assertion explicit - anything above it FAILS.
B. inference replay - each of the 57 selected checkpoints is re-loaded
   (sha256 re-verified against selection.json) and re-run on CUDA over its
   own domain's native test windows (eval mode, no_grad, EVAL_BATCH=64,
   stored row order). Predictions are compared elementwise against the
   saved npz. PRIMARY assertion: bitwise equality. A mismatch first records
   full diagnostics (max abs/rel, first mismatch index); the pre-frozen
   grace recheck_tolerances.inference_replay.grace_rtol (rel<=1e-6) may only
   downgrade that run to PASS_WITH_DISCLOSURE - tolerances are never
   loosened after a failure, and every non-bitwise outcome stays disclosed
   in the report and in the final status.

Run after aggregate; requires selection.json + 57 test npz + aggregate
outputs. CUDA is REQUIRED (no CPU fallback). Writes results/recheck_report
.json; exit 0 on PASS or PASS_WITH_DISCLOSURE.
"""
import json
import os
import sys

import numpy as np
import torch

import aggregate
from common import (CFG, CHECKPOINTS, DOMAIN_NAMES, EVAL_BATCH, PREDICTIONS,
                    RESULTS, SEEDS, atomic_write_json, launch_gate,
                    sha256_file)
from data import DenseData, window_d
from model_adapter import forward_pred, load_model, setup_numerics

TOL = CFG["recheck_tolerances"]["metric_recompute"]["abs_tol"]
GRACE_RTOL = CFG["recheck_tolerances"]["inference_replay"]["grace_rtol"]
REPORT = os.path.join(RESULTS, "recheck_report.json")


def fail(msg):
    print(f"RECHECK ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_npz(rid):
    with np.load(os.path.join(PREDICTIONS, f"{rid}__test.npz"),
                 allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def max_rel(a, b):
    denom = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


def diff_or_inf(a, b):
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def replay_rows(model, rows, data, device):
    """Re-run inference on saved test windows; identical recipe to
    evaluate_test.predict_rows (eval mode + no_grad, EVAL_BATCH chunks,
    float32 inputs, float64 outputs, d_i from raw history)."""
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
    launch_gate("recheck")
    sel_path = os.path.join(RESULTS, "selection.json")
    agg_path = os.path.join(RESULTS, "aggregate_summary.json")
    for p in (sel_path, agg_path):
        if not os.path.isfile(p):
            fail(f"{os.path.basename(p)} missing - run freeze/test/aggregate "
                 f"first")
    with open(sel_path) as f:
        sel = json.load(f)
    if not sel.get("frozen"):
        fail("selection.json not frozen")
    with open(agg_path) as f:
        agg = json.load(f)

    setup_numerics()
    if not torch.cuda.is_available():
        fail("CUDA unavailable - no CPU fallback per protocol")
    device = "cuda"
    data = DenseData()

    metric_runs = []
    infer_runs = []
    metric_ok = True
    bitwise_all = True
    disclosed = []

    # ---------------- A. metric recompute ----------------
    print("[recheck] A: metric recompute from npz + raw series", flush=True)
    per_seed_tables = {s: [] for s in SEEDS}
    for dom in DOMAIN_NAMES:
        block = sel["domains"][dom]
        for seed_s, ent in block["seeds"].items():
            seed = int(seed_s)
            rid = ent["run_id"]
            z = load_npz(rid)
            vks = [str(x) for x in z["var_key"]]
            sids = [str(x) for x in z["sample_id"]]
            rows = data.split_rows(dom, "test")
            if [r[0] for r in rows] != vks or [r[1] for r in rows] != sids:
                fail(f"{rid}: npz row order != snapshot split_rows order")
            starts = np.array([data.start_of(vk, sid)
                               for vk, sid in zip(vks, sids)], dtype=np.int64)
            if not np.array_equal(starts, z["start_idx"]):
                fail(f"{rid}: stored start_idx != manifest lookup - evidence "
                     f"corruption, refusing to recompute")
            y_rec = np.stack([r[3] for r in rows])
            d_rec = np.array([window_d(r[2], data.fb_std[r[0]])
                              for r in rows], dtype=np.float64)
            pred = z["pred"]
            raw_err = pred - y_rec
            mse_rec = np.mean((raw_err / d_rec[:, None]) ** 2, axis=1)
            mae_rec = np.mean(np.abs(raw_err) / d_rec[:, None], axis=1)
            diffs = {
                "max_abs_target_diff": diff_or_inf(y_rec, z["target"]),
                "max_abs_d_diff": diff_or_inf(d_rec, z["d"]),
                "max_abs_mse_diff": diff_or_inf(mse_rec, z["mse"]),
                "max_abs_mae_diff": diff_or_inf(mae_rec, z["mae"])}
            ok = all(v <= TOL and np.isfinite(v) for v in diffs.values())
            metric_ok = metric_ok and ok
            metric_runs.append({"run_id": rid, "ok": ok,
                                "tol": TOL, **{k: float(v) for k, v
                                               in diffs.items()}})
            per_seed_tables[seed].append(
                aggregate.var_table_of({
                    "var": vks, "mse": mse_rec, "mae": mae_rec,
                    "raw_mse": np.mean(raw_err ** 2, axis=1),
                    "raw_mae": np.mean(np.abs(raw_err), axis=1)}))
            if not ok:
                print(f"  [recheck] FAIL recompute {rid}: "
                      + " ".join(f"{k}={v:.3e}" for k, v in diffs.items()),
                      flush=True)
            del z

    seed_ov = {}
    for seed in SEEDS:
        merged = aggregate.merge_var_tables(per_seed_tables[seed])
        _, seed_ov[seed] = aggregate.dom_overall(merged)
    rec_overall = {
        "mse": list(aggregate.seed_stats([seed_ov[s]["mse"] for s in SEEDS])),
        "mae": list(aggregate.seed_stats([seed_ov[s]["mae"] for s in SEEDS]))}
    agg_p0 = agg.get("methods_overall", {}).get("P0")
    if agg_p0 is None:
        fail("aggregate_summary.json has no methods_overall.P0")
    agg_diffs = {"mse_mean": abs(rec_overall["mse"][0] - agg_p0["mse"][0]),
                 "mse_std": abs(rec_overall["mse"][1] - agg_p0["mse"][1]),
                 "mae_mean": abs(rec_overall["mae"][0] - agg_p0["mae"][0]),
                 "mae_std": abs(rec_overall["mae"][1] - agg_p0["mae"][1])}
    agg_ok = all(v <= TOL for v in agg_diffs.values())
    metric_ok = metric_ok and agg_ok and agg.get("n_vars") == 190

    # 6-dp cross-check against overall.csv first data row (P0)
    csv_ok = None
    ov_path = os.path.join(RESULTS, "overall.csv")
    if os.path.isfile(ov_path):
        import csv as _csv
        with open(ov_path) as f:
            rows = list(_csv.reader(f))
        p0_rows = [r for r in rows[1:] if r and r[0] == "P0"]
        if len(p0_rows) == 1:
            mse_c, mae_c = aggregate.parse_pm(p0_rows[0][2]), \
                aggregate.parse_pm(p0_rows[0][3])
            csv_ok = (aggregate.fmt(*rec_overall["mse"]) == p0_rows[0][2]
                      and aggregate.fmt(*rec_overall["mae"]) == p0_rows[0][3]
                      and abs(mse_c[0] - rec_overall["mse"][0]) < 5e-7
                      and abs(mae_c[0] - rec_overall["mae"][0]) < 5e-7)
    metric_ok = metric_ok and (csv_ok is not False)
    print(f"[recheck] A: recomputed overall MSE "
          f"{aggregate.fmt(*rec_overall['mse'])} MAE "
          f"{aggregate.fmt(*rec_overall['mae'])} vs aggregate "
          f"(max diff {max(agg_diffs.values()):.3e}, "
          f"overall.csv={'match' if csv_ok else 'n/a'})", flush=True)

    # ---------------- B. inference replay ----------------
    print("[recheck] B: inference replay of 57 selected checkpoints",
          flush=True)
    for dom in DOMAIN_NAMES:
        block = sel["domains"][dom]
        for seed_s, ent in block["seeds"].items():
            seed = int(seed_s)
            rid = ent["run_id"]
            ck = os.path.join(CHECKPOINTS, rid, os.path.basename(ent["ckpt"]))
            if sha256_file(ck) != ent["ckpt_sha256"]:
                fail(f"{rid}: checkpoint sha256 drift vs selection.json")
            model = load_model(device)
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=True)
            model.to(device)
            z = load_npz(rid)
            rows = data.split_rows(dom, "test")
            pred_rec, _ = replay_rows(model, rows, data, device)
            n_mismatch = int(np.sum(pred_rec != z["pred"]))
            entry = {"run_id": rid, "n_windows": int(pred_rec.shape[0]),
                     "bitwise": n_mismatch == 0}
            if n_mismatch:
                bitwise_all = False
                m = pred_rec != z["pred"]
                first = int(np.argmax(m.any(axis=1)))
                rel = max_rel(pred_rec, z["pred"])
                entry.update({"n_mismatch": n_mismatch,
                              "max_abs_diff":
                                  float(np.max(np.abs(pred_rec
                                                      - z["pred"]))),
                              "max_rel_diff": rel,
                              "first_mismatch_row": first})
                if rel <= GRACE_RTOL:
                    entry["verdict"] = "PASS_WITH_DISCLOSURE"
                    disclosed.append(rid)
                    print(f"  [recheck] DISCLOSED {rid}: non-bitwise but "
                          f"max_rel={rel:.3e} <= grace {GRACE_RTOL:.0e}",
                          flush=True)
                else:
                    entry["verdict"] = "FAIL"
                    infer_ok = False
                    print(f"  [recheck] FAIL replay {rid}: n={n_mismatch} "
                          f"max_rel={rel:.3e} > grace {GRACE_RTOL:.0e} - "
                          f"investigate, do not loosen tolerances",
                          flush=True)
            else:
                entry["verdict"] = "bitwise"
            infer_runs.append(entry)
            print(f"  [recheck] replay {rid}: {entry['verdict']}", flush=True)
            del model, sd, z
            torch.cuda.empty_cache()

    n_inf_fail = sum(1 for r in infer_runs if r["verdict"] == "FAIL")
    status = "FAIL" if (not metric_ok or n_inf_fail) else \
        ("PASS_WITH_DISCLOSURE" if disclosed else "PASS")
    report = {
        "metric_recompute": {
            "abs_tol": TOL, "ok": metric_ok, "runs": metric_runs,
            "recomputed_overall": {"mse": rec_overall["mse"],
                                   "mae": rec_overall["mae"]},
            "aggregate_overall": {"mse": agg_p0["mse"], "mae": agg_p0["mae"]},
            "max_abs_aggregate_diff":
                {k: float(v) for k, v in agg_diffs.items()},
            "overall_csv_match": csv_ok},
        "inference_replay": {
            "primary": CFG["recheck_tolerances"]["inference_replay"]
            ["primary"],
            "grace_rtol": GRACE_RTOL, "bitwise_all": bitwise_all,
            "n_disclosed": len(disclosed), "disclosed_runs": disclosed,
            "runs": infer_runs},
        "status": status}
    tmp = REPORT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    os.replace(tmp, REPORT)
    print(f"[recheck] status={status} "
          f"(metric {'OK' if metric_ok else 'FAIL'}, replay "
          f"{len(infer_runs) - n_inf_fail}/{len(infer_runs)} within criteria, "
          f"{len(disclosed)} disclosed) -> results/recheck_report.json",
          flush=True)
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
