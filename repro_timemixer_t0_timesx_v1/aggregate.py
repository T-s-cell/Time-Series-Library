#!/usr/bin/env python3
"""Stage aggregate: T0 tables + T0-vs-F0 and T0-vs-T1 comparisons
(comparison metrics PARSED from the reference runs' results/, never
recomputed).

Aggregation (LN-identical closed forms): window mean -> variable -> merge all
19 domains of one seed [assert no duplicate vars, 190 vars / 19 domains] ->
domain equal-weight -> overall; 3-seed mean +/- std(ddof=1). Raw-scale metrics
only at variable level.

Reference inputs: LN run (from --ln-dir, default config; F0 = LN-native main
result, no zero-shot fallback) and T1 results (from --t1-dir, the
TimeMixer-dense run). Both provide overall.csv, domain_summary.csv,
test_variable_level.csv. Values like "0.123+/-0.004" are parsed as
(mean, std). If LN outputs are unavailable the tables degrade to T0-only
with comparison_status=ln_results_unavailable; missing --t1-dir degrades the
T0-vs-T1 block with t1_comparison_status=t1_results_unavailable. Re-runs are
idempotent once references finish.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

from common import (CFG, DOMAIN_NAMES, PREDICTIONS, RESULTS, SEEDS,
                    atomic_write_json, sha256_file)

METRICS = ("mse", "mae", "raw_mse", "raw_mae")


class AggregateError(RuntimeError):
    pass


def load_set(path):
    with np.load(path, allow_pickle=False) as z:
        pred, target, d = z["pred"], z["target"], z["d"]
        vks = [str(x) for x in z["var_key"]]
    err = pred - target
    return {"var": vks,
            "mse": np.mean((err / d[:, None]) ** 2, axis=1),
            "mae": np.mean(np.abs(err) / d[:, None], axis=1),
            "raw_mse": np.mean(err ** 2, axis=1),
            "raw_mae": np.mean(np.abs(err), axis=1)}


def var_table_of(entry):
    per_var = {}
    for vk, sm, sa, rm, ra in zip(entry["var"], entry["mse"], entry["mae"],
                                  entry["raw_mse"], entry["raw_mae"]):
        pv = per_var.setdefault(vk, {k: [] for k in METRICS})
        pv["mse"].append(sm)
        pv["mae"].append(sa)
        pv["raw_mse"].append(rm)
        pv["raw_mae"].append(ra)
    return {vk: {k: float(np.mean(v[k])) for k in METRICS} | {"n": len(v["mse"])}
            for vk, v in per_var.items()}


def merge_var_tables(tables, expect_vars=190, strict=True):
    merged = {}
    for t in tables:
        for vk, v in t.items():
            if vk in merged:
                raise AggregateError(f"duplicate var {vk} across domain runs")
            merged[vk] = v
    if strict:
        doms = {vk.split("__", 1)[0] for vk in merged}
        missing = sorted(set(DOMAIN_NAMES) - doms)
        extra = sorted(doms - set(DOMAIN_NAMES))
        if missing or extra:
            raise AggregateError(f"domain coverage broken: missing={missing[:3]} "
                                 f"extra={extra[:3]}")
        if len(merged) != expect_vars:
            raise AggregateError(f"{len(merged)} vars merged, expected "
                                 f"{expect_vars}")
    return merged


def dom_overall(merged, strict=True):
    dom_t = {}
    for d in DOMAIN_NAMES:
        vs = [merged[vk] for vk in merged if vk.split("__", 1)[0] == d]
        if not vs:
            if strict:
                raise AggregateError(f"no variables for domain {d}")
            continue
        dom_t[d] = {k: float(np.mean([v[k] for v in vs])) for k in METRICS}
    if not dom_t:
        raise AggregateError("no domains matched the merged var table")
    _assert_finite(dom_t, "domain table")
    overall = {k: float(np.mean([dom_t[d][k] for d in dom_t]))
               for k in METRICS}
    _assert_finite(overall, "overall")
    return dom_t, overall


def _assert_finite(obj, what):
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, float) and not math.isfinite(o):
            raise AggregateError(f"non-finite {what} at {path}")
    walk(obj, what)


def seed_stats(values):
    if len(values) == 1:
        return float(values[0]), None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt(mean, std):
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f}+/-{std:.6f}"


def parse_pm(s):
    """'0.123456+/-0.000010' -> (mean, std); plain number -> (v, None)."""
    s = s.strip()
    if "+/-" in s:
        a, b = s.split("+/-", 1)
        return float(a), float(b)
    return float(s), None


def locate_ln_dir(cli_val):
    cands = []
    if cli_val:
        cands.append(cli_val)
    src = CFG["source_snapshot"]
    cands.extend([src.get("ln_dir_default"),
                  src.get("ln_dir_local_fallback")])
    for d in cands:
        if not d:
            continue
        if os.path.isfile(os.path.join(d, "overall.csv")):
            return d
        if os.path.isfile(os.path.join(d, "results", "overall.csv")):
            return os.path.join(d, "results")
    return None


def locate_t1_dir(cli_val):
    if cli_val and (os.path.isfile(os.path.join(cli_val, "overall.csv"))
                    or os.path.isfile(
                        os.path.join(cli_val, "results", "overall.csv"))):
        return cli_val if os.path.isfile(os.path.join(cli_val, "overall.csv")) \
            else os.path.join(cli_val, "results")
    return None


# read_results_dir parses either reference run's CSVs - same writer code, same
# schema (method = column 0).
def read_results_dir(ln_dir):
    """Parse reference CSVs; returns {overall:{m:(mse,mae)},
    domain:{m:{d:(mse,mae)}}, var:{m:{vk:{...}}}} or None if files missing."""
    def rd(name):
        p = os.path.join(ln_dir, name)
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            return list(csv.reader(f))

    ov, dm, vr = rd("overall.csv"), rd("domain_summary.csv"), \
        rd("test_variable_level.csv")
    if ov is None or dm is None or vr is None:
        return None
    out = {"overall": {}, "domain": {}, "var": {}, "sources": {}}
    hdr = ov[0]
    im, ise, ima = 0, hdr.index("MSE"), hdr.index("MAE")
    for row in ov[1:]:
        out["overall"][row[im]] = (parse_pm(row[ise]), parse_pm(row[ima]))
    hdr = dm[0]
    im, idom, ise, ima = 0, hdr.index("domain"), hdr.index("MSE"), \
        hdr.index("MAE")
    for row in dm[1:]:
        out["domain"].setdefault(row[im], {})[row[idom]] = \
            (parse_pm(row[ise]), parse_pm(row[ima]))
    hdr = vr[0]
    im, ivk, ins = 0, hdr.index("var_key"), hdr.index("n_windows")
    ic = {k: hdr.index(k) for k in ("std_MSE", "std_MAE", "raw_MSE", "raw_MAE")}
    for row in vr[1:]:
        out["var"].setdefault(row[im], {})[row[ivk]] = {
            "n": int(row[ins]),
            "mse": parse_pm(row[ic["std_MSE"]]),
            "mae": parse_pm(row[ic["std_MAE"]]),
            "raw_mse": parse_pm(row[ic["raw_MSE"]]),
            "raw_mae": parse_pm(row[ic["raw_MAE"]])}
    for name in ("overall.csv", "domain_summary.csv",
                 "test_variable_level.csv", "aggregate_summary.json"):
        p = os.path.join(ln_dir, name)
        if os.path.isfile(p):
            out["sources"][name] = sha256_file(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln-dir", default=None,
                    help="LN run dir holding results/ CSVs (default: config)")
    ap.add_argument("--t1-dir", default=None,
                    help="T1 (TimeMixer-dense) results dir with overall.csv / "
                         "domain_summary.csv / test_variable_level.csv")
    args = ap.parse_args()

    sel_path = os.path.join(RESULTS, "selection.json")
    if not os.path.isfile(sel_path):
        print("AGGREGATE ABORT: results/selection.json missing", file=sys.stderr)
        return 2
    with open(sel_path) as f:
        sel = json.load(f)

    seed_merged, seed_dom, seed_ov = {}, {}, {}
    for seed in SEEDS:
        tables = []
        for dom, block in sel["domains"].items():
            rid = block["seeds"][str(seed)]["run_id"]
            tables.append(var_table_of(load_set(
                os.path.join(PREDICTIONS, f"{rid}__test.npz"))))
        seed_merged[seed] = merge_var_tables(tables)
        seed_dom[seed], seed_ov[seed] = dom_overall(seed_merged[seed])
    all_vars = sorted(seed_merged[SEEDS[0]])

    ln_loc = locate_ln_dir(args.ln_dir)
    ln = read_results_dir(ln_loc) if ln_loc is not None else None
    comparison_status = "ok" if ln else "ln_results_unavailable"
    if ln is None:
        print("[aggregate] LN results not found - degrading to T0-only "
              "(comparison_status=ln_results_unavailable); re-run "
              "idempotently once LN finishes", flush=True)
    side = CFG["comparison"]["side_by_side"] if ln else []

    t1_loc = locate_t1_dir(args.t1_dir)
    t1 = read_results_dir(t1_loc) if t1_loc is not None else None
    t1_status = "ok" if (t1 and "T1" in t1["overall"]) \
        else "t1_results_unavailable"
    if t1_status != "ok":
        print("[aggregate] T1 results not found/usable - degrading the "
              "T0-vs-T1 block (t1_comparison_status=t1_results_unavailable); "
              "re-run idempotently", flush=True)

    # ---------- overall ----------
    overall_rows = [["T0", "mean+-std over 3 seeds",
                     fmt(*seed_stats([seed_ov[s]["mse"] for s in SEEDS])),
                     fmt(*seed_stats([seed_ov[s]["mae"] for s in SEEDS]))]]
    for m in side:
        if m in ln["overall"]:
            (mse, mse_sd), (mae, mae_sd) = ln["overall"][m]
            overall_rows.append([f"{m} (LN)", "as reported",
                                 fmt(mse, mse_sd), fmt(mae, mae_sd)])
    if t1_status == "ok":
        (mse, mse_sd), (mae, mae_sd) = t1["overall"]["T1"]
        overall_rows.append(["T1 (dense)", "as reported",
                             fmt(mse, mse_sd), fmt(mae, mae_sd)])

    # ---------- domain summary ----------
    domain_rows = []
    for d in DOMAIN_NAMES:
        domain_rows.append(["T0", d, "mean+-std",
                            fmt(*seed_stats([seed_dom[s][d]["mse"]
                                             for s in SEEDS])),
                            fmt(*seed_stats([seed_dom[s][d]["mae"]
                                             for s in SEEDS]))])
    for m in side:
        if m in ln["domain"]:
            for d in DOMAIN_NAMES:
                (mse, msd), (mae, masd) = ln["domain"][m][d]
                domain_rows.append([f"{m} (LN)", d, "as reported",
                                    fmt(mse, msd), fmt(mae, masd)])
    if t1_status == "ok":
        for d in DOMAIN_NAMES:
            (mse, msd), (mae, masd) = t1["domain"]["T1"][d]
            domain_rows.append(["T1 (dense)", d, "as reported",
                                fmt(mse, msd), fmt(mae, masd)])

    # ---------- variable level ----------
    var_rows = []
    for vk in all_vars:
        per_seed = [seed_merged[s][vk] for s in SEEDS]
        ms = {k: [t[k] for t in per_seed] for k in METRICS}
        var_rows.append(["T0", vk, per_seed[0]["n"]] +
                        [fmt(*seed_stats(ms[k])) for k in METRICS])
    for m in side:
        if m in ln["var"]:
            for vk in all_vars:
                t = ln["var"][m].get(vk)
                if t is None:
                    continue
                var_rows.append([f"{m} (LN)", vk, t["n"]] +
                                [fmt(*t[k]) for k in METRICS])
    if t1_status == "ok":
        for vk in all_vars:
            t = t1["var"]["T1"].get(vk)
            if t is None:
                continue
            var_rows.append(["T1 (dense)", vk, t["n"]] +
                            [fmt(*t[k]) for k in METRICS])

    # ---------- T0 vs F0 (primary) and T0 vs T1 comparisons ----------
    comp_rows, impr_rows = [], []

    def comparison_block(label, ref_name, ref_ov, ref_dom_tab, ref_var_tab):
        (r_mse, _), (r_mae, _) = ref_ov
        v_mse = float(np.mean([seed_ov[s]["mse"] for s in SEEDS]))
        v_mae = float(np.mean([seed_ov[s]["mae"] for s in SEEDS]))
        for metric, v, ref in (("mse", v_mse, r_mse), ("mae", v_mae, r_mae)):
            delta = v - ref
            rel = f"{-delta / ref * 100:+.2f}%" if ref != 0 \
                else "ref=0, ratio undefined"
            comp_rows.append([f"{label}-{ref_name}", metric, f"{delta:+.6f}",
                              rel])
        for metric in ("mse", "mae"):
            for level, n_total in (("domain(19)", 19), ("var(190)", 190)):
                improved = degraded = equal = 0
                keys = DOMAIN_NAMES if level.startswith("domain") else all_vars
                for k in keys:
                    if level.startswith("domain"):
                        v = float(np.mean([seed_dom[s][k][metric]
                                           for s in SEEDS]))
                        (ref, _) = ref_dom_tab[k][0 if metric == "mse" else 1]
                    else:
                        v = float(np.mean([seed_merged[s][k][metric]
                                           for s in SEEDS]))
                        (ref, _) = ref_var_tab[k][metric]
                    improved += v < ref
                    degraded += v > ref
                    equal += v == ref
                impr_rows.append([label, ref_name, level, metric, improved,
                                  degraded, equal, n_total])

    if ln and "F0" in ln["overall"]:
        comparison_block("T0", "F0", ln["overall"]["F0"],
                         ln["domain"]["F0"], ln["var"]["F0"])
    if t1_status == "ok":
        comparison_block("T0", "T1", t1["overall"]["T1"],
                         t1["domain"]["T1"], t1["var"]["T1"])

    def w_csv(name, header, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    w_csv("overall.csv", ["method", "seed_stat", "MSE", "MAE"], overall_rows)
    w_csv("domain_summary.csv",
          ["method", "domain", "seed_stat", "MSE", "MAE"], domain_rows)
    w_csv("test_variable_level.csv",
          ["method", "var_key", "n_windows", "std_MSE", "std_MAE",
           "raw_MSE", "raw_MAE"], var_rows)
    w_csv("comparisons.csv",
          ["comparison", "metric", "delta", "relative_improvement"], comp_rows)
    w_csv("improvement_counts.csv",
          ["method", "reference", "level", "metric", "improved", "degraded",
           "equal", "total"], impr_rows)

    t0_overall = {"mse": list(seed_stats([seed_ov[s]["mse"] for s in SEEDS])),
                  "mae": list(seed_stats([seed_ov[s]["mae"] for s in SEEDS]))}
    summary = {
        "protocol_version": CFG["protocol_version"],
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "comparison_status": comparison_status,
        "t1_comparison_status": t1_status,
        "methods_overall": {"T0": t0_overall} | (
            {m: {"mse": list(ln["overall"][m][0]),
                 "mae": list(ln["overall"][m][1])}
             for m in side if m in ln["overall"]} if ln else {}) | (
            {"T1": {"mse": list(t1["overall"]["T1"][0]),
                    "mae": list(t1["overall"]["T1"][1])}}
            if t1_status == "ok" else {}),
        "diverged_disclosure": sel.get("diverged_disclosure", []),
        "ln_sources": (ln["sources"] if ln else {}),
        "t1_sources": (t1["sources"] if t1_status == "ok" else {}),
        "disclosures": {
            "lineage": "T0 = TimeMixer random-init full-param training on the "
                       "native train pool; F0 = VisionTS pretrained + LN "
                       "fine-tuning on the same native pool (LN runs fixed 10 "
                       "epochs, argmin selection, no early stopping, no "
                       "zero-shot fallback in the main result); T1 = same "
                       "TimeMixer protocol on the dense pool.",
            "budget": "T0 per-epoch budget is dense-derived: U_d = "
                      "ceil(D_dense_d/32), S_d = 32*U_d - identical to T1 and "
                      "to LN F0's native runs; sampling draws WITH replacement "
                      "from each variable's native pool. Reported updates are "
                      "actual (early stopping may stop before the 100-epoch "
                      "upper bound).",
            "environment": "T0 runs on theta (RTX 3090, env bound in "
                           "snapshot/protocol.json); F0/T1 ran on eta (L20, "
                           "pinned env). Cross-experiment differences are "
                           "env/GPU lineage, disclosed not normalized."},
        "n_vars": len(all_vars), "n_domains": len(DOMAIN_NAMES)}
    atomic_write_json(summary, os.path.join(RESULTS, "aggregate_summary.json"))
    print(f"[aggregate] ln={comparison_status} t1={t1_status}; overall MSE "
          f"{fmt(*t0_overall['mse'])} MAE {fmt(*t0_overall['mae'])}; "
          f"{len(all_vars)} vars; diverged={len(summary['diverged_disclosure'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AggregateError as e:
        print(f"[aggregate] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
