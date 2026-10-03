#!/usr/bin/env python3
"""Stage aggregate: TN0 tables + TN0-vs-F0 / TN0-vs-T0 / TN0-vs-P0
comparisons (comparison metrics PARSED from the reference runs' results/,
never recomputed).

Aggregation (LN-identical closed forms): window mean -> variable -> merge all
19 domains of one seed [assert no duplicate vars, 190 vars / 19 domains] ->
domain equal-weight -> overall; 3-seed mean +/- std(ddof=1). Raw-scale metrics
only at variable level.

Reference inputs: LN run results/ (--ln-dir; F0 = VisionTS LN-native main
result), T0 results/ (--t0-dir; TimeMixer-native) and P0 results/ (--p0-dir;
PatchTST-native). All three share the writer schema of this framework
(method = column 0). Values like "0.123+/-0.004" parse as (mean, std).
IMPORT VALIDATION: the parsed overall (mean, std) of F0, T0 and P0 must equal
the pre-registered values in CFG["comparison"]["expected_overall_6dp"] at
6-decimal precision (the precision the references were published at) - any
mismatch aborts. Missing references degrade the corresponding block
(comparison_status / t0ref_comparison_status / p0ref_comparison_status)
instead of crashing; re-runs are idempotent.

Also writes: main_table.md (19 domains + Overall x F0/T0/P0/TN0, standardized
MSE/MAE mean+/-std), report.md (disclosures + lr-selection counts + the
candidates that hit the 100-epoch cap), and counts of the selected lrs.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

from common import (CFG, CHECKPOINTS, DOMAIN_NAMES, PREDICTIONS, RESULTS,
                    SEEDS, atomic_write_json, sha256_file)

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


def locate_ref_dir(cli_val, cfg_key):
    cands = []
    if cli_val:
        cands.append(cli_val)
    dflt = CFG["comparison"].get(cfg_key)
    if dflt:
        cands.append(dflt)
    for d in cands:
        if os.path.isfile(os.path.join(d, "overall.csv")):
            return d
        if os.path.isfile(os.path.join(d, "results", "overall.csv")):
            return os.path.join(d, "results")
    return None


def read_results_dir(ref_dir):
    """Parse reference CSVs; returns {overall:{m:(mse,mae)},
    domain:{m:{d:(mse,mae)}}, var:{m:{vk:{...}}}} or None if files missing."""
    def rd(name):
        p = os.path.join(ref_dir, name)
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
        p = os.path.join(ref_dir, name)
        if os.path.isfile(p):
            out["sources"][name] = sha256_file(p)
    return out


def validate_imported_overall(ref, method, expected, what):
    """6-decimal import validation: parsed overall (mean, std) must equal the
    pre-registered published values at the precision they were written."""
    if expected is None:
        return
    got_mse, got_mae = ref["overall"][method]
    for (got, (em, esd)), metric in (
            ((got_mse, expected["mse"]), "mse"),
            ((got_mae, expected["mae"]), "mae")):
        gm, gs = got
        wm, wesd = em, esd
        ok = (abs(round(gm, 6) - wm) < 5e-7
              and (gs is None or abs(round(gs, 6) - wesd) < 5e-7))
        if not ok:
            raise AggregateError(
                f"{what}: parsed overall {metric} ({gm}, {gs}) != "
                f"pre-registered {expected[metric]} at 6-dp precision - "
                f"reference import refused")
    print(f"[aggregate] import validation OK: {what} overall matches the "
          f"pre-registered 6-dp values")


def lr_and_epoch_counts():
    """Selected-lr histogram (57 selected runs) + terminal-state scan over
    ALL candidates (19 domains x 3 lrs x 3 seeds = 171): every checkpoint dir
    must hold exactly one of SUCCESS.json / DIVERGED.json."""
    sel_path = os.path.join(RESULTS, "selection.json")
    with open(sel_path) as f:
        sel = json.load(f)
    lr_counts, sel_max_epoch = {}, []
    for dom, block in sel["domains"].items():
        lr = block["selected_lr"]
        lr_counts["%g" % lr] = lr_counts.get("%g" % lr, 0) + 1
        for seed_s, ent in block["seeds"].items():
            sp = os.path.join(CHECKPOINTS, ent["run_id"], "SUCCESS.json")
            with open(sp) as f:
                info = json.load(f)
            if info.get("stopped_by") == "max_epochs":
                sel_max_epoch.append({"run_id": ent["run_id"],
                                      "epochs_run": info.get("epochs_run"),
                                      "still_improving":
                                          info.get("still_improving_at_max")})

    expected = len(DOMAIN_NAMES) * len(CFG["optim"]["lrs"]) \
        * len(CFG["budget"]["seeds"])
    n_ok = n_div = 0
    max_epoch = []
    diverged = []
    for rid in sorted(os.listdir(CHECKPOINTS)):
        cdir = os.path.join(CHECKPOINTS, rid)
        if not os.path.isdir(cdir):
            continue
        sp, dp = (os.path.join(cdir, "SUCCESS.json"),
                  os.path.join(cdir, "DIVERGED.json"))
        has_s, has_d = os.path.isfile(sp), os.path.isfile(dp)
        if has_s and has_d:
            raise AggregateError(f"{rid}: both SUCCESS.json and DIVERGED.json")
        if not has_s and not has_d:
            raise AggregateError(f"{rid}: no terminal state (expected "
                                 f"{expected} candidates with terminal "
                                 f"states; campaign incomplete?)")
        if has_s:
            with open(sp) as f:
                info = json.load(f)
            n_ok += 1
            if info.get("stopped_by") == "max_epochs":
                max_epoch.append({"run_id": rid,
                                  "epochs_run": info.get("epochs_run"),
                                  "still_improving":
                                      info.get("still_improving_at_max")})
        else:
            with open(dp) as f:
                info = json.load(f)
            n_div += 1
            diverged.append({"run_id": rid,
                             "reason": info.get("reason"),
                             "epoch": info.get("epoch")})
    if n_ok + n_div != expected:
        raise AggregateError(f"campaign terminal states {n_ok + n_div} != "
                             f"expected {expected} candidates")
    return {"lr_selection_counts": lr_counts,
            "max_epoch_candidates": max_epoch,
            "max_epoch_selected_candidates": sel_max_epoch,
            "campaign_terminal_states": {
                "expected_candidates": expected,
                "success": n_ok,
                "diverged": n_div,
                "diverged_detail": diverged}}


def w_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln-dir", default=None,
                    help="LN run dir holding results/ CSVs (default: config)")
    ap.add_argument("--t0-dir", default=None,
                    help="T0 (TimeMixer-native) results dir with overall.csv / "
                         "domain_summary.csv / test_variable_level.csv")
    ap.add_argument("--p0-dir", default=None,
                    help="P0 (PatchTST-native) results dir with the same "
                         "three CSVs")
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

    ln_loc = locate_ref_dir(args.ln_dir, "ln_results_dir_default")
    ln = read_results_dir(ln_loc) if ln_loc is not None else None
    comparison_status = "ok" if (ln and "F0" in ln["overall"]) \
        else "ln_results_unavailable"
    t0_loc = locate_ref_dir(args.t0_dir, "t0_results_dir_default")
    t0 = read_results_dir(t0_loc) if t0_loc is not None else None
    t0_status = "ok" if (t0 and "T0" in t0["overall"]) \
        else "t0_results_unavailable"
    p0_loc = locate_ref_dir(args.p0_dir, "p0_results_dir_default")
    p0 = read_results_dir(p0_loc) if p0_loc is not None else None
    p0_status = "ok" if (p0 and "P0" in p0["overall"]) \
        else "p0_results_unavailable"
    if ln is None:
        print("[aggregate] LN results not found - degrading the TN0-vs-F0 "
              "block (comparison_status=ln_results_unavailable)", flush=True)
    if t0_status != "ok":
        print("[aggregate] T0 results not found/usable - degrading the "
              "TN0-vs-T0 block (t0ref_comparison_status="
              "t0_results_unavailable)", flush=True)
    if p0_status != "ok":
        print("[aggregate] P0 results not found/usable - degrading the "
              "TN0-vs-P0 block (p0ref_comparison_status="
              "p0_results_unavailable)", flush=True)

    exp = CFG["comparison"].get("expected_overall_6dp", {})
    if comparison_status == "ok":
        validate_imported_overall(ln, "F0", exp.get("F0"), "F0 (LN)")
    if t0_status == "ok":
        validate_imported_overall(t0, "T0", exp.get("T0"), "T0")
    if p0_status == "ok":
        validate_imported_overall(p0, "P0", exp.get("P0"), "P0")

    ref_status = (("F0", comparison_status, ln, "LN"),
                  ("T0", t0_status, t0, None),
                  ("P0", p0_status, p0, None))
    refs = {name: ref for name, st, ref, _ in ref_status if st == "ok"}

    # ---------- overall ----------
    overall_rows = [["TN0", "mean+-std over 3 seeds",
                     fmt(*seed_stats([seed_ov[s]["mse"] for s in SEEDS])),
                     fmt(*seed_stats([seed_ov[s]["mae"] for s in SEEDS]))]]
    if "F0" in refs:
        (mse, mse_sd), (mae, mae_sd) = ln["overall"]["F0"]
        overall_rows.append(["F0 (LN)", "as reported",
                             fmt(mse, mse_sd), fmt(mae, mae_sd)])
    if "T0" in refs:
        (mse, mse_sd), (mae, mae_sd) = t0["overall"]["T0"]
        overall_rows.append(["T0", "as reported",
                             fmt(mse, mse_sd), fmt(mae, mae_sd)])
    if "P0" in refs:
        (mse, mse_sd), (mae, mae_sd) = p0["overall"]["P0"]
        overall_rows.append(["P0", "as reported",
                             fmt(mse, mse_sd), fmt(mae, mae_sd)])

    # ---------- domain summary ----------
    domain_rows = []
    for d in DOMAIN_NAMES:
        domain_rows.append(["TN0", d, "mean+-std",
                            fmt(*seed_stats([seed_dom[s][d]["mse"]
                                             for s in SEEDS])),
                            fmt(*seed_stats([seed_dom[s][d]["mae"]
                                             for s in SEEDS]))])
    if "F0" in refs:
        for d in DOMAIN_NAMES:
            (mse, msd), (mae, masd) = ln["domain"]["F0"][d]
            domain_rows.append(["F0 (LN)", d, "as reported",
                                fmt(mse, msd), fmt(mae, masd)])
    if "T0" in refs:
        for d in DOMAIN_NAMES:
            (mse, msd), (mae, masd) = t0["domain"]["T0"][d]
            domain_rows.append(["T0", d, "as reported",
                                fmt(mse, msd), fmt(mae, masd)])
    if "P0" in refs:
        for d in DOMAIN_NAMES:
            (mse, msd), (mae, masd) = p0["domain"]["P0"][d]
            domain_rows.append(["P0", d, "as reported",
                                fmt(mse, msd), fmt(mae, masd)])

    # ---------- variable level ----------
    var_rows = []
    for vk in all_vars:
        per_seed = [seed_merged[s][vk] for s in SEEDS]
        ms = {k: [t[k] for t in per_seed] for k in METRICS}
        var_rows.append(["TN0", vk, per_seed[0]["n"]] +
                        [fmt(*seed_stats(ms[k])) for k in METRICS])
    if "F0" in refs:
        for vk in all_vars:
            t = ln["var"]["F0"].get(vk)
            if t is None:
                continue
            var_rows.append(["F0 (LN)", vk, t["n"]] +
                            [fmt(*t[k]) for k in METRICS])
    if "T0" in refs:
        for vk in all_vars:
            t = t0["var"]["T0"].get(vk)
            if t is None:
                continue
            var_rows.append(["T0", vk, t["n"]] +
                            [fmt(*t[k]) for k in METRICS])
    if "P0" in refs:
        for vk in all_vars:
            t = p0["var"]["P0"].get(vk)
            if t is None:
                continue
            var_rows.append(["P0", vk, t["n"]] +
                            [fmt(*t[k]) for k in METRICS])

    # ---------- TN0 vs F0 (primary) / T0 / P0 comparisons ----------
    comp_rows, impr_rows = [], []

    def comparison_block(ref_name, ref_ov, ref_dom_tab, ref_var_tab):
        (r_mse, _), (r_mae, _) = ref_ov
        v_mse = float(np.mean([seed_ov[s]["mse"] for s in SEEDS]))
        v_mae = float(np.mean([seed_ov[s]["mae"] for s in SEEDS]))
        for metric, v, ref in (("mse", v_mse, r_mse), ("mae", v_mae, r_mae)):
            delta = v - ref
            rel = f"{-delta / ref * 100:+.2f}%" if ref != 0 \
                else "ref=0, ratio undefined"
            comp_rows.append([f"TN0-{ref_name}", metric, f"{delta:+.6f}",
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
                impr_rows.append(["TN0", ref_name, level, metric, improved,
                                  degraded, equal, n_total])

    if "F0" in refs:
        comparison_block("F0", ln["overall"]["F0"],
                         ln["domain"]["F0"], ln["var"]["F0"])
    if "T0" in refs:
        comparison_block("T0", t0["overall"]["T0"],
                         t0["domain"]["T0"], t0["var"]["T0"])
    if "P0" in refs:
        comparison_block("P0", p0["overall"]["P0"],
                         p0["domain"]["P0"], p0["var"]["P0"])

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

    # ---------- main_table.md: 19 domains + Overall x F0/T0/P0/TN0 ----------
    def cell(dom, metric):
        if dom is None:
            vals = [seed_ov[s][metric] for s in SEEDS]
        else:
            vals = [seed_dom[s][dom][metric] for s in SEEDS]
        return fmt(*seed_stats(vals))

    # ref_cells[method][domain_or_"overall"]["mse"|"mae"] -> formatted string
    ref_cells = {}
    for name, ref, src in (("F0", ln, ln), ("T0", t0, t0), ("P0", p0, p0)):
        if name not in refs:
            continue
        ref_cells[name] = {d: {"mse": fmt(*ref["domain"][name][d][0]),
                               "mae": fmt(*ref["domain"][name][d][1])}
                           for d in DOMAIN_NAMES}
        ref_cells[name]["overall"] = {
            "mse": fmt(*ref["overall"][name][0]),
            "mae": fmt(*ref["overall"][name][1])}

    md = ["# TN0 (TimesNet-native) main table - standardized MSE / MAE,"
          " mean+/-std over seeds 2021-2023",
          "",
          "| domain | F0 MSE | T0 MSE | P0 MSE | TN0 MSE "
          "| F0 MAE | T0 MAE | P0 MAE | TN0 MAE |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in DOMAIN_NAMES + [None]:
        name = "Overall (19-domain equal-weight)" if d is None else d
        row = [name]
        for m in ("mse", "mae"):
            for meth in ("F0", "T0", "P0"):
                row.append(ref_cells[meth]["overall" if d is None else d][m]
                           if meth in ref_cells else "n/a")
            row.append(cell(d, m))
        md.append("| " + " | ".join(row) + " |")
    with open(os.path.join(RESULTS, "main_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---------- summary + report ----------
    counts = lr_and_epoch_counts()
    tn0_overall = {"mse": list(seed_stats([seed_ov[s]["mse"] for s in SEEDS])),
                   "mae": list(seed_stats([seed_ov[s]["mae"] for s in SEEDS]))}
    methods_overall = {"TN0": tn0_overall}
    if "F0" in refs:
        methods_overall["F0"] = {"mse": list(ln["overall"]["F0"][0]),
                                 "mae": list(ln["overall"]["F0"][1])}
    if "T0" in refs:
        methods_overall["T0"] = {"mse": list(t0["overall"]["T0"][0]),
                                 "mae": list(t0["overall"]["T0"][1])}
    if "P0" in refs:
        methods_overall["P0"] = {"mse": list(p0["overall"]["P0"][0]),
                                 "mae": list(p0["overall"]["P0"][1])}
    summary = {
        "protocol_version": CFG["protocol_version"],
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "comparison_status": comparison_status,
        "t0ref_comparison_status": t0_status,
        "p0ref_comparison_status": p0_status,
        "methods_overall": methods_overall,
        **counts,
        "diverged_disclosure": sel.get("diverged_disclosure", []),
        "ln_sources": (ln["sources"] if "F0" in refs else {}),
        "t0_sources": (t0["sources"] if "T0" in refs else {}),
        "p0_sources": (p0["sources"] if "P0" in refs else {}),
        "disclosures": {
            "lineage": "TN0 = official thuml TimesNet forecasting path "
                       "(Time-Series-Library models/TimesNet.py, vendored at "
                       "commit 74e58ebf of T-s-cell/Time-Series-Library; "
                       "adaptation = 2 import lines, bitwise-proven equal by "
                       "model_check.py), random init, full-param training of "
                       "all learnable parameters, on the native train pool; "
                       "F0 = VisionTS pretrained + LN fine-tuning on the same "
                       "native pool; T0 = TimeMixer and P0 = PatchTST "
                       "random-init full-param training on the same pool. "
                       "Sampling/evaluation/budget inherited from the "
                       "accepted T0 framework unchanged.",
            "model": "Fixed small-dataset structure held constant across all "
                     "171 candidates: task=long_term_forecast, 96->12, "
                     "e_layers=2, d_model=16, d_ff=32, top_k=5, num_kernels=6, "
                     "dropout=0.1, embed=timeF, freq=h; only the learning "
                     "rate is searched. This is a controlled baseline under "
                     "the project's 96->12 task and 3-lr budget, NOT a claim "
                     "of reproducing every training detail of the original "
                     "paper or its full-size configs; VisionTS uses "
                     "pretraining resources and a training regime TN0 does "
                     "not have, and an equal per-epoch sampling budget is not "
                     "an equal total-compute budget.",
            "eval_batch_1": "Every evaluation forward (epoch-0 diagnostic, "
                            "per-epoch validation, final test, recheck replay) "
                            "runs at batch size 1: TimesNet's FFT_for_Period "
                            "selects top-k periods from the batch-mean "
                            "amplitude spectrum, so any B>1 would couple the "
                            "windows' period selections. Enforced by the "
                            "forward_pred B==1 assertion; training keeps the "
                            "protocol batch of 32. Fixed before any results "
                            "were seen.",
            "label_len_unused": "The vendored forecast path is decoder-free: "
                                "x_dec/x_mark_dec are never referenced and "
                                "label_len is assigned but unused; the model "
                                "is called model(x_enc, None, None, None) "
                                "everywhere.",
            "temporal_embed_unused": "With x_mark=None the timeF temporal "
                                     "embedding (Linear 4->16, bias=False) "
                                     "never enters the forward: its gradient "
                                     "stays None after every backward "
                                     "(proven in preflight B by a positive "
                                     "perturbation test) and Adam skips it; "
                                     "it is counted in the frozen parameter "
                                     "pins and trained-never, disclosed not "
                                     "removed (vendor code unchanged).",
            "environment": "TN0 runs on theta (RTX 3090, env bound in "
                           "snapshot/protocol.json); F0 ran on eta (L20); T0 "
                           "and P0 ran on theta (RTX 3090). Cross-experiment "
                           "differences are env/GPU lineage, disclosed not "
                           "normalized. Overall quality and per-domain "
                           "quality are reported separately."},
        "n_vars": len(all_vars), "n_domains": len(DOMAIN_NAMES)}
    atomic_write_json(summary, os.path.join(RESULTS, "aggregate_summary.json"))

    rep = ["# TN0 (TimesNet-native) results report",
           "",
           f"Generated: {summary['created_utc']}  |  protocol: "
           f"{CFG['protocol_version']}",
           "",
           f"- TN0 overall: std-MSE {fmt(*tn0_overall['mse'])}, std-MAE "
           f"{fmt(*tn0_overall['mae'])} (3-seed mean+/-std, ddof=1)",
           f"- lr selection counts: {counts['lr_selection_counts']}",
           f"- candidates reaching the 100-epoch cap: "
           f"{len(counts['max_epoch_candidates'])}"
           + (f" ({', '.join(c['run_id'] for c in counts['max_epoch_candidates'])}"
              if counts["max_epoch_candidates"] else ""),
           f"- diverged candidates: "
           f"{len(summary['diverged_disclosure'])}",
           f"- comparison status: F0={comparison_status}, T0={t0_status}, "
           f"P0={p0_status}",
           "",
           "See main_table.md for the per-domain table, comparisons.csv / "
           "improvement_counts.csv for TN0-vs-F0/T0/P0 deltas and "
           "win/tie/loss counts, aggregate_summary.json for full-precision "
           "values and disclosures.",
           "",
           "## Disclosure",
           ""]
    rep += [f"- {v}" for v in summary["disclosures"].values()]
    with open(os.path.join(RESULTS, "report.md"), "w") as f:
        f.write("\n".join(rep) + "\n")

    print(f"[aggregate] ln={comparison_status} t0={t0_status} "
          f"p0={p0_status}; overall MSE {fmt(*tn0_overall['mse'])} MAE "
          f"{fmt(*tn0_overall['mae'])}; {len(all_vars)} vars; "
          f"lr={counts['lr_selection_counts']}; "
          f"max-epoch={len(counts['max_epoch_candidates'])}; "
          f"diverged={len(summary['diverged_disclosure'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AggregateError as e:
        print(f"[aggregate] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
