#!/usr/bin/env python3
"""freeze_selection: verify 171 terminal states, re-audit every successful
run's trail, select per-domain lr by the 3-seed mean of best val std-MSE,
and emit results/selection.json with the 57 selected checkpoints' sha256.

Scoring (per user-approved protocol):
- any diverged seed -> (domain, lr) score +inf; NEVER average remaining seeds;
- otherwise arithmetic mean of the 3 seeds' best val std-MSE;
- lowest mean wins; |a-b| <= 1e-8*max(1,min) tie -> smaller lr;
- freeze requires every domain to have >= 1 lr whose 3 seeds all succeeded;
  a domain whose lrs are all +inf is a must-park violation -> abort.

Launch gate: the preflight report must be PASS and its bindings (snapshot /
TSLib / code / env / CUDA) must match the live tree before anything runs.
"""
import datetime
import json
import os
import sys

from common import (CHECKPOINTS, CFG, LOGS, RESULTS, SEEDS, LRS, TIE_TOL,
                    atomic_write_json, launch_gate, sha256_file)
from trainer import select_best_epoch


def fail(msg):
    print(f"FREEZE ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def terminal_state(rid):
    ck = os.path.join(CHECKPOINTS, rid)
    for name, status in (("SUCCESS.json", "success"),
                         ("DIVERGED.json", "diverged")):
        p = os.path.join(ck, name)
        if os.path.isfile(p):
            return status, load_json(p)
    return None, None


def lr_score(best_vals, diverged):
    """(domain, lr) selection score: +inf if ANY seed diverged (never average
    remaining seeds), else arithmetic mean of the 3 seeds' best val."""
    if diverged:
        return float("inf")
    return sum(best_vals) / len(best_vals)


def pick_lr(scores):
    """scores: {lr: score} over ascending LRS. Lowest mean wins;
    |a-b| <= 1e-8*max(1,min) tie -> smaller lr (strict-improve rule over the
    ascending lr order). +inf scores (diverged seeds) are always replaceable
    by any finite score."""
    import math
    best_lr = None
    best_v = None
    for lr in LRS:
        v = scores[lr]
        if best_v is None or math.isinf(best_v) or \
                v < best_v - TIE_TOL * max(1.0, best_v):
            best_lr, best_v = lr, v
    return best_lr, best_v


def require_available(scores, dom):
    if all(v == float("inf") for v in scores.values()):
        fail(f"{dom}: every lr has a diverged seed - domain unavailable, "
             f"must-park condition hit (never waived)")


def jsonl_budget_ok(rid, u_d):
    """Max-attempt dedup over the jsonl trail; completed epochs must carry
    exactly U_d updates and one epoch_end row consistent with ep{E}.val.json."""
    path = os.path.join(LOGS, f"{rid}.jsonl")
    if not os.path.isfile(path):
        return False, "no jsonl trail"
    best = {}
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("type"), row.get("epoch"), row.get("update"))
            if key not in best or row.get("attempt", -1) >= \
                    best[key].get("attempt", -1):
                best[key] = row
    updates = [r for (t, e, u), r in best.items() if t == "update"]
    ends = {r["epoch"]: r for (t, e, u), r in best.items() if t == "epoch_end"}
    by_epoch = {}
    for r in updates:
        by_epoch.setdefault(r["epoch"], []).append(r)
    for e, rows in by_epoch.items():
        if len(rows) != u_d:
            return False, f"epoch {e}: {len(rows)} updates != U_d {u_d}"
        if len({r["update"] for r in rows}) != u_d:
            return False, f"epoch {e}: duplicate update indices"
    for e, er in ends.items():
        if er.get("updates") != u_d or len(by_epoch.get(e, [])) != u_d:
            return False, f"epoch {e}: epoch_end trail mismatch"
        vp = os.path.join(CHECKPOINTS, rid, f"ep{e:03d}.val.json")
        if not os.path.isfile(vp):
            return False, f"epoch {e}: missing ep val json"
        vj = load_json(vp)
        if vj["val_mse"] != er["val_mse"] or vj["attempt"] != er["attempt"]:
            return False, f"epoch {e}: val json != epoch_end row"
        if not er.get("slot_hash"):
            return False, f"epoch {e}: missing slot_hash"
    return True, {"epochs": sorted(ends), "updates_total": len(updates),
                  "n_epoch_end": len(ends)}


def main():
    launch_gate("freeze")
    domains = [r["domain"] for r in CFG["budget"]["expected_domain_table"]]
    u_d = {r["domain"]: -(-r["dense"] // CFG["optim"]["batch"]) for r in
           CFG["budget"]["expected_domain_table"]}
    n_expected = len(domains) * len(LRS) * len(SEEDS)
    states = {}
    missing, nonterminal = [], []
    for dom in domains:
        for lr in LRS:
            for seed in SEEDS:
                rid = f"{dom}__lr{lr:g}__s{seed}"
                st, rec = terminal_state(rid)
                states[rid] = (dom, lr, seed, st, rec)
                if st is None:
                    if os.path.isdir(os.path.join(CHECKPOINTS, rid)):
                        nonterminal.append(rid)
                    else:
                        missing.append(rid)
    if missing or nonterminal:
        fail(f"{len(missing)} runs never started {missing[:5]}... and "
             f"{len(nonterminal)} runs lack a terminal state "
             f"{nonterminal[:5]}... - freeze refused")

    n_div = sum(1 for v in states.values() if v[3] == "diverged")
    print(f"[freeze] 171 terminal states: {n_expected - n_div} success, "
          f"{n_div} diverged")

    audited = {}
    for rid, (dom, lr, seed, st, rec) in states.items():
        if st != "success":
            continue
        ok, detail = jsonl_budget_ok(rid, u_d[dom])
        if not ok:
            fail(f"{rid}: trail audit failed: {detail}")
        sel = select_best_epoch({e: load_json(os.path.join(
            CHECKPOINTS, rid, f"ep{e:03d}.val.json"))["val_mse"]
            for e in detail["epochs"]})
        if sel["epoch"] != rec["frozen_best_epoch"] or \
                abs(sel["value"] - rec["frozen_best_val"]) > 0:
            fail(f"{rid}: re-audited selection {sel} != recorded "
                 f"({rec['frozen_best_epoch']}, {rec['frozen_best_val']})")
        audited[rid] = {"best_epoch": sel["epoch"], "best_val": sel["value"],
                        "updates_total": detail["updates_total"]}
    print(f"[freeze] trail re-audit clean for {len(audited)} successful runs")

    domain_blocks = {}
    for dom in domains:
        scores = {}
        diverged = []
        for lr in LRS:
            vals, this_div = [], []
            for seed in SEEDS:
                rid = f"{dom}__lr{lr:g}__s{seed}"
                _, _, _, st, rec = states[rid]
                if st == "diverged":
                    this_div.append({"seed": seed, "run_id": rid,
                                     "reason": rec.get("reason"),
                                     "epoch": rec.get("epoch"),
                                     "update": rec.get("update")})
                else:
                    vals.append(audited[rid]["best_val"])
            scores[str(lr)] = lr_score(vals, this_div)
            if this_div:
                diverged.extend(this_div)
        require_available(scores, dom)
        best_lr, _ = pick_lr({float(k): v for k, v in scores.items()})
        seeds_block = {}
        for seed in SEEDS:
            rid = f"{dom}__lr{best_lr:g}__s{seed}"
            _, _, _, st, rec = states[rid]
            ep = rec["frozen_best_epoch"]
            ck = os.path.join(CHECKPOINTS, rid, f"ep{ep:03d}.pt")
            if not os.path.isfile(ck):
                fail(f"{rid}: selected checkpoint ep{ep:03d}.pt missing")
            seeds_block[str(seed)] = {
                "run_id": rid, "epoch": ep, "val_mse": rec["frozen_best_val"],
                "ckpt": os.path.relpath(ck, os.path.dirname(CHECKPOINTS)),
                "ckpt_sha256": sha256_file(ck)}
        domain_blocks[dom] = {"selected_lr": best_lr, "scores": scores,
                              "diverged": diverged, "seeds": seeds_block}
        inf_str = {k: ("+inf" if v == float("inf") else f"{v:.6f}")
                   for k, v in scores.items()}
        print(f"[freeze] {dom}: lr {best_lr:g} (scores {inf_str})")

    out = {"frozen": True,
           "created_utc": datetime.datetime.now(datetime.timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "protocol_version": CFG["protocol_version"],
           "n_terminal": n_expected, "n_diverged": n_div,
           "diverged_disclosure": [
               {"run_id": rid, "domain": dom, "lr": lr, "seed": seed,
                "reason": rec.get("reason"), "epoch": rec.get("epoch"),
                "update": rec.get("update")}
               for rid, (dom, lr, seed, st, rec) in sorted(states.items())
               if st == "diverged"],
           "domains": domain_blocks}
    atomic_write_json(out, os.path.join(RESULTS, "selection.json"))
    n_ckpt = sum(len(b["seeds"]) for b in domain_blocks.values())
    print(f"[freeze] selection.json written: {len(domain_blocks)} domains, "
          f"{n_ckpt} checkpoints pinned by sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
