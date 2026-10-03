#!/usr/bin/env python3
"""Serial queue over the 171 candidates (19 domains x 3 lrs x 3 seeds).

Order: domains ascending by dense total (ties by name) -> seed -> lr
(1e-4, 1e-3, 1e-2). Small domains complete their full loop first so
end-to-end problems surface within minutes; lr=1e-2 sits last in each triple.

- Launch gate: audit/preflight_report.json must be PASS and its bindings
  (snapshot / TSLib / code / env / CUDA) must match the live tree.
- GPU gate before EVERY candidate: free VRAM on TN0_GPU_UUID >= preflight peak
  + 3 GiB margin; PAUSE sentinel honored; never touches other processes.
- 3-way fingerprint binding (snapshot / TSLib / code) at queue start and
  re-checked before every candidate.
- DivergenceError -> DIVERGED.json evidence + queue CONTINUES (per protocol).
- Every lr of a domain with >=1 diverged seed -> results/MUST_PARK.json,
  exit 4 (must-park, never waived; a restart refuses while the file exists).
- OOM / data / code errors -> scene preserved, manifest failed, exit 3.
- Resume: SUCCESS.json / DIVERGED.json short-circuit inside run_training.
"""
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

import torch

from common import (CFG, CHECKPOINTS, DivergenceError, LRS, LOGS,
                    PAUSE_SENTINEL, RESULTS, SNAPSHOT_PROTOCOL,
                    actual_fingerprints, append_jsonl, atomic_write_json,
                    binding_mismatches, launch_gate, run_id)
from data import DenseData
from trainer import run_training

VRAM_MARGIN_MB = CFG["resources"]["vram_margin_mb"]
MUST_PARK_JSON = os.path.join(RESULTS, "MUST_PARK.json")


def fail(msg, code=3):
    print(f"QUEUE FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def queue_order(data):
    rows = sorted(((data.domain_dense_total(d), d) for d in data.domain_vars))
    order = []
    for _, dom in rows:
        for seed in CFG["budget"]["seeds"]:
            for lr in CFG["optim"]["lrs"]:
                order.append((dom, lr, seed))
    return order


def gpu_uuid():
    uuid = os.environ.get("TN0_GPU_UUID", "").strip()
    if uuid:
        out = subprocess.run(["nvidia-smi", f"--id={uuid}",
                              "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            fail(f"TN0_GPU_UUID {uuid} not visible to nvidia-smi: "
                 f"{out.stderr.strip()}")
        return uuid
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid",
                          "--format=csv,noheader"], capture_output=True,
                         text=True)
    gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if len(gpus) != 1:
        fail(f"TN0_GPU_UUID unset and {len(gpus)} GPUs visible - refusing to "
             f"guess; set TN0_GPU_UUID")
    print(f"[queue] TN0_GPU_UUID unset; single GPU {gpus[0]}")
    return gpus[0]


def free_vram_mb(uuid):
    out = subprocess.run(["nvidia-smi", f"--id={uuid}",
                          "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[0])


def gpu_gate(uuid, need_mb, rid):
    """Block until free VRAM >= need_mb; PAUSE sentinel = manual hold."""
    while True:
        if os.path.isfile(PAUSE_SENTINEL):
            print(f"[queue] PAUSE sentinel present - holding before {rid} "
                  f"(remove {PAUSE_SENTINEL} to resume)", flush=True)
            time.sleep(30)
            continue
        free = free_vram_mb(uuid)
        if free >= need_mb:
            return
        print(f"[queue] {rid}: free VRAM {free:.0f} MiB < need {need_mb:.0f} "
              f"MiB - waiting 60s (co-tenant jobs untouched)", flush=True)
        time.sleep(60)


def load_preflight_peak():
    rep = launch_gate("train")
    peak = rep.get("sections", {}).get("I_throughput", {}) \
        .get("peak_vram_mb")
    if not isinstance(peak, (int, float)) or peak <= 0:
        fail("preflight report has no measured peak_vram_mb - train refused")
    return float(peak)


def domain_dead(dom, div_by_lr):
    """True when every lr of the domain already has >=1 diverged seed (any
    diverged seed voids that lr for selection, so the domain is unavailable
    and the queue must park)."""
    d = div_by_lr.get(dom, {})
    return all(d.get(lr) for lr in LRS)


def park(dom, div_by_lr, manifest, manifest_path):
    detail = {"type": "must_park", "domain": dom,
              "reason": "every lr has >=1 diverged seed - domain unavailable "
                        "(must-park, never waived)",
              "lrs": {"%g" % lr: {"diverged_seeds":
                                  sorted(div_by_lr[dom][lr])}
                      for lr in LRS if div_by_lr[dom].get(lr)},
              "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ")}
    atomic_write_json(detail, MUST_PARK_JSON)
    manifest["parked"] = detail
    atomic_write_json(manifest, manifest_path)
    append_jsonl(os.path.join(LOGS, "queue_events.jsonl"),
                 {"type": "must_park", **detail})
    fail(f"{dom}: every lr has a diverged seed - MUST PARK (queue stopped, "
         f"see {MUST_PARK_JSON})", code=4)


def write_diverged(rid, ckpt_dir, exc, t0, dom, lr, seed):
    ev = {"run_id": rid, "status": "diverged", "domain": dom, "lr": lr,
          "seed": seed, "reason": str(exc), "epoch": getattr(exc, "epoch",
                                                            None),
          "update": getattr(exc, "update", None),
          "wall_s": time.time() - t0,
          "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
          .strftime("%Y-%m-%dT%H:%M:%SZ")}
    atomic_write_json(ev, os.path.join(ckpt_dir, "DIVERGED.json"))
    append_jsonl(os.path.join(LOGS, "queue_events.jsonl"),
                 {"type": "diverged", **ev})
    print(f"[queue] {rid} DIVERGED at epoch={ev['epoch']} "
          f"update={ev['update']}: {str(exc)[:160]}", flush=True)


def main():
    if not torch.cuda.is_available():
        fail("CUDA not available - no CPU fallback per protocol")
    peak = load_preflight_peak()
    if os.path.isfile(MUST_PARK_JSON):
        fail("results/MUST_PARK.json exists - a domain is unavailable "
             "(must-park); human decision required, inspect and remove the "
             "file only after deciding", code=4)
    uuid = gpu_uuid()
    device = "cuda"
    need_mb = peak + VRAM_MARGIN_MB

    if not os.path.isfile(SNAPSHOT_PROTOCOL):
        fail("snapshot/protocol.json missing - run prepare first")
    with open(SNAPSHOT_PROTOCOL) as f:
        proto = json.load(f)
    act = actual_fingerprints()
    bad = binding_mismatches(proto["bindings"], act)
    if bad:
        fail(f"environment drift vs frozen protocol in {bad} - re-run "
             f"prepare+preflight; refusing to train")
    print(f"[queue] fingerprint binding OK (peak {peak:.0f} MiB + margin "
          f"{VRAM_MARGIN_MB} MiB = gate {need_mb:.0f} MiB)", flush=True)

    data = DenseData()
    order = queue_order(data)
    manifest_path = os.path.join(RESULTS, "run_manifest.json")
    manifest = {"started_utc": datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "order": [run_id(d, lr, s) for d, lr, s in order],
                "completed": [], "diverged": [], "failed": [],
                "peak_vram_gate_mb": need_mb, "gpu_uuid": uuid}
    n_done = 0
    div_by_lr = {}
    for dom, lr, seed in order:
        rid = run_id(dom, lr, seed)
        n_done += 1
        bad = binding_mismatches(proto["bindings"], actual_fingerprints())
        if bad:
            fail(f"drift detected before {rid}: {bad} - scene preserved")
        gpu_gate(uuid, need_mb, rid)
        print(f"[queue] ({n_done}/{len(order)}) {rid} starting", flush=True)
        t0 = time.time()
        try:
            info = run_training(dom, lr, seed, data, device)
        except SystemExit:
            raise
        except DivergenceError as exc:
            write_diverged(rid, os.path.join(CHECKPOINTS, rid), exc, t0,
                           dom, lr, seed)
            manifest["diverged"].append(rid)
            div_by_lr.setdefault(dom, {}).setdefault(lr, set()).add(seed)
        except Exception as exc:  # noqa: BLE001 - scene preserved, exit 3
            tb = traceback.format_exc()
            append_jsonl(os.path.join(LOGS, "queue_events.jsonl"),
                         {"type": "failed", "run_id": rid, "domain": dom,
                          "lr": lr, "seed": seed, "error": repr(exc),
                          "traceback": tb, "wall_s": time.time() - t0})
            manifest["failed"].append({"run_id": rid, "error": repr(exc)})
            atomic_write_json(manifest, manifest_path)
            fail(f"{rid}: non-divergence failure ({type(exc).__name__}: "
                 f"{exc}) - queue stopped with scene preserved")
        else:
            if info.get("skipped"):
                term = info.get("terminal")
                print(f"[queue] {rid} skipped (terminal {term})", flush=True)
                if term == "success":
                    manifest["completed"].append(rid)
                else:
                    manifest["diverged"].append(rid)
                    div_by_lr.setdefault(dom, {}).setdefault(lr, set()) \
                        .add(seed)
            elif info.get("status") == "success":
                manifest["completed"].append(rid)
                print(f"[queue] {rid} SUCCESS "
                      f"best={info['local_best_val_mse']:.6f}"
                      f"@{info['local_best_epoch']} "
                      f"({info['stopped_by']}, {info['wall_s']:.0f}s)",
                      flush=True)
            else:
                fail(f"{rid}: unexpected run_training result {info}")
        manifest["last_update_utc"] = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        atomic_write_json(manifest, manifest_path)
        if domain_dead(dom, div_by_lr):
            park(dom, div_by_lr, manifest, manifest_path)

    n_div = len(manifest["diverged"])
    print(f"[queue] DONE: {len(manifest['completed'])} success, "
          f"{n_div} diverged, {len(manifest['failed'])} failed", flush=True)
    if n_div:
        print("[queue] diverged candidates (selection score +inf, full "
              "disclosure in report):", flush=True)
        for rid in manifest["diverged"]:
            print(f"  - {rid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
