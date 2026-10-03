#!/usr/bin/env python3
"""Candidate training for TN0 (TimesNet-native) - one run per invocation.

- Sampling: native-pool clone of the LN run's build_slots F0 branch (identical
  RNG families and stream consumption; slots hold native sample_ids resolved
  via sample_start_idx; epochs 1..10 reproduce LN F0 order for the same
  (domain, seed); per-epoch budget stays dense-derived: U_d = ceil(D_d/32));
  per-epoch slot hash recorded in the jsonl trail.
- Loss/metrics: d_i = pstdev(raw 96-step input, ddof=0) with frozen fallback,
  no gradient through d_i; float64 evaluation.
- Early stopping: best val std-MSE from epoch 1 on, improve = v < best -
  1e-8*max(1,best); patience 15 completed epochs; epoch 0 is diagnostic only.
- Resume: resume.pt carries model/optimizer/epoch/best/patience plus the four
  RNG states (python/numpy/torch/cuda); hashes+lr bound, mismatch refuses.
  A resume whose replayed patience is already exhausted finalizes the run
  immediately (never trains extra epochs after early stop).
- Divergence: any NaN/Inf in loss/grad/params/val raises DivergenceError
  (candidate terminal DIVERGED); OOM/data/code errors propagate as-is.
"""
import hashlib
import json
import os
import random
import time

import numpy as np

from common import (CHECKPOINTS, CFG, DOMAIN_IDX, DivergenceError, EPOCHS,
                    EVAL_BATCH, LOGS, PATIENCE, SNAPSHOT_PROTOCOL, TIE_TOL,
                    actual_fingerprints, append_jsonl, atomic_torch_save,
                    atomic_write_json, binding_mismatches,
                    environment_snapshot, run_id, u_d_s_d)
from data import window_d
from model_adapter import forward_pred, forward_train, load_model


def seed_everything(seed, dom_idx):
    """Init/aux seeds depend only on (seed, domain) - identical initial
    weights and sampling streams across lrs for the same (domain, seed)."""
    import torch
    init_seed = int(np.random.default_rng(
        [seed, 1100 + dom_idx]).integers(0, 2**31 - 1))
    aux_seed = int(np.random.default_rng(
        [seed, 1200 + dom_idx]).integers(0, 2**31 - 1))
    random.seed(aux_seed)
    np.random.seed(aux_seed % (2**32))
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    return {"init_seed": init_seed, "aux_seed": aux_seed}


def build_slots(data, domain, seed, epoch):
    """Variable-balanced native slots; deterministic per (seed, epoch).
    Line-level clone of LN trainer.build_slots (F0 native branch): same RNG
    families, same consumption order, so epochs 1..10 match LN F0. Slots hold
    native train sample_ids (with replacement); the epoch budget stays
    dense-derived (U_d = ceil(D_d/32)) - never shrunk to the native pool."""
    dom_idx = DOMAIN_IDX[domain]
    vars_ = data.domain_vars[domain]
    n_vars = len(vars_)
    U, S = u_d_s_d(data.domain_dense_total(domain))
    per, rem = S // n_vars, S % n_vars
    extra = set(np.random.default_rng(
        [seed, 3000 + dom_idx, epoch]).permutation(n_vars)[:rem].tolist())
    rng_base = np.random.default_rng([seed, 9000 + dom_idx, epoch])
    slots = []
    for i, vk in enumerate(vars_):
        k = per + (1 if i in extra else 0)
        ids = data.native_ids(vk, "train")
        pick = rng_base.integers(0, len(ids), size=k)
        slots.extend((vk, ids[j]) for j in pick)
    rng_shuf = np.random.default_rng([seed, 7000 + dom_idx, epoch])
    order = rng_shuf.permutation(len(slots))
    return [slots[j] for j in order], U, S


def slot_hash(slots):
    h = hashlib.sha256()
    for vk, idx in slots:
        h.update(f"{vk}|{idx}\x00".encode())
    return h.hexdigest()


def select_best_epoch(val_by_epoch, tie_tol=None):
    """Earliest-epoch argmin over epochs >= 1 with relative tie tolerance."""
    if tie_tol is None:
        tie_tol = TIE_TOL
    candidates = sorted(e for e in val_by_epoch if e >= 1)
    if not candidates:
        raise ValueError("no selectable epoch (>=1) in val history")
    best_e = candidates[0]
    best_v = val_by_epoch[best_e]
    for e in candidates[1:]:
        v = val_by_epoch[e]
        if v < best_v - tie_tol * max(1.0, best_v):
            best_e, best_v = e, v
    return {"epoch": best_e, "value": best_v}


def evaluate_split(model, data, domain, split, device):
    """Full native windows; eval+no_grad, float64, shuffle=False,
    drop_last=False (tail batch evaluated too). Var-internal mean then
    var-equal mean. Restores train mode afterwards."""
    import torch
    rows = data.split_rows(domain, split)
    per_var = {}
    n_done = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(rows), EVAL_BATCH):
            chunk = rows[i0:i0 + EVAL_BATCH]
            xt = torch.tensor(np.stack([r[2] for r in chunk]),
                              dtype=torch.float32,
                              device=device).unsqueeze(-1)
            pred = forward_pred(model, xt).detach().cpu().numpy().astype(
                np.float64)
            for j, (vk, sid, x, y) in enumerate(chunk):
                d = window_d(x, data.fb_std[vk])
                e = (pred[j] - y) / d
                mse = float(np.mean(e ** 2))
                mae = float(np.mean(np.abs(e)))
                pv = per_var.setdefault(vk, {"mse": 0.0, "mae": 0.0, "n": 0})
                pv["mse"] += mse
                pv["mae"] += mae
                pv["n"] += 1
                n_done += 1
    if was_training:
        model.train()
    for pv in per_var.values():
        pv["mse"] /= pv["n"]
        pv["mae"] /= pv["n"]
    val_mse = float(np.mean([pv["mse"] for pv in per_var.values()]))
    val_mae = float(np.mean([pv["mae"] for pv in per_var.values()]))
    if not (np.isfinite(val_mse) and np.isfinite(val_mae)):
        raise DivergenceError(
            f"non-finite {split} metric (mse={val_mse}, mae={val_mae})")
    expected = data.domain_native_total(domain, split)
    if n_done != expected:
        raise RuntimeError(
            f"{domain}/{split}: evaluated {n_done} windows != {expected}")
    return {"split": split, "domain": domain, "n_windows": n_done,
            "val_mse": val_mse, "val_mae": val_mae, "per_var": per_var}


def _max_attempt(log_path):
    mx = -1
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = r.get("attempt")
                if isinstance(a, int):
                    mx = max(mx, a)
    return mx


def _replay_state(ckpt_dir, upto_epoch):
    """Rebuild best/patience/totals from the authoritative per-epoch val
    files (epochs 1..upto_epoch)."""
    best = None
    best_epoch = None
    patience_cnt = 0
    updates_total = 0
    presented_total = 0
    val_by_epoch = {}
    for e in range(1, upto_epoch + 1):
        p = os.path.join(ckpt_dir, f"ep{e:03d}.val.json")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing {os.path.basename(p)} while replaying")
        with open(p) as f:
            rec = json.load(f)
        v = rec["val_mse"]
        val_by_epoch[e] = v
        updates_total += int(rec.get("updates", 0))
        presented_total += int(rec.get("presented", 0))
        if best is None or v < best - TIE_TOL * max(1.0, best):
            best, best_epoch = v, e
            patience_cnt = 0
        else:
            patience_cnt += 1
    return {"val_by_epoch": val_by_epoch, "best": best,
            "best_epoch": best_epoch, "patience_cnt": patience_cnt,
            "updates_total": updates_total, "presented_total": presented_total}


def _get_rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)}


def _set_rng(st, device):
    import torch
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    elif st["cuda"] is None and device != "cpu":
        raise RuntimeError(
            "resume checkpoint has no CUDA RNG state but CUDA is requested")


def _to_cpu(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _to_cpu(v)
        elif hasattr(v, "detach"):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def run_training(domain, lr, seed, data, device):
    """One candidate run; returns info dict. Terminal states: SUCCESS.json
    (success) / DIVERGED.json (diverged) - both short-circuit re-invocation."""
    import torch
    rid = run_id(domain, lr, seed)
    ckpt_dir = os.path.join(CHECKPOINTS, rid)
    os.makedirs(ckpt_dir, exist_ok=True)
    success_path = os.path.join(ckpt_dir, "SUCCESS.json")
    diverged_path = os.path.join(ckpt_dir, "DIVERGED.json")
    if os.path.isfile(success_path):
        with open(success_path) as f:
            return {**json.load(f), "run_id": rid, "status": "skipped",
                    "skipped": True, "terminal": "success"}
    if os.path.isfile(diverged_path):
        with open(diverged_path) as f:
            return {**json.load(f), "run_id": rid, "status": "skipped",
                    "skipped": True, "terminal": "diverged"}

    t0 = time.time()
    seed_everything(seed, DOMAIN_IDX[domain])
    model = load_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           betas=tuple(CFG["optim"]["betas"]),
                           eps=1e-8, weight_decay=0.0)
    U_d, S_d = u_d_s_d(data.domain_dense_total(domain))

    if not os.path.isfile(SNAPSHOT_PROTOCOL):
        raise RuntimeError("snapshot/protocol.json missing - run prepare first")
    with open(SNAPSHOT_PROTOCOL) as f:
        proto = json.load(f)
    act = actual_fingerprints()
    bad = binding_mismatches(proto["bindings"], act)
    if bad:
        raise RuntimeError(
            f"{rid}: environment drift vs frozen protocol in {bad} - re-run "
            f"prepare+preflight; refusing to train")
    want = {**proto["bindings"], "lr": float(lr)}

    log_path = os.path.join(LOGS, f"{rid}.jsonl")
    jsonl_max = _max_attempt(log_path)
    resume_path = os.path.join(ckpt_dir, "resume.pt")

    # epoch 0: random-model diagnostic, never selectable
    ep0_json = os.path.join(ckpt_dir, "ep000.val.json")
    if not os.path.isfile(ep0_json):
        val0 = evaluate_split(model, data, domain, "val", device)
        atomic_torch_save(_to_cpu(model.state_dict()),
                          os.path.join(ckpt_dir, "ep000.pt"))
        atomic_write_json({"epoch": 0, "attempt": 0, "selectable": False,
                           "val_mse": val0["val_mse"],
                           "val_mae": val0["val_mae"]}, ep0_json)

    start_epoch = 1
    finalize_only = False
    if os.path.isfile(resume_path):
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        if rs["hashes"] != want or rs.get("versions") != environment_snapshot():
            diff = [k for k in want if rs["hashes"].get(k) != want[k]]
            raise RuntimeError(
                f"{rid}: resume refused (hash/version mismatch in {diff}) - "
                f"manual inspection required, not silently restarted")
        model.load_state_dict(rs["model_state"], strict=True)
        opt.load_state_dict(rs["opt_state"])
        _set_rng(rs["rng"], device)
        rep = _replay_state(ckpt_dir, rs["epoch_completed"])
        for k in ("best", "best_epoch", "patience_cnt"):
            if rep[k] != rs.get(k):
                raise RuntimeError(
                    f"{rid}: resume {k}={rs.get(k)} != replay {rep[k]}")
        best, best_epoch, patience_cnt = rep["best"], rep["best_epoch"], \
            rep["patience_cnt"]
        updates_total, presented_total = rep["updates_total"], \
            rep["presented_total"]
        start_epoch = rs["epoch_completed"] + 1
        attempt = max(rs["attempt"] + 1, jsonl_max + 1)
        if rep["patience_cnt"] >= PATIENCE:
            # early stop was already reached before the interrupt - finalize
            # from the recorded best, never train extra epochs
            finalize_only = True
        print(f"[{rid}] resume at epoch {start_epoch} "
              f"(prev attempt {rs['attempt']}, new attempt {attempt})",
              flush=True)
    else:
        attempt = jsonl_max + 1
        best = None
        best_epoch = None
        patience_cnt = 0
        updates_total = 0
        presented_total = 0

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)

    stopped_early = False
    last_epoch = start_epoch - 1
    if finalize_only or start_epoch > EPOCHS:
        why = "patience already exhausted" if finalize_only else \
            f"all {EPOCHS} epochs complete"
        print(f"[{rid}] {why}; finalizing without further epochs "
              f"(attempt {attempt})", flush=True)
    else:
        for epoch in range(start_epoch, EPOCHS + 1):
            last_epoch = epoch
            slots, U, S = build_slots(data, domain, seed, epoch)
            if (U, S) != (U_d, S_d):
                raise RuntimeError(f"{rid}: budget {(U, S)} != {(U_d, S_d)}")
            sh = slot_hash(slots)
            model.train()
            ep_pred, ep_gn = [], []
            for u in range(U):
                chunk = slots[u * 32:(u + 1) * 32]
                opt.zero_grad(set_to_none=True)
                xs, ys, ds = [], [], []
                for vk, s0 in chunk:
                    x, y = data.split_window_by_id(vk, s0)
                    xs.append(x)
                    ys.append(y)
                    ds.append(window_d(x, data.fb_std[vk]))
                xt = torch.tensor(np.stack(xs), dtype=torch.float32,
                                  device=device).unsqueeze(-1)
                yt = torch.tensor(np.stack(ys), dtype=torch.float32,
                                  device=device)
                dt = torch.tensor(np.stack(ds), dtype=torch.float32,
                                  device=device).unsqueeze(1)
                try:
                    # training forward: protocol batch of 32, any-B entry
                    # (FFT top-k from the batch-mean spectrum is intended
                    # here; evaluation is the B==1 forward_pred path)
                    pred = forward_train(model, xt)
                except DivergenceError as exc:
                    raise DivergenceError(
                        f"{rid} epoch {epoch} update {u}: {exc}",
                        epoch=epoch, update=u) from exc
                loss = torch.mean(((pred - yt) / dt) ** 2)
                if not bool(torch.isfinite(loss)):
                    raise DivergenceError(
                        f"{rid} epoch {epoch} update {u}: non-finite loss",
                        epoch=epoch, update=u)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0))
                post = float(torch.sqrt(sum(
                    (p.grad ** 2).sum() for p in model.parameters()
                    if p.grad is not None)))
                n_grad = sum(1 for p in model.parameters()
                             if p.grad is not None)
                if not np.isfinite(gnorm):
                    raise DivergenceError(
                        f"{rid} epoch {epoch} update {u}: non-finite grad "
                        f"norm {gnorm}", epoch=epoch, update=u)
                opt.step()
                for p in model.parameters():
                    if not bool(torch.isfinite(p.detach()).all()):
                        raise DivergenceError(
                            f"{rid} epoch {epoch} update {u}: non-finite "
                            f"parameter after step", epoch=epoch, update=u)
                ep_pred.append(float(loss.detach()))
                ep_gn.append(gnorm)
                updates_total += 1
                append_jsonl(log_path, {
                    "type": "update", "attempt": attempt, "epoch": epoch,
                    "update": u, "pred_loss": float(loss.detach()),
                    "grad_norm_pre_clip": gnorm, "grad_norm_post_clip": post,
                    "n_grad_tensors": n_grad})
            presented_total += len(slots)

            try:
                val = evaluate_split(model, data, domain, "val", device)
            except DivergenceError as exc:
                raise DivergenceError(f"{rid} epoch {epoch} val: {exc}",
                                      epoch=epoch, update=None) from exc
            improved = best is None or val["val_mse"] < best - TIE_TOL * max(
                1.0, best)
            if improved:
                best, best_epoch = val["val_mse"], epoch
                patience_cnt = 0
            else:
                patience_cnt += 1

            atomic_torch_save(_to_cpu(model.state_dict()),
                              os.path.join(ckpt_dir, f"ep{epoch:03d}.pt"))
            atomic_write_json({
                "epoch": epoch, "attempt": attempt, "selectable": True,
                "slot_hash": sh,
                "pred_loss_mean": float(np.mean(ep_pred)),
                "grad_norm_mean": float(np.mean(ep_gn)),
                "presented": len(slots), "updates": U,
                "val_mse": val["val_mse"], "val_mae": val["val_mae"],
                "best_epoch": best_epoch, "best_val": best,
                "patience_cnt": patience_cnt},
                os.path.join(ckpt_dir, f"ep{epoch:03d}.val.json"))
            append_jsonl(log_path, {
                "type": "epoch_end", "attempt": attempt, "epoch": epoch,
                "slot_hash": sh,
                "pred_loss_mean": float(np.mean(ep_pred)),
                "grad_norm_mean": float(np.mean(ep_gn)),
                "presented": len(slots), "updates": U,
                "val_mse": val["val_mse"], "val_mae": val["val_mae"],
                "best_epoch": best_epoch, "best_val": best,
                "patience_cnt": patience_cnt})

            rs_state = {"model_state": _to_cpu(model.state_dict()),
                        "opt_state": _to_cpu(opt.state_dict()),
                        "epoch_completed": epoch, "attempt": attempt,
                        "best": best, "best_epoch": best_epoch,
                        "patience_cnt": patience_cnt,
                        "updates_total": updates_total,
                        "presented_total": presented_total,
                        "hashes": want, "versions": environment_snapshot(),
                        "rng": _get_rng()}
            atomic_torch_save(rs_state, resume_path)
            print(f"[{rid}] epoch {epoch}: val_mse={val['val_mse']:.6f} "
                  f"loss={float(np.mean(ep_pred)):.6f} "
                  f"best={best:.6f}@{best_epoch} patience={patience_cnt} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if patience_cnt >= PATIENCE:
                stopped_early = True
                break

    rep = _replay_state(ckpt_dir, last_epoch)
    sel = select_best_epoch(rep["val_by_epoch"])
    if sel["epoch"] != rep["best_epoch"] or sel["value"] != rep["best"]:
        raise RuntimeError(f"{rid}: selector {sel} != replayed best "
                           f"({rep['best_epoch']}, {rep['best']})")
    stopped_by = "early_stop" if (stopped_early or
                                  rep["patience_cnt"] >= PATIENCE) \
        else "max_epochs"
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
               if device != "cpu" else 0.0)
    info = {"run_id": rid, "status": "success", "method": "TN0",
            "domain": domain, "lr": lr, "seed": seed,
            "wall_s": time.time() - t0, "peak_vram_mb": peak_mb,
            "updates_total": rep["updates_total"],
            "presented_total": rep["presented_total"],
            "U_d": U_d, "S_d": S_d, "epochs_run": last_epoch,
            "stopped_by": stopped_by,
            "still_improving_at_max": stopped_by == "max_epochs"
            and rep["patience_cnt"] < PATIENCE,
            "local_best_epoch": sel["epoch"],
            "local_best_val_mse": sel["value"],
            "val_mse_by_epoch": rep["val_by_epoch"],
            "finish_attempt": attempt}
    atomic_write_json({**info, "frozen_best_epoch": sel["epoch"],
                       "frozen_best_val": sel["value"]}, success_path)
    return info
