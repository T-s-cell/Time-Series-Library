#!/usr/bin/env python3
"""preflight: A-M evidence gates. ALL PASS is required before train.

A  environment + snapshot bindings + disk + TSLib isolation
B  model/grad structure: 63531/60 pins, 5 None-grad tensors (None after
   backward AND unchanged after Adam step), dropout train/eval difference,
   eval determinism
C  init determinism: same (domain, seed) -> bitwise-equal initial weights;
   different seed -> different weights
D  slot probes: OUR build_slots vs the LN code's real build_slots (ast
   extraction, cross-machine), 19-domain budget table, sum(U_d)=1157
E  loss semantics: closed-form batch loss, constant-history fallback_std,
   evaluate_split closed form incl. tail batch (drop_last=False)
F  numerics: TF32 off / deterministic flags / bitwise repeat of 3 updates
   (on CUDA when available - catches TF32 leaks on the real path)
G  test-target isolation: NaN-poisoned test rows leave epoch-1 training+val
   bitwise unchanged (train/val never read test)
H  resume ON CUDA WHEN AVAILABLE: first finite val initializes best; bitwise
   epoch-2 continuation after resume (4-RNG replay incl. the CUDA dropout
   stream); hash-mismatch refusal; DIVERGED terminal skip
I  throughput + peak VRAM on the co-tenanted GPU (gate basis for the queue)
K  selection/aggregation closed forms incl. diverged-seed +inf rule, tie ->
   smaller lr, all-lr-invalid park, earliest-argmin epoch rule
M  fingerprint binding: live recompute + tamper detection + resume roundtrip

Scratch output stays under audit/pf_tmp/ and is wiped at the end. No test
metrics are produced. Set TM_PREFLIGHT_ALLOW_CPU=1 to develop on a CPU box
(section I is skipped, report marked DEV); eta runs never set it.
"""
import datetime
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

from common import (AUDIT, CFG, DOMAIN_IDX, SNAPSHOT_PROTOCOL,
                    actual_fingerprints, atomic_torch_save, atomic_write_json,
                    binding_mismatches, environment_snapshot, launch_bindings,
                    tslib_state, u_d_s_d)
import trainer
from data import DenseData, window_d
from freeze_selection import lr_score, pick_lr, require_available
from ln_probe import compute_probe_hashes
from model_adapter import forward_pred, load_model, setup_numerics
from trainer import (build_slots, evaluate_split, run_training,
                     select_best_epoch, seed_everything, slot_hash)

PF_TMP = os.path.join(AUDIT, "pf_tmp")
ALLOW_CPU = os.environ.get("TM_PREFLIGHT_ALLOW_CPU") == "1"


def fail(msg):
    print(f"PREFLIGHT ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


class Section:
    def __init__(self, name):
        self.name = name
        self.checks = {}
        self.extras = {}
        self.ok = True
        self.t0 = time.time()

    def check(self, name, cond, detail=""):
        self.checks[name] = {"ok": bool(cond), "detail": str(detail)[:400]}
        if not cond:
            self.ok = False
            print(f"  [{self.name}] FAIL {name}: {str(detail)[:200]}",
                  flush=True)
        return bool(cond)

    def done(self):
        out = {"status": "PASS" if self.ok else "FAIL",
               "checks": self.checks,
               "seconds": round(time.time() - self.t0, 2)}
        out.update(self.extras)
        return out


class Sandbox:
    """Redirect trainer.CHECKPOINTS/LOGS into audit/pf_tmp/<name>/."""

    def __init__(self, name):
        self.root = os.path.join(PF_TMP, name)
        self._saved = {}

    def __enter__(self):
        os.makedirs(os.path.join(self.root, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "logs"), exist_ok=True)
        self._saved = {"CHECKPOINTS": trainer.CHECKPOINTS, "LOGS": trainer.LOGS}
        trainer.CHECKPOINTS = os.path.join(self.root, "checkpoints")
        trainer.LOGS = os.path.join(self.root, "logs")
        return self

    def __exit__(self, *exc):
        trainer.CHECKPOINTS = self._saved["CHECKPOINTS"]
        trainer.LOGS = self._saved["LOGS"]
        return False


def load_json(path):
    with open(path) as f:
        return json.load(f)


def real_batch(data, dom, seed=2021, epoch=1, k=32):
    import torch
    slots, _, _ = build_slots(data, dom, seed, epoch)
    chunk = slots[:k]
    xs, ys, ds = [], [], []
    for vk, s0 in chunk:
        x, y = data.window(vk, s0)
        xs.append(x)
        ys.append(y)
        ds.append(window_d(x, data.fb_std[vk]))
    xt = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(np.stack(ys), dtype=torch.float32)
    dt = torch.tensor(np.stack(ds), dtype=torch.float32).unsqueeze(1)
    return xt, yt, dt, slots


def section_a():
    s = Section("A_env")
    env = environment_snapshot()
    s.check("env_recorded", all(env.values()), env)
    proto = load_json(SNAPSHOT_PROTOCOL)
    s.check("bindings_match", not binding_mismatches(proto["bindings"],
                                                     actual_fingerprints()),
            "snapshot/protocol.json bindings vs live recomputation")
    s.check("protocol_self_present",
            isinstance(proto["bindings"].get("protocol_self"), str),
            "two-phase projection intact")
    man = os.path.join(os.path.dirname(SNAPSHOT_PROTOCOL), "manifests")
    for f in ("data_cache.npz", "split_manifest.json", "protocol.json"):
        s.check(f"snapshot_{f}", os.path.isfile(os.path.join(man, f)))
    free_gb = shutil.disk_usage(SNAPSHOT_PROTOCOL).free / 1024 ** 3
    s.check("disk_free", free_gb >= CFG["resources"]["min_disk_free_gb"],
            f"{free_gb:.1f} GiB")
    ts = tslib_state()
    s.check("tslib_isolation", not ts["violations"], ts["violations"])
    s.check("config_budget",
            (CFG["budget"]["epochs"], CFG["budget"]["patience"],
             CFG["budget"]["seeds"], CFG["optim"]["lrs"])
            == (100, 15, [2021, 2022, 2023], [0.0001, 0.001, 0.01]),
            "epochs/patience/seeds/lrs")
    return s.done(), env


def section_b(data):
    import torch
    s = Section("B_model")
    m = load_model("cpu")
    s.check("param_count", sum(p.numel() for p in m.parameters())
            == CFG["model"]["params_expected"],
            sum(p.numel() for p in m.parameters()))
    s.check("tensor_count", len(list(m.parameters()))
            == CFG["model"]["tensors_expected"], len(list(m.parameters())))
    seed_everything(2021, DOMAIN_IDX["arts"])
    m = load_model("cpu")
    xt, yt, dt, _ = real_batch(data, "arts")
    m.train()
    pred = forward_pred(m, xt)
    loss = torch.mean(((pred - yt) / dt) ** 2)
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    none_names = [n for n, p in m.named_parameters() if p.grad is None]
    s.check("grad_bearing", n_grad == CFG["model"]["grad_bearing_expected"],
            n_grad)
    s.check("grad_none_names", none_names == CFG["model"]["grad_none_expected"],
            none_names)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    snap_none = {n: p.detach().clone()
                 for n, p in m.named_parameters() if p.grad is None}
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    s.check("none_grad_untouched_by_step",
            all(torch.equal(snap_none[n], p)
                for n, p in m.named_parameters() if p.grad is None),
            "Adam must skip None-grad tensors")
    m.train()
    s.check("dropout_active_in_train",
            not torch.equal(forward_pred(m, xt), forward_pred(m, xt)),
            "two train forwards must differ (dropout 0.1)")
    m.eval()
    with torch.no_grad():
        e1 = forward_pred(m, xt)
        e2 = forward_pred(m, xt)
    s.check("eval_deterministic", torch.equal(e1, e2),
            "two eval forwards must be bitwise equal")
    return s.done()


def section_c():
    import torch
    s = Section("C_init")
    di = DOMAIN_IDX["arts"]

    def init_sd(seed):
        seed_everything(seed, di)
        return {k: v.clone() for k, v in load_model("cpu").state_dict().items()}

    a1, a2 = init_sd(2021), init_sd(2021)
    s.check("same_seed_bitwise_identical",
            all(torch.equal(a1[k], a2[k]) for k in a1),
            "init must not depend on invocation order")
    a3 = init_sd(2022)
    s.check("different_seed_differs",
            any(not torch.equal(a1[k], a3[k]) for k in a1),
            "seed must change init")
    return s.done()


def section_d(data):
    s = Section("D_slots")
    pins = CFG.get("slot_probe_hashes") or {}
    if not pins:
        s.check("pins_present", False,
                "configs/v1.yaml slot_probe_hashes empty - generate with "
                "ln_probe.py before preflight")
        return s.done()
    src = CFG["source_snapshot"]
    ln_dir = src["ln_dir_default"] if os.path.isfile(
        os.path.join(src["ln_dir_default"], "trainer.py")) \
        else src["ln_dir_local_fallback"]
    got = compute_probe_hashes(ln_dir, data)
    for key, rec in sorted(pins.items()):
        g = got.get(key)
        s.check(f"probe_{key}", g is not None
                and g["slot_hash"] == rec["slot_hash"]
                and g["n_slots"] == rec["n_slots"],
                f"want {rec} got {g}")
    sum_u = 0
    for row in CFG["budget"]["expected_domain_table"]:
        tot = data.domain_dense_total(row["domain"])
        u, sd_ = u_d_s_d(tot)
        sum_u += u
        s.check(f"budget_{row['domain']}", tot == row["dense"]
                and sd_ == 32 * u, f"D={tot} U={u} S={sd_}")
    s.check("sum_U_d", sum_u == CFG["budget"]["sum_U_d_expected"], sum_u)
    s.check("build_slots_deterministic",
            slot_hash(build_slots(data, "arts", 2021, 1)[0])
            == slot_hash(build_slots(data, "arts", 2021, 1)[0]),
            "same inputs -> same slots")
    return s.done()


def section_e(data):
    import torch
    s = Section("E_loss")
    xt, yt, dt, _ = real_batch(data, "arts")
    seed_everything(2021, DOMAIN_IDX["arts"])
    m = load_model("cpu")
    m.train()
    pred = forward_pred(m, xt)
    loss_t = float(torch.mean(((pred - yt) / dt) ** 2).detach())
    pf = pred.detach().numpy().astype(np.float64)
    yf = yt.numpy().astype(np.float64)
    df = dt.numpy().astype(np.float64)
    loss_np = float(np.mean(np.mean(((pf - yf) / df) ** 2, axis=1)))
    rel = abs(loss_t - loss_np) / max(1e-12, loss_np)
    s.check("loss_closed_form", rel < 1e-5,
            f"torch={loss_t} numpy={loss_np} rel={rel:.2e}")
    vk0 = data.domain_vars["arts"][0]
    s.check("fallback_constant_history",
            window_d(np.full(96, 3.5), data.fb_std[vk0]) == data.fb_std[vk0],
            "pstdev(const)=0 < std_eps -> fallback")
    tiny = np.full(96, 1.0) + np.linspace(0, 1e-10, 96)
    s.check("fallback_below_eps",
            window_d(tiny, data.fb_std[vk0]) == data.fb_std[vk0],
            f"std={float(np.std(tiny)):.2e} < 1e-8")
    val = evaluate_split(m, data, "arts", "val", "cpu")
    rows = data.split_rows("arts", "val")
    m.eval()
    per_var, preds_all = {}, []
    with torch.no_grad():
        for i0 in range(0, len(rows), CFG["test"]["eval_batch"]):
            chunk = rows[i0:i0 + CFG["test"]["eval_batch"]]
            xb = torch.tensor(np.stack([r[2] for r in chunk]),
                              dtype=torch.float32).unsqueeze(-1)
            preds_all.append(forward_pred(m, xb).numpy().astype(np.float64))
    preds = np.concatenate(preds_all)
    for j, (vk, sid, x, y) in enumerate(rows):
        d = window_d(x, data.fb_std[vk])
        e = (preds[j] - y.astype(np.float64)) / d
        pv = per_var.setdefault(vk, {"mse": 0.0, "mae": 0.0, "n": 0})
        pv["mse"] += float(np.mean(e ** 2))
        pv["mae"] += float(np.mean(np.abs(e)))
        pv["n"] += 1
    mse = float(np.mean([pv["mse"] / pv["n"] for pv in per_var.values()]))
    s.check("evaluate_split_closed_form", mse == val["val_mse"],
            f"manual={mse!r} evaluate_split={val['val_mse']!r} (bitwise, "
            f"n={len(rows)} windows incl. tail batch)")
    s.check("var_equal_weighting", len(per_var)
            == len(data.domain_vars["arts"]),
            f"{len(per_var)} vars aggregated equally")
    return s.done()


def section_f(data):
    import torch
    s = Section("F_numerics")
    setup_numerics()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s.extras["device"] = dev
    s.check("tf32_matmul_off", not torch.backends.cuda.matmul.allow_tf32, "")
    s.check("tf32_cudnn_off", not torch.backends.cudnn.allow_tf32, "")
    s.check("cudnn_deterministic", torch.backends.cudnn.deterministic, "")
    s.check("deterministic_algorithms",
            torch.are_deterministic_algorithms_enabled(), "")

    def three_losses():
        seed_everything(2021, DOMAIN_IDX["arts"])
        m = load_model(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        m.train()
        slots, _, _ = build_slots(data, "arts", 2021, 1)
        out = []
        for u in range(3):
            chunk = slots[u * 32:(u + 1) * 32]
            xs, ys, ds = [], [], []
            for vk, s0 in chunk:
                x, y = data.window(vk, s0)
                xs.append(x)
                ys.append(y)
                ds.append(window_d(x, data.fb_std[vk]))
            xb = torch.tensor(np.stack(xs), dtype=torch.float32,
                              device=dev).unsqueeze(-1)
            yb = torch.tensor(np.stack(ys), dtype=torch.float32,
                              device=dev)
            db = torch.tensor(np.stack(ds), dtype=torch.float32,
                              device=dev).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean(((forward_pred(m, xb) - yb) / db) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            out.append(float(loss.detach()))
        return out

    l1, l2 = three_losses(), three_losses()
    s.check("repeat_bitwise", l1 == l2, f"{l1} vs {l2} on {dev}")
    return s.done()


class Poisoned(DenseData):
    def split_rows(self, domain, split):
        rows = super().split_rows(domain, split)
        if split == "test":
            rows = [(vk, sid, x, np.full(y.shape, np.nan))
                    for vk, sid, x, y in rows]
        return rows


def section_g(clean_val_mse):
    s = Section("G_isolation")
    data = Poisoned()
    trainer.EPOCHS = 1
    try:
        with Sandbox("g"):
            info = run_training("arts", 1e-4, 2021, data, "cpu")
        s.check("poisoned_run_completes", info.get("status") == "success",
                info.get("status"))
        s.check("val_bitwise_unchanged",
                info["val_mse_by_epoch"][1] == clean_val_mse,
                f"poisoned={info['val_mse_by_epoch'][1]!r} "
                f"clean={clean_val_mse!r} - test rows never read")
    except Exception as exc:  # noqa: BLE001
        s.check("poisoned_run_completes", False, repr(exc))
        return s.done()
    try:
        evaluate_split(load_model("cpu"), data, "arts", "test", "cpu")
        s.check("poison_propagates", False,
                "poisoned test eval should raise DivergenceError")
    except trainer.DivergenceError:
        s.check("poison_propagates", True, "NaN test targets detected")
    return s.done()


def section_h(data):
    import torch
    s = Section("H_resume")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s.extras["device"] = dev
    rid = "arts__lr0.0001__s2021"

    def ckpt(p):
        return os.path.join(PF_TMP, p, "checkpoints", rid)

    with Sandbox("h_a"):
        trainer.EPOCHS = 1
        info1 = run_training("arts", 1e-4, 2021, data, dev)
        s.check("epoch1_completes", info1.get("status") == "success",
                info1.get("status"))
        vj = load_json(os.path.join(ckpt("h_a"), "ep001.val.json"))
        s.check("first_finite_val_initializes_best",
                vj["best_epoch"] == 1 and vj["best_val"] == vj["val_mse"]
                and vj["patience_cnt"] == 0,
                f"best={vj['best_val']!r}@{vj['best_epoch']} "
                f"patience={vj['patience_cnt']}")
        os.remove(os.path.join(ckpt("h_a"), "SUCCESS.json"))
    random = __import__("random")
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    with Sandbox("h_a"):
        trainer.EPOCHS = 2
        info2 = run_training("arts", 1e-4, 2021, data, dev)
        s.check("resume_completes", info2.get("status") == "success",
                info2.get("status"))
        s.check("resume_val_history",
                sorted(map(int, info2["val_mse_by_epoch"])) == [1, 2],
                info2["val_mse_by_epoch"])
    with Sandbox("h_b"):
        trainer.EPOCHS = 2
        ref = run_training("arts", 1e-4, 2021, data, dev)
    s.check("resume_bitwise_continuation",
            info2["val_mse_by_epoch"] == ref["val_mse_by_epoch"],
            f"resumed={info2['val_mse_by_epoch']} "
            f"uninterrupted={ref['val_mse_by_epoch']} (4-RNG replay incl. "
            f"dropout, device={dev})")
    rp = os.path.join(ckpt("h_b"), "resume.pt")
    if os.path.isfile(rp):
        rs = torch.load(rp, map_location="cpu", weights_only=False)
        rs["hashes"]["lr"] = 42.0
        atomic_torch_save(rs, rp)
        with Sandbox("h_b"):
            os.remove(os.path.join(ckpt("h_b"), "SUCCESS.json"))
            trainer.EPOCHS = 3
            try:
                run_training("arts", 1e-4, 2021, data, dev)
                s.check("hash_mismatch_refused", False,
                        "resume accepted drift!")
            except RuntimeError as exc:
                s.check("hash_mismatch_refused", "resume refused" in str(exc),
                        str(exc)[:200])
    else:
        s.check("hash_mismatch_refused", False, "reference resume.pt missing")
    rid2 = "arts__lr0.001__s2022"
    with Sandbox("h_c"):
        os.makedirs(os.path.join(PF_TMP, "h_c", "checkpoints", rid2),
                    exist_ok=True)
        atomic_write_json({"status": "diverged", "reason": "preflight probe",
                           "epoch": 0, "update": 3},
                          os.path.join(PF_TMP, "h_c", "checkpoints", rid2,
                                       "DIVERGED.json"))
        info = run_training("arts", 1e-3, 2022, data, dev)
        s.check("diverged_terminal_skip", info.get("skipped") is True
                and info.get("terminal") == "diverged", info)
    return s.done()


def section_i(data):
    import torch
    s = Section("I_throughput")
    if not torch.cuda.is_available():
        if ALLOW_CPU:
            s.check("gpu_skipped_dev", True, "TM_PREFLIGHT_ALLOW_CPU=1")
            return s.done()
        s.check("gpu_required", False,
                "CUDA unavailable - preflight must run on the eta L20")
        return s.done()
    device = "cuda"
    dom = "Currency"
    u_d = u_d_s_d(data.domain_dense_total(dom))[0]
    seed_everything(2021, DOMAIN_IDX[dom])
    m = load_model(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    m.train()
    slots, _, _ = build_slots(data, dom, 2021, 1)
    torch.cuda.reset_peak_memory_stats(device)
    n_meas = min(20, u_d)
    t0 = time.time()
    for uu in range(n_meas):
        chunk = slots[uu * 32:(uu + 1) * 32]
        xs, ys, ds = [], [], []
        for vk, s0 in chunk:
            x, y = data.window(vk, s0)
            xs.append(x)
            ys.append(y)
            ds.append(window_d(x, data.fb_std[vk]))
        xb = torch.tensor(np.stack(xs), dtype=torch.float32,
                          device=device).unsqueeze(-1)
        yb = torch.tensor(np.stack(ys), dtype=torch.float32, device=device)
        db = torch.tensor(np.stack(ds), dtype=torch.float32,
                          device=device).unsqueeze(1)
        opt.zero_grad(set_to_none=True)
        loss = torch.mean(((forward_pred(m, xb) - yb) / db) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize(device)
    s_per_update = (time.time() - t0) / n_meas
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    t1 = time.time()
    evaluate_split(m, data, dom, "val", device)
    dt_val = time.time() - t1
    n_val = data.domain_native_total(dom, "val")
    s_per_val_window = dt_val / n_val
    total_upd = 9 * CFG["budget"]["epochs"] * CFG["budget"]["sum_U_d_expected"]
    total_val = 9 * CFG["budget"]["epochs"] * 895
    est_h = (total_upd * s_per_update + total_val * s_per_val_window) / 3600
    s.extras["peak_vram_mb"] = round(peak, 1)
    s.extras["s_per_update"] = round(s_per_update, 4)
    s.extras["s_per_val_window"] = round(s_per_val_window, 5)
    s.extras["est_upper_bound_hours"] = round(est_h, 2)
    s.check("measured", True,
            f"s/update={s_per_update:.4f} peak_vram={peak:.0f}MiB "
            f"val({n_val}w)={dt_val:.1f}s "
            f"({s_per_val_window * 1000:.1f} ms/window)")
    s.check("est_upper_bound_hours", est_h < 72,
            f"{est_h:.1f} h without early stopping ({total_upd} updates + "
            f"{total_val} val windows); early stopping will cut this")
    margin = CFG["resources"]["vram_margin_mb"]
    free = float(os.popen("nvidia-smi --query-gpu=memory.total "
                          "--format=csv,noheader,nounits").read().strip()
                 .splitlines()[0])
    s.check("vram_gate_feasible", peak + margin < free,
            f"peak {peak:.0f} + margin {margin} MiB < total {free:.0f} MiB")
    return s.done()


def section_k():
    s = Section("K_selection")
    import aggregate as agg

    t1 = {"arts__a": {"mse": 1.0, "mae": 2.0, "raw_mse": 3.0, "raw_mae": 4.0,
                      "n": 2},
          "arts__b": {"mse": 5.0, "mae": 6.0, "raw_mse": 7.0, "raw_mae": 8.0,
                      "n": 3},
          "traffic__c": {"mse": 9.0, "mae": 10.0, "raw_mse": 11.0,
                         "raw_mae": 12.0, "n": 4}}
    dom_t, overall = agg.dom_overall(t1, strict=False)
    s.check("dom_overall_closed_form",
            dom_t["arts"]["mse"] == 3.0 and dom_t["traffic"]["mse"] == 9.0
            and overall["mse"] == 6.0 and overall["mae"] == 7.0,
            f"{dom_t} {overall}")
    s.check("dom_overall_strict_unknown_domain", _raises(
        lambda: agg.dom_overall(t1, strict=True)),
        "unknown domain prefix must raise in strict mode")
    s.check("merge_duplicate_raises", _raises(
        lambda: agg.merge_var_tables([t1, {"d1__a": t1["d1__a"]}],
                                     strict=False)), "duplicate var")
    s.check("merge_coverage_raises", _raises(
        lambda: agg.merge_var_tables([{"d1__a": t1["d1__a"]}], strict=True)),
        "unknown/missing domains")
    s.check("seed_stats", agg.seed_stats([1.0, 2.0, 3.0]) == (2.0, 1.0)
            and agg.seed_stats([5.0]) == (5.0, None), "ddof=1")
    s.check("lr_score_diverged_inf", lr_score([0.0001], ["d"]) == float("inf"),
            "a diverged seed must void the mean, never average remaining")
    s.check("lr_score_mean", lr_score([1.0, 2.0, 3.0], []) == 2.0, "")
    s.check("pick_lr_lowest", pick_lr({1e-4: 3.0, 1e-3: 2.0, 1e-2: 2.5})
            == (1e-3, 2.0), "")
    s.check("pick_lr_tie_smaller",
            pick_lr({1e-4: 2.0, 1e-3: 2.0 + 5e-9, 1e-2: 9.9}) == (1e-4, 2.0),
            "5e-9 gap <= 1e-8 tie tol -> smaller lr kept")
    s.check("pick_lr_late_better_wins",
            pick_lr({1e-4: 2.0, 1e-3: 2.0, 1e-2: 1.99999})
            == (1e-2, 1.99999), "strictly better late lr must win")
    s.check("pick_lr_inf_skip",
            pick_lr({1e-4: float("inf"), 1e-3: 3.0, 1e-2: 2.0}) == (1e-2, 2.0),
            "+inf lr must be skipped")
    s.check("all_inf_parks", _raises(lambda: require_available(
        {1e-4: float("inf"), 1e-3: float("inf"), 1e-2: float("inf")}, "x")),
        "must-park condition")
    s.check("earliest_argmin",
            select_best_epoch({1: 1.0, 2: 0.9, 3: 0.9, 4: 0.5})["epoch"] == 4
            and select_best_epoch({1: 1.0, 2: 1.0 + 5e-9})["epoch"] == 1,
            "argmin with tie tolerance keeps earlier epoch")
    return s.done()


def _raises(fn):
    try:
        fn()
        return False
    except BaseException:  # noqa: BLE001 - SystemExit counts as "raised"
        return True


def section_m():
    s = Section("M_fingerprints")
    proto = load_json(SNAPSHOT_PROTOCOL)
    act = actual_fingerprints()
    s.check("binding_clean", not binding_mismatches(proto["bindings"], act),
            "live recompute vs frozen bindings")
    want = {**proto["bindings"], "data_cache": "0" * 64}
    s.check("tamper_detected", "data_cache"
            in binding_mismatches(want, act), "corrupted pin must be caught")
    s.check("resume_roundtrip", not binding_mismatches(
        {**proto["bindings"], "lr": 0.001}, act),
        "resume.pt want-dict incl. lr must verify clean")
    ts = tslib_state()
    s.check("tslib_commit", ts["commit"] == proto["bindings"]["tslib_commit"],
            ts["commit"])
    return s.done()


def main():
    t0 = time.time()
    os.makedirs(PF_TMP, exist_ok=True)
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "protocol_version": CFG["protocol_version"],
              "dev_mode_cpu": ALLOW_CPU, "sections": {}}
    # launch bindings: every later stage (train/freeze/test) re-verifies these
    # against the live tree before acting
    report["bindings"] = launch_bindings()
    try:
        if not os.path.isfile(SNAPSHOT_PROTOCOL):
            fail("snapshot/protocol.json missing - run prepare first")
        data = DenseData()
        report["sections"]["A_env"], report["environment"] = section_a()
        report["sections"]["B_model"] = section_b(data)
        report["sections"]["C_init"] = section_c()
        report["sections"]["D_slots"] = section_d(data)
        report["sections"]["E_loss"] = section_e(data)
        report["sections"]["F_numerics"] = section_f(data)
        with Sandbox("g_clean"):
            trainer.EPOCHS = 1
            info = run_training("arts", 1e-4, 2021, data, "cpu")
            clean_val = info["val_mse_by_epoch"][1]
        report["sections"]["G_isolation"] = section_g(clean_val)
        report["sections"]["H_resume"] = section_h(data)
        report["sections"]["I_throughput"] = section_i(data)
        report["sections"]["K_selection"] = section_k()
        report["sections"]["M_fingerprints"] = section_m()
    except Exception as exc:  # noqa: BLE001
        report["uncaught"] = {"error": repr(exc),
                              "traceback": traceback.format_exc()}
        print(report["uncaught"]["traceback"], file=sys.stderr)
    finally:
        report["wall_s"] = round(time.time() - t0, 1)
        hard = [k for k, v in report["sections"].items()
                if v["status"] not in ("PASS", "SKIP(dev)")]
        report["status"] = "FAIL" if hard or report.get("uncaught") else "PASS"
        atomic_write_json(report, os.path.join(AUDIT, "preflight_report.json"))
        shutil.rmtree(PF_TMP, ignore_errors=True)
    print(f"[preflight] status={report['status']} ({report['wall_s']}s) -> "
          f"audit/preflight_report.json", flush=True)
    for k, v in report["sections"].items():
        print(f"  {k}: {v['status']} ({v['seconds']}s)", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
