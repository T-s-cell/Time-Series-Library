#!/usr/bin/env python3
"""Shared foundation for the TimeMixer TimesX T1 experiment (spec v1).

- All paths are built from THIS file's location; never from the cwd.
- configs/v1.yaml is JSON (a YAML subset) so no yaml dependency is needed.
- torch is imported lazily so prepare.py / freeze_selection.py / audit.py run
  without it. PYTHONDONTWRITEBYTECODE=1 and CUBLAS_WORKSPACE_CONFIG are set
  before any torch import can happen.
"""
import contextlib
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)                       # Time-Series-Library/

CONFIG_PATH = os.path.join(HERE, "configs", "v1.yaml")
with open(CONFIG_PATH) as _f:
    CFG = json.load(_f)

SNAPSHOT = os.path.join(HERE, "snapshot", "manifests")
SNAPSHOT_DIR = os.path.join(HERE, "snapshot")
CHECKPOINTS = os.path.join(HERE, "checkpoints")
LOGS = os.path.join(HERE, "logs")
PREDICTIONS = os.path.join(HERE, "predictions")
RESULTS = os.path.join(HERE, "results")
AUDIT = os.path.join(HERE, "audit")
for _d in (SNAPSHOT, CHECKPOINTS, LOGS, PREDICTIONS, RESULTS, AUDIT):
    os.makedirs(_d, exist_ok=True)

DATA_CACHE = os.path.join(SNAPSHOT, "data_cache.npz")
SPLIT_MANIFEST = os.path.join(SNAPSHOT, "split_manifest.json")
PROTOCOL_LN = os.path.join(SNAPSHOT, "protocol.json")
SPLIT_COUNTS = os.path.join(SNAPSHOT, "split_counts.csv")
SNAPSHOT_PROTOCOL = os.path.join(SNAPSHOT_DIR, "protocol.json")
BASELINE_JSON = os.path.join(AUDIT, "baseline.json")
PREFLIGHT_JSON = os.path.join(AUDIT, "preflight_report.json")
SELECTION_JSON = os.path.join(RESULTS, "selection.json")
RUN_MANIFEST = os.path.join(RESULTS, "run_manifest.json")
PAUSE_SENTINEL = os.path.join(HERE, "PAUSE")

DOMAINS = sorted(CFG["budget"]["expected_domain_table"],
                 key=lambda r: r["domain"])
DOMAIN_NAMES = [r["domain"] for r in DOMAINS]
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAIN_NAMES)}
CTX = CFG["data"]["ctx"]
PRED = CFG["data"]["pred"]
STD_EPS = CFG["data"]["std_eps"]
SEEDS = list(CFG["budget"]["seeds"])
LRS = list(CFG["optim"]["lrs"])
EPOCHS = CFG["budget"]["epochs"]
PATIENCE = CFG["budget"]["patience"]
BATCH = CFG["optim"]["batch"]
EVAL_BATCH = CFG["test"]["eval_batch"]
CLIP = CFG["optim"]["clip_grad"]
TIE_TOL = CFG["selection"]["tie_tol"]


class DivergenceError(RuntimeError):
    """Numerical divergence ONLY (NaN/Inf in loss/grad/params/val outputs).

    Strictly separated from OOM / data / code errors: a DivergenceError marks
    the candidate DIVERGED (terminal) and the queue continues; any other
    error stops the queue with the scene preserved.
    """

    def __init__(self, message, epoch=None, update=None):
        super().__init__(message)
        self.epoch = epoch
        self.update = update


def sha256_file(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path, buf_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json_sha256(obj):
    """Content-level hash; identical recipe to the LN audit evidence
    (json.dumps with sort_keys=True, default separators)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


_CODE_EXCLUDE_DIRS = {"snapshot", "checkpoints", "logs", "predictions",
                      "results", "audit", "__pycache__"}


def code_fingerprint():
    """md5 of every .py/.sh/.yaml in this experiment dir (isolation+resume)."""
    out = {}
    for root, _dirs, files in os.walk(HERE):
        rel_root = os.path.relpath(root, HERE)
        if rel_root.split(os.sep)[0] in _CODE_EXCLUDE_DIRS:
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".sh", ".yaml")):
                rel = os.path.normpath(os.path.join(rel_root, fn))
                out[rel] = md5_file(os.path.join(root, fn))
    return out


def tslib_state():
    """Live TSLib repo state. violations lists every porcelain entry that is
    NOT an untracked path inside this experiment dir (tracked files must be
    untouched; new content must stay inside repro_timemixer_timesx_v1/)."""
    def _git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout
    commit = _git("rev-parse", "HEAD").strip()
    lines = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    violations = []
    for ln in lines:
        path = ln[3:].strip().strip('"')
        if ln.startswith("??") and (path == "repro_timemixer_timesx_v1"
                                    or path.startswith(
                                        "repro_timemixer_timesx_v1" + os.sep)
                                    or path.startswith(
                                        "repro_timemixer_timesx_v1/")):
            continue
        violations.append(ln)
    core_md5 = {rel: md5_file(os.path.join(REPO_ROOT, rel))
                for rel in sorted(CFG["model"]["core_md5"])}
    return {"commit": commit, "violations": violations, "core_md5": core_md5}


def protocol_self_hash():
    """sha256 of snapshot/protocol.json with bindings.protocol_self set to
    null (the phase-1 file written by prepare). Deterministic re-serialization
    (json indent=1 sort_keys=True) makes this reproducible from the final
    file without circularity."""
    if not os.path.isfile(SNAPSHOT_PROTOCOL):
        return None
    with open(SNAPSHOT_PROTOCOL) as f:
        proto = json.load(f)
    proto = json.loads(json.dumps(proto))
    proto["bindings"]["protocol_self"] = None
    return hashlib.sha256(
        json.dumps(proto, indent=1, sort_keys=True).encode("utf-8")).hexdigest()


def snapshot_hashes():
    """Container sha256 of the exported read-only LN materials + the
    self-excluded hash of this experiment's frozen protocol (None when not
    yet exported)."""
    def _h(p):
        return sha256_file(p) if os.path.isfile(p) else None
    return {"data_cache": _h(DATA_CACHE),
            "split_manifest": _h(SPLIT_MANIFEST),
            "protocol_ln": _h(PROTOCOL_LN),
            "protocol_self": protocol_self_hash()}


def actual_fingerprints():
    """RECOMPUTED hashes of everything the frozen protocol pins (never trust
    declared values alone)."""
    return {"snapshot": snapshot_hashes(),
            "tslib": tslib_state(),
            "code": code_fingerprint()}


def binding_mismatches(want, act):
    """Compare a want-dict (from snapshot/protocol.json or resume.pt) against
    freshly recomputed fingerprints. Returns list of mismatched keys."""
    bad = []
    for k in ("data_cache", "split_manifest", "protocol_ln", "protocol_self"):
        if want.get(k) != act["snapshot"].get(k):
            bad.append(k)
    if want.get("tslib_commit") != act["tslib"]["commit"]:
        bad.append("tslib_commit")
    if list(want.get("tslib_violations") or []) != act["tslib"]["violations"]:
        bad.append("tslib_violations")
    if want.get("tslib_core_md5") != act["tslib"]["core_md5"]:
        bad.append("tslib_core_md5")
    if want.get("code") != act["code"]:
        bad.append("code")
    return bad


def atomic_torch_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def environment_snapshot():
    import numpy as np
    import torch
    import sys as _s
    return {"python": _s.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def cuda_state():
    """CUDA identity snapshot; part of the launch bindings so a stage never
    silently runs somewhere its preflight evidence does not describe."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False, "device_name": None, "device_count": 0}
    return {"available": True, "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count()}


def launch_bindings():
    """Everything a stage gate must re-verify: frozen materials + TSLib state
    + experiment code + env + CUDA. Stored in the preflight report; compared
    live by train/freeze/test via launch_gate."""
    act = actual_fingerprints()
    return {"snapshot": act["snapshot"],
            "tslib": {"commit": act["tslib"]["commit"],
                      "violations": act["tslib"]["violations"],
                      "core_md5": act["tslib"]["core_md5"]},
            "code": act["code"],
            "env": environment_snapshot(),
            "cuda": cuda_state()}


def launch_gate_drift(want):
    """Keys where the live tree/env drifted from a preflight report's
    launch bindings."""
    cur = launch_bindings()
    bad = []
    for k in ("snapshot", "code", "env", "cuda"):
        if want.get(k) != cur[k]:
            bad.append(k)
    wt = want.get("tslib") or {}
    for k in ("commit", "violations", "core_md5"):
        if wt.get(k) != cur["tslib"][k]:
            bad.append(f"tslib.{k}")
    return bad


def launch_gate(stage):
    """Hard gate for train/freeze/test: the preflight report must exist, be
    PASS, and its bindings must match the live tree/env. SystemExit(2) on any
    refusal (stale/drifted evidence is never silently accepted)."""
    tag = stage.upper()
    if not os.path.isfile(PREFLIGHT_JSON):
        print(f"{tag} ABORT: audit/preflight_report.json missing - run "
              f"preflight first", file=sys.stderr)
        raise SystemExit(2)
    with open(PREFLIGHT_JSON) as f:
        rep = json.load(f)
    if rep.get("status") != "PASS":
        print(f"{tag} ABORT: preflight status {rep.get('status')} != PASS",
              file=sys.stderr)
        raise SystemExit(2)
    if "bindings" not in rep:
        print(f"{tag} ABORT: preflight report has no bindings block (stale "
              f"report) - re-run preflight", file=sys.stderr)
        raise SystemExit(2)
    drift = launch_gate_drift(rep["bindings"])
    if drift:
        print(f"{tag} ABORT: live tree/env drifted from the preflight report "
              f"in {drift} - re-run prepare+preflight", file=sys.stderr)
        raise SystemExit(2)
    return rep


def run_id(domain, lr, seed):
    return f"{domain}__lr{lr:g}__s{seed}"


def u_d_s_d(dense_count):
    """U_d = ceil(D_d/32) updates per epoch; S_d = 32*U_d presentations."""
    u = -(-dense_count // BATCH)
    return u, BATCH * u


@contextlib.contextmanager
def train_mode(model, training):
    was = model.training
    model.train(training)
    yield
    model.train(was)
