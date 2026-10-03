#!/usr/bin/env python3
"""model_check: prove the adapted vendored TimesNet is the upstream model.

The adapted/ tree differs from upstream/ ONLY by import-line package
isolation (tn_models/tn_layers vs models/layers, exactly 2 changed import
lines - see vendor/TimesNet/import_diff.patch); state_dict keys contain no
package names, so weights transfer 1:1. Because BOTH variants occupy the
`models`/`layers` namespace, they can never be imported into one process -
the upstream reference therefore runs in an isolated subprocess with
cwd/sys.path rooted at vendor/TimesNet/upstream.

Protocol (CPU, eval, no_grad, deterministic, single-threaded; MUST run in
the pinned CUDA_VISIBLE_DEVICES env so the recorded launch bindings agree
with the other evidence reports - a device_count mismatch would refuse the
train launch gate):
1. parent seeds torch, builds the adapted model, saves a fixed random
   state_dict + two fixed input batches to audit/mc_tmp/io.pt;
2. parent runs the same build+load+forward locally (adapted side), 4-arg
   forecast call model(x, None, None, None);
3. child subprocess (upstream side) loads io.pt, builds Model from the
   PRISTINE sources, loads the same weights, forwards the same batches
   with the same 4-arg call; prints sha256 of the float32 output bytes;
4. parent requires byte-identical outputs from both sides, identical
   module file locations under the vendored tree, and live re-verification
   of every pinned upstream/adapted/import-diff hash.

Writes audit/model_check_report.json; this file's PASS is a hard launch
requirement (common.launch_gate).
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import torch

from common import AUDIT, CFG, HERE, atomic_write_json, launch_bindings
from model_adapter import VENDOR_ADAPTED, build_configs, setup_numerics

UPSTREAM = os.path.join(HERE, "vendor", "TimesNet", "upstream")
TMP = os.path.join(AUDIT, "mc_tmp")

_CHILD = r'''
import hashlib, json, sys
import torch
# multi-threaded CPU reductions split work by an order that varies between
# process instances (measured on theta: 2 distinct output digests across 6
# identical child runs) - single thread is required for the bitwise proof
torch.set_num_threads(1)
sys.path.insert(0, ".")
from models.TimesNet import Model
from types import SimpleNamespace
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model = Model(SimpleNamespace(**blob["attrs"]))
model.load_state_dict(blob["state_dict"], strict=True)
model.eval()
outs = []
with torch.no_grad():
    for xb in blob["inputs"]:
        outs.append(model(xb, None, None, None))
h = hashlib.sha256()
for o in outs:
    h.update(o.numpy().tobytes())
digest = h.hexdigest()
print(json.dumps({"module_file": sys.modules["models.TimesNet"].__file__,
                  "output_sha256": digest,
                  "shapes": [list(o.shape) for o in outs]}))
'''


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pinned_hashes_ok():
    v = CFG["model"]["vendor"]
    for rel, want in sorted(v["upstream_sha256"].items()):
        p = os.path.join(UPSTREAM, rel)
        if not os.path.isfile(p) or _sha256_file(p) != want:
            return False
    for rel, want in sorted(v["adapted_sha256"].items()):
        p = os.path.join(VENDOR_ADAPTED, rel)
        if not os.path.isfile(p) or _sha256_file(p) != want:
            return False
    p = os.path.join(HERE, "vendor", "TimesNet", "import_diff.patch")
    if not os.path.isfile(p) or _sha256_file(p) != v["import_diff_sha256"]:
        return False
    return True


def adapted_digest(blob):
    from types import SimpleNamespace
    from model_adapter import _import_model
    Model = _import_model()
    model = Model(SimpleNamespace(**blob["attrs"]))
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for xb in blob["inputs"]:
            outs.append(model(xb, None, None, None))
    mod_file = os.path.abspath(sys.modules["tn_models.TimesNet"].__file__)
    h = hashlib.sha256()
    for o in outs:
        h.update(o.numpy().tobytes())
    digest = h.hexdigest()
    return mod_file, digest, [list(o.shape) for o in outs]


def main():
    os.makedirs(TMP, exist_ok=True)
    setup_numerics()
    # parent side of the same single-thread requirement as _CHILD (see there)
    torch.set_num_threads(1)
    # the recorded launch bindings must describe the SAME environment the
    # other evidence reports used (one visible training device)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            "model_check must run in the pinned CUDA env "
            "(CUDA_VISIBLE_DEVICES=<training gpu>); got "
            f"available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} - refusing (its "
            f"bindings would mismatch the other reports in launch_gate)")
    torch.manual_seed(4242)
    g = torch.Generator().manual_seed(1717)
    attrs = {k: v for k, v in vars(build_configs()).items()}
    from model_adapter import instantiate_model
    model = instantiate_model("cpu")   # pin-agnostic build; weights replaced
    inputs = [torch.randn(40, 96, 1, generator=g),
              torch.randn(7, 96, 1, generator=g)]           # incl. tail batch
    blob = {"attrs": attrs, "state_dict": model.state_dict(),
            "inputs": inputs}
    torch.save(blob, os.path.join(TMP, "io.pt"))

    a_file, a_digest, shapes = adapted_digest(blob)

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0")
    out = subprocess.run([sys.executable, "-c", _CHILD,
                          os.path.join(TMP, "io.pt")], cwd=UPSTREAM,
                         capture_output=True, text=True, env=env, timeout=300)
    if out.returncode != 0:
        print(out.stdout, out.stderr, file=sys.stderr)
        raise SystemExit(f"model_check: upstream subprocess failed "
                         f"(rc={out.returncode})")
    up = json.loads(out.stdout.strip().splitlines()[-1])

    checks = {
        "upstream_output_bitwise_adapted": up["output_sha256"] == a_digest,
        "shapes_equal": up["shapes"] == shapes,
        "shapes_expected": shapes == [[40, 12, 1], [7, 12, 1]],
        "adapted_module_venv_root":
            a_file.startswith(os.path.abspath(VENDOR_ADAPTED) + os.sep),
        "upstream_module_pristine_root":
            os.path.abspath(up["module_file"]).startswith(
                os.path.abspath(UPSTREAM) + os.sep),
        "vendor_files_md5_live": _vendor_md5_ok(),
        "vendor_pinned_hashes_live": _pinned_hashes_ok(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "status": status,
              "bindings": launch_bindings(),
              "upstream_dir": UPSTREAM,
              "adapted_dir": VENDOR_ADAPTED,
              "upstream_module_file": os.path.abspath(up["module_file"]),
              "adapted_module_file": a_file,
              "output_sha256_upstream": up["output_sha256"],
              "output_sha256_adapted": a_digest,
              "output_shapes": shapes,
              "checks": checks}
    atomic_write_json(report, os.path.join(AUDIT, "model_check_report.json"))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"[model_check] status={status}")
    for k, v in checks.items():
        print(f"  {'ok' if v else 'FAIL'} {k}")
    return 0 if status == "PASS" else 1


def _vendor_md5_ok():
    import hashlib
    for rel, want in sorted(CFG["model"]["vendor"]["files_md5"].items()):
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            return False
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
