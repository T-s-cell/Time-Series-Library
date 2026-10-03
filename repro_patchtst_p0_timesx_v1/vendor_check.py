#!/usr/bin/env python3
"""vendor_check: prove the adapted vendored PatchTST is the upstream model.

The adapted/ tree differs from upstream/ ONLY by import-line package
isolation (ptst_models/ptst_layers vs models/layers) and directory renames;
state_dict keys contain no package names, so weights transfer 1:1. Because
BOTH variants occupy the `models`/`layers` namespace, they can never be
imported into one process - the upstream reference therefore runs in an
isolated subprocess with cwd/sys.path rooted at vendor/PatchTST/upstream.

Protocol (CPU, eval, no_grad, deterministic):
1. parent seeds torch, builds the adapted model, saves a fixed random
   state_dict + two fixed input batches to audit/vc_tmp/io.pt;
2. parent runs the same build+load+forward locally (adapted side);
3. child subprocess (upstream side) loads io.pt, builds Model from the
   PRISTINE sources, loads the same weights, forwards the same batches;
   prints sha256 of the float32 output bytes;
4. parent requires byte-identical outputs from both sides and identical
   module file locations under the vendored tree.

Writes audit/vendor_check_report.json; this file's PASS is a hard launch
requirement (common.launch_gate).
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from common import AUDIT, CFG, HERE, atomic_write_json, launch_bindings
from model_adapter import VENDOR_ADAPTED, build_configs, setup_numerics

UPSTREAM = os.path.join(HERE, "vendor", "PatchTST", "upstream")
TMP = os.path.join(AUDIT, "vc_tmp")

_CHILD = r'''
import hashlib, json, sys
import torch
sys.path.insert(0, ".")
from models.PatchTST import Model
from types import SimpleNamespace
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
model = Model(SimpleNamespace(**blob["attrs"]))
model.load_state_dict(blob["state_dict"], strict=True)
model.eval()
outs = []
with torch.no_grad():
    for xb in blob["inputs"]:
        outs.append(model(xb))
h = hashlib.sha256()
for o in outs:
    h.update(o.numpy().tobytes())
digest = h.hexdigest()
print(json.dumps({"module_file": sys.modules["models.PatchTST"].__file__,
                  "output_sha256": digest,
                  "shapes": [list(o.shape) for o in outs]}))
'''


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
            outs.append(model(xb))
    mod_file = os.path.abspath(sys.modules["ptst_models.PatchTST"].__file__)
    h = hashlib.sha256()
    for o in outs:
        h.update(o.numpy().tobytes())
    digest = h.hexdigest()
    return mod_file, digest, [list(o.shape) for o in outs]


def main():
    os.makedirs(TMP, exist_ok=True)
    setup_numerics()
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
        raise SystemExit(f"vendor_check: upstream subprocess failed "
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
    atomic_write_json(report, os.path.join(AUDIT, "vendor_check_report.json"))
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"[vendor_check] status={status}")
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
