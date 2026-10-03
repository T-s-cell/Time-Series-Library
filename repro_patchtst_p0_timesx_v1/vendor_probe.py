#!/usr/bin/env python3
"""vendor_probe: MEASURE the PatchTST structure pins on this machine/GPU.

Runs BEFORE configs/v1.yaml pins are frozen (they may still be 0/[]
placeholders): instantiate_model() deliberately bypasses the pin enforcement
of load_model(), because this script's output IS the source of those pins.
Workflow: vendor_probe -> fill pins into configs/v1.yaml -> prepare ->
preflight (whose section B then enforces every value measured here).

Measured and recorded per device (cpu + cuda when available):
- n_params / n_tensors / state_dict entries / named_buffers entries
- non-trainable parameter names (expected: the 3 official fixed attention
  scales; they are reported, never unfrozen)
- grad-None names after backward on a real-shaped synthetic batch
- BatchNorm buffers present in state_dict (running_mean/var +
  num_batches_tracked per norm layer)
- derived structure: W_pos (patch_num x d_model), head_nf, head linear
  in/out features, RevIN eps

Uses ONLY synthetic inputs - no snapshot access, no data dependency, safe to
run before prepare. Writes audit/vendor_probe.json.
"""
import datetime
import json
import os
import sys

import numpy as np

from common import AUDIT, atomic_write_json
from model_adapter import VENDOR_ADAPTED, forward_pred, instantiate_model


def probe(device):
    import torch
    rec = {"device": device}
    torch.manual_seed(2021)
    if device != "cpu":
        torch.cuda.manual_seed_all(2021)
    model = instantiate_model(device)
    rec["module_file"] = os.path.abspath(
        sys.modules["ptst_models.PatchTST"].__file__)
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    rec["n_tensors"] = len(list(model.parameters()))
    rec["nontrainable"] = sorted(n for n, p in model.named_parameters()
                                 if not p.requires_grad)
    rec["state_dict_entries"] = len(model.state_dict())
    rec["n_named_buffers"] = len(list(model.named_buffers()))
    rec["bn_buffers"] = sorted(n for n, _ in model.named_buffers()
                               if "running_" in n or "num_batches" in n)
    bb = model.model                      # PatchTSTBackbone
    enc = bb.backbone                     # TSTiEncoder (channel-independent)
    rec["w_pos_shape"] = list(enc.W_pos.shape)
    rec["head_nf"] = int(bb.head_nf)
    rec["head_linear"] = [int(bb.head.linear.in_features),
                          int(bb.head.linear.out_features)]
    rec["revin_eps"] = float(bb.revin_layer.eps)
    rec["revin_affine"] = bool(bb.revin_layer.affine)

    xt = torch.randn(32, 96, 1, device=device if device != "cpu" else "cpu")
    if device != "cpu":
        xt = xt.to(device)
    yt = torch.randn(32, 12, device=device if device != "cpu" else "cpu")
    model.train()
    pred = forward_pred(model, xt)
    loss = torch.mean((pred - yt) ** 2)
    loss.backward()
    rec["grad_none_names"] = sorted(n for n, p in model.named_parameters()
                                    if p.grad is None)
    rec["grad_bearing"] = sum(1 for p in model.parameters()
                              if p.grad is not None)
    sd = model.state_dict()
    rec["state_dict_has_all_buffers"] = all(
        n in sd for n, _ in model.named_buffers())
    return rec


def main():
    import torch
    os.makedirs(AUDIT, exist_ok=True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ"),
              "vendor_adapted_dir": VENDOR_ADAPTED,
              "torch": torch.__version__,
              "devices": {}}
    for dev in devices:
        report["devices"][dev] = probe(dev)
    if "cuda" in report["devices"] and "cpu" in report["devices"]:
        a, b = report["devices"]["cpu"], report["devices"]["cuda"]
        report["cpu_cuda_consistent"] = all(
            a[k] == b[k] for k in
            ("n_params", "n_tensors", "nontrainable", "state_dict_entries",
             "grad_none_names", "grad_bearing", "w_pos_shape", "head_nf",
             "head_linear", "revin_eps"))
    atomic_write_json(report, os.path.join(AUDIT, "vendor_probe.json"))
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
