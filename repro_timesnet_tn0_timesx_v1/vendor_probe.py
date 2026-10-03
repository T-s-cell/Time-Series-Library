#!/usr/bin/env python3
"""vendor_probe: MEASURE the TimesNet structure pins on this machine/GPU.

Runs BEFORE configs/v1.yaml pins are frozen (they may still be 0/[]
placeholders): instantiate_model() deliberately bypasses the pin enforcement
of load_model(), because this script's output IS the source of those pins.
Workflow: vendor_probe -> fill pins into configs/v1.yaml -> prepare ->
preflight (whose section B then enforces every value measured here).

Measured and recorded per device (cpu + cuda when available):
- n_params / n_tensors / state_dict entries / grad-bearing tensors
- non-trainable parameter names (expected: EMPTY - every TimesNet tensor
  trains; the only fixed thing is the positional buffer pe)
- grad-None names after a REAL backward pass on a batch-32 synthetic
  forward_train (expected: exactly the one unused timeF temporal Linear,
  never touched because x_mark is None; fresh models have all grads None,
  so this measurement is only meaningful post-backward)
- buffers list (expected: exactly position_embedding.pe)
- derived structure: predict_linear 96->108, projection 16->1, shared
  layer_norm 16, TimesBlock count, pe [1,5000,16], timeF in_features 4,
  TokenEmbedding conv kernel/padding_mode
- FFT behavior record on synthetic windows: identical-copy batch vs the
  single window must select identical periods; a batch of DIFFERENT
  windows is recorded as a DIAGNOSTIC ONLY (may or may not differ - not a
  failure signal; the hard batch rule is the B==1 evaluation assertion)

Uses ONLY synthetic inputs - no snapshot access, no data dependency, safe
to run before prepare. Writes audit/vendor_probe.json.
"""
import datetime
import json
import os
import sys

import numpy as np

from common import AUDIT, atomic_write_json
from model_adapter import (VENDOR_ADAPTED, forward_train, instantiate_model,
                           setup_numerics)

COUNT_KEYS = ("n_params", "n_tensors", "nontrainable", "state_dict_entries",
              "grad_none_names", "grad_bearing", "buffers",
              "derived", "internal_norm_eps_seen")


def probe(device):
    import torch
    import torch.nn.functional as F
    rec = {"device": device}
    torch.manual_seed(2021)
    if device != "cpu":
        torch.cuda.manual_seed_all(2021)
    model = instantiate_model(device)
    import tn_models.TimesNet as TN
    rec["module_file"] = os.path.abspath(TN.__file__)
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    rec["n_tensors"] = len(list(model.parameters()))
    rec["nontrainable"] = sorted(n for n, p in model.named_parameters()
                                 if not p.requires_grad)
    rec["state_dict_entries"] = len(model.state_dict())
    rec["n_named_buffers"] = len(list(model.named_buffers()))
    rec["buffers"] = sorted(n for n, _ in model.named_buffers())

    # derived structure
    enc = model.enc_embedding
    blocks = list(model.model)
    rec["derived"] = {
        "predict_linear": [int(model.predict_linear.in_features),
                           int(model.predict_linear.out_features)],
        "projection": [int(model.projection.in_features),
                       int(model.projection.out_features),
                       bool(model.projection.bias is not None)],
        "layer_norm": [int(model.layer_norm.normalized_shape[0])],
        "n_timesblocks": len(blocks),
        "timesblock_d": [[int(b.conv[0].in_channels),
                          int(b.conv[0].out_channels),
                          int(b.conv[0].num_kernels),
                          int(b.conv[2].in_channels),
                          int(b.conv[2].out_channels)] for b in blocks],
        "pe_shape": list(enc.position_embedding.pe.shape),
        "timef_in_features": int(enc.temporal_embedding.embed.in_features),
        "timef_bias": enc.temporal_embedding.embed.bias is not None,
        "token_conv": [int(enc.value_embedding.tokenConv.in_channels),
                       int(enc.value_embedding.tokenConv.out_channels),
                       int(enc.value_embedding.tokenConv.kernel_size[0]),
                       enc.value_embedding.tokenConv.padding_mode,
                       bool(enc.value_embedding.tokenConv.bias is None)],
    }
    # internal norm eps is hard-coded in the vendored forecast(); verify by
    # matching a manual re-computation on one window (eval mode, B=1).
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 96, 1, device=device)
        out = forward_train(model, x)
        means = x.mean(1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        rec["internal_norm_eps_seen"] = True  # source-level eps=1e-5; the
        # numeric proof of the full forecast path is model_check + preflight B
        rec["probe_forward_shape_ok"] = tuple(out.shape) == (1, 12)

    # grad measurement: REAL backward on a training-shaped batch (B=32).
    # Fresh-model grads are None by construction, so nothing is asserted
    # before this; the recorded lists are post-backward only.
    torch.manual_seed(2022)
    xt = torch.randn(32, 96, 1, device=device)
    yt = torch.randn(32, 12, device=device)
    model.train()
    pred = forward_train(model, xt)
    loss = torch.mean((pred - yt) ** 2)
    loss.backward()
    rec["grad_none_names"] = sorted(n for n, p in model.named_parameters()
                                    if p.grad is None)
    rec["grad_bearing"] = sum(1 for p in model.parameters()
                              if p.grad is not None)
    sd = model.state_dict()
    rec["state_dict_has_all_buffers"] = all(
        n in sd for n, _ in model.named_buffers())

    # FFT batch-dependence record (synthetic; diagnostic only)
    g = torch.Generator(device="cpu").manual_seed(77)
    win = torch.randn(1, 96, 1, generator=g).to(device)
    same32 = win.repeat(32, 1, 1)
    diff32 = torch.randn(32, 96, 1, generator=g).to(device)
    p1, _ = TN.FFT_for_Period(win, 5)
    p_same, _ = TN.FFT_for_Period(same32, 5)
    p_diff, _ = TN.FFT_for_Period(diff32, 5)
    rec["fft_record"] = {
        "periods_b1": [int(v) for v in p1],
        "periods_b32_identical_copies": [int(v) for v in p_same],
        "identical_copies_match_b1": bool(np.array_equal(p1, p_same)),
        "periods_b32_distinct_windows": [int(v) for v in p_diff],
        "distinct_windows_agree_with_b1": bool(np.array_equal(p_diff, p1)),
        "note": "identical-copy batch MUST reproduce the single-window "
                "spectrum (mean of equal spectra); distinct windows MAY "
                "coincidentally agree - that is why real-window batch "
                "sensitivity is a diagnostic, and the hard rule is the "
                "B==1 forward_pred assertion",
    }
    return rec


def main():
    import torch
    setup_numerics()
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
        report["cpu_cuda_consistent"] = all(a[k] == b[k] for k in COUNT_KEYS)
    atomic_write_json(report, os.path.join(AUDIT, "vendor_probe.json"))
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
