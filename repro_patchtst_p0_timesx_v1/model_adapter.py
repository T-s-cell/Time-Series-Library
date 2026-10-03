#!/usr/bin/env python3
"""Adapter for the VENDORED official PatchTST (yuqinie98/PatchTST,
PatchTST_supervised) - random init, full-param training of all learnable
parameters.

vendor/PatchTST/adapted/ holds the runtime copy (import-only package
isolation: ptst_models/ptst_layers instead of models/layers, which would
collide with the TSLib repo's own top-level packages). REPO_ROOT is NEVER
put on sys.path here. The pristine upstream/ copy exists only for the
vendor_check equivalence proof.

Single-arg forward: model(x) with x [B,96,1] float32 -> out [B,12,1];
the adapter returns [B,12]. Inputs stay in raw scale - RevIN (eps=1e-5,
var unbiased=False, affine=False) normalizes with the history's own
mean/std and denormalizes the output; no external scaler exists anywhere.

vendor_probe.py uses instantiate_model() BEFORE the pins are frozen (it must
measure, not enforce). load_model() - the only entry every real stage uses -
enforces the pinned parameter/tensor counts and the pinned non-trainable
name list (exactly the 3 official fixed attention scales; never unfrozen).
Any structural drift vs the pinned table or the vendored-file md5s raises
instead of training on a changed model core.
"""
import os
import sys
from types import SimpleNamespace

from common import CFG, CTX, DivergenceError, HERE, PRED

VENDOR_ADAPTED = os.path.join(HERE, "vendor", "PatchTST", "adapted")

_MODEL_ATTRS = ("enc_in", "seq_len", "pred_len", "e_layers", "n_heads",
                "d_model", "d_ff", "dropout", "fc_dropout", "head_dropout",
                "individual", "patch_len", "stride", "padding_patch",
                "revin", "affine", "subtract_last", "decomposition",
                "kernel_size")


def build_configs():
    m = CFG["model"]
    return SimpleNamespace(**{k: m["structure"][k] for k in _MODEL_ATTRS})


def setup_numerics():
    """Idempotent; call once before the first CUDA/torch work."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _import_model():
    if VENDOR_ADAPTED not in sys.path:
        sys.path.insert(0, VENDOR_ADAPTED)
    # `models`/`layers` of the TSLib repo must stay unimported from here:
    # the vendored packages are ptst_models/ptst_layers (unique names), so no
    # sys.modules collision is possible.
    from ptst_models.PatchTST import Model
    mod = sys.modules.get("ptst_models.PatchTST")
    src = os.path.abspath(getattr(mod, "__file__", ""))
    if not src.startswith(os.path.abspath(VENDOR_ADAPTED) + os.sep):
        raise RuntimeError(
            f"imported PatchTST from {src}, not the vendored {VENDOR_ADAPTED}"
            f" - refusing")
    return Model


def instantiate_model(device):
    """Fresh PatchTST from random init WITHOUT pin enforcement (vendor_probe
    only). Caller seeds torch BEFORE this."""
    setup_numerics()
    Model = _import_model()
    model = Model(build_configs()).float()
    return model.to(device)


def load_model(device):
    """Fresh PatchTST from random init, pins enforced. Caller seeds torch
    BEFORE this."""
    model = instantiate_model(device)
    pins = CFG["model"]["pins"]
    n_params = sum(p.numel() for p in model.parameters())
    n_tensors = len(list(model.parameters()))
    if n_params != pins["params_expected"]:
        raise RuntimeError(
            f"PatchTST parameter count {n_params} != pinned "
            f"{pins['params_expected']} - vendor drift, refusing")
    if n_tensors != pins["tensors_expected"]:
        raise RuntimeError(
            f"PatchTST tensor count {n_tensors} != pinned "
            f"{pins['tensors_expected']} - vendor drift, refusing")
    nontrainable = sorted(n for n, p in model.named_parameters()
                          if not p.requires_grad)
    if nontrainable != sorted(pins["nontrainable_expected"]):
        raise RuntimeError(
            f"non-trainable parameters {nontrainable} != pinned official "
            f"fixed scales {sorted(pins['nontrainable_expected'])} - "
            f"refusing (fixed scales stay fixed, never unfrozen)")
    return model


def forward_pred(model, xt):
    """xt [B,96,1] float32 -> pred [B,12] float32; raises on non-finite."""
    import torch
    if xt.dim() != 3 or xt.shape[1] != CTX or xt.shape[2] != 1:
        raise RuntimeError(f"input shape {tuple(xt.shape)} != (B,{CTX},1)")
    if xt.dtype != torch.float32:
        raise RuntimeError("input must be float32")
    out = model(xt)
    if tuple(out.shape) != (xt.shape[0], PRED, 1):
        raise RuntimeError(f"output shape {tuple(out.shape)} != (B,{PRED},1)")
    if not bool(torch.isfinite(out).all()):
        raise DivergenceError("non-finite model output")
    return out[..., 0]


def param_report(model):
    return {"n_params": sum(p.numel() for p in model.parameters()),
            "n_tensors": len(list(model.parameters()))}
