#!/usr/bin/env python3
"""Adapter for the VENDORED official thuml TimesNet forecasting path -
random init, full-param training of all learnable parameters.

vendor/TimesNet/adapted/ holds the runtime copy (import-only package
isolation: tn_models/tn_layers instead of models/layers, which would
collide with the TSLib repo's own top-level packages). REPO_ROOT is NEVER
put on sys.path here. The pristine upstream/ copy exists only for the
model_check equivalence proof. Adaptation is exactly 2 import lines
(see vendor/TimesNet/import_diff.patch); state-dict keys carry no package
names, so weights transfer 1:1 - proven bitwise by model_check.py.

TWO forward entry points, mandated by the frozen protocol:

- forward_train(model, xt): any batch size. Used ONLY by the training
  update loop and by the preflight numerics/backward diagnostics that
  exercise training-style batches. TimesNet's FFT_for_Period selects
  top-k periods from the BATCH-MEAN amplitude spectrum
  (abs(rfft(x)).mean(0)), so with batch>1 the selected periods couple
  every window in the batch.

- forward_pred(model, xt): evaluation ONLY, hard-asserts B==1. Every
  evaluation forward in the protocol (per-epoch validation, final test,
  recheck replay, epoch-0 diagnostic) runs through here, so each
  window's period selection is a function of that window alone. A batch
  of identical copies of one window reproduces that window's own
  spectrum, but heterogeneous batches do not - hence B=1 is the only
  batch composition with a window-local meaning. Fixed before any
  results were seen; disclosed in aggregate.py.

Single internal normalization: mean/var(unbiased=False) + 1e-5 inside
the sqrt, computed from the window itself and detached - non-adaptive.
The adapter therefore adds NO second normalization; inputs stay raw.

vendor_probe.py uses instantiate_model() BEFORE the pins are frozen (it
must measure, not enforce). load_model() - the only entry every real
stage uses - enforces the pinned parameter/tensor counts and that the
non-trainable list is EMPTY (all 32 tensors train; the only buffer is
the fixed positional pe). Gradient presence is NOT checked here: a
freshly built model has .grad None on every parameter by construction;
the exactly-one-grad-None assertion (temporal_embedding.embed.weight,
unused when x_mark=None) is made AFTER a real backward pass in
preflight section B. Any structural drift vs the pinned table or the
vendored-file md5s raises instead of training on a changed model core.
"""
import os
import sys
from types import SimpleNamespace

from common import CFG, CTX, DivergenceError, HERE, PRED

VENDOR_ADAPTED = os.path.join(HERE, "vendor", "TimesNet", "adapted")

_MODEL_ATTRS = ("task_name", "enc_in", "dec_in", "c_out", "seq_len",
                "label_len", "pred_len", "e_layers", "d_model", "d_ff",
                "top_k", "num_kernels", "dropout", "embed", "freq")


def build_configs():
    m = CFG["model"]
    ns = SimpleNamespace(**{k: m["structure"][k] for k in _MODEL_ATTRS})
    # activation is disclosed in configs (superset) but not read by the
    # vendored forecasting path; expose it anyway for probe completeness.
    ns.activation = m["structure"]["activation"]
    return ns


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
    # the vendored packages are tn_models/tn_layers (unique names), so no
    # sys.modules collision is possible.
    from tn_models.TimesNet import Model
    mod = sys.modules.get("tn_models.TimesNet")
    src = os.path.abspath(getattr(mod, "__file__", ""))
    if not src.startswith(os.path.abspath(VENDOR_ADAPTED) + os.sep):
        raise RuntimeError(
            f"imported TimesNet from {src}, not the vendored {VENDOR_ADAPTED}"
            f" - refusing")
    return Model


def instantiate_model(device):
    """Fresh TimesNet from random init WITHOUT pin enforcement (vendor_probe
    only). Caller seeds torch BEFORE this."""
    setup_numerics()
    Model = _import_model()
    model = Model(build_configs()).float()
    return model.to(device)


def load_model(device):
    """Fresh TimesNet from random init, pins enforced. Caller seeds torch
    BEFORE this. No .grad assertions here - a fresh model has every
    .grad None; the backward-time grad check lives in preflight B."""
    model = instantiate_model(device)
    pins = CFG["model"]["pins"]
    n_params = sum(p.numel() for p in model.parameters())
    n_tensors = len(list(model.parameters()))
    if n_params != pins["params_expected"]:
        raise RuntimeError(
            f"TimesNet parameter count {n_params} != pinned "
            f"{pins['params_expected']} - vendor drift, refusing")
    if n_tensors != pins["tensors_expected"]:
        raise RuntimeError(
            f"TimesNet tensor count {n_tensors} != pinned "
            f"{pins['tensors_expected']} - vendor drift, refusing")
    nontrainable = sorted(n for n, p in model.named_parameters()
                          if not p.requires_grad)
    if nontrainable != sorted(pins["nontrainable_expected"]):
        raise RuntimeError(
            f"non-trainable parameters {nontrainable} != pinned "
            f"{sorted(pins['nontrainable_expected'])} - refusing "
            f"(pinned expectation is zero non-trainable parameters)")
    return model


def _check_input(xt):
    import torch
    if xt.dim() != 3 or xt.shape[1] != CTX or xt.shape[2] != 1:
        raise RuntimeError(f"input shape {tuple(xt.shape)} != (B,{CTX},1)")
    if xt.dtype != torch.float32:
        raise RuntimeError("input must be float32")


def _check_output(out, b):
    import torch
    if tuple(out.shape) != (b, PRED, 1):
        raise RuntimeError(f"output shape {tuple(out.shape)} != ({b},{PRED},1)")
    if not bool(torch.isfinite(out).all()):
        raise DivergenceError("non-finite model output")


def forward_pred(model, xt):
    """EVALUATION ONLY. xt [1,96,1] float32 -> pred [1,12] float32.

    Hard-asserts batch size 1: TimesNet's FFT top-k period selection is
    computed from the batch-mean spectrum, so any B>1 would couple the
    windows' predictions. Every evaluation entry point in the protocol
    must route through this function."""
    import torch
    _check_input(xt)
    if xt.shape[0] != 1:
        raise RuntimeError(
            f"forward_pred requires batch size 1 (FFT top-k is batch-"
            f"dependent); got B={xt.shape[0]} - evaluation protocol is "
            f"window-local by frozen rule eval_batch_size_1")
    out = model(xt, None, None, None)
    _check_output(out, 1)
    return out[..., 0]


def forward_train(model, xt):
    """TRAINING ONLY (and training-style preflight diagnostics). xt
    [B,96,1] float32, any B >= 1 -> pred [B,12] float32. Batch-mean FFT
    spectrum is intended here: the update gradient is defined on the
    protocol batch of 32."""
    import torch
    _check_input(xt)
    if xt.shape[0] < 1:
        raise RuntimeError("forward_train requires B >= 1")
    out = model(xt, None, None, None)
    _check_output(out, xt.shape[0])
    return out[..., 0]


def param_report(model):
    return {"n_params": sum(p.numel() for p in model.parameters()),
            "n_tensors": len(list(model.parameters()))}
