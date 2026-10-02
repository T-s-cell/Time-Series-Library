#!/usr/bin/env python3
"""Adapter for the repo's own models/TimeMixer.py (READ-ONLY reuse).

Raw numeric input [B,96,1] -> prediction [B,12]; x_mark_enc/x_dec/x_mark_dec
are None (official no-temporal-branch path). FP32, TF32 both off, AMP off,
deterministic algorithms on. Any structural drift vs the pinned parameter
table raises instead of training on a changed model core.
"""
import sys
from types import SimpleNamespace

from common import CFG, CTX, PRED, REPO_ROOT, DivergenceError

_MODEL_ATTRS = ("task_name", "seq_len", "label_len", "pred_len",
                "down_sampling_window", "down_sampling_layers",
                "down_sampling_method", "channel_independence", "e_layers",
                "d_model", "d_ff", "decomp_method", "moving_avg", "enc_in",
                "dec_in", "c_out", "embed", "freq", "dropout", "use_norm",
                "top_k")


def build_configs():
    m = CFG["model"]
    return SimpleNamespace(**{k: m[k] for k in _MODEL_ATTRS})


def setup_numerics():
    """Idempotent; call once before the first CUDA/torch work."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_model(device):
    """Fresh TimeMixer from random init. Caller seeds torch BEFORE this."""
    setup_numerics()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from models.TimeMixer import Model
    model = Model(build_configs()).float()
    n_params = sum(p.numel() for p in model.parameters())
    n_tensors = len(list(model.parameters()))
    if n_params != CFG["model"]["params_expected"]:
        raise RuntimeError(
            f"TimeMixer parameter count {n_params} != pinned "
            f"{CFG['model']['params_expected']} - TSLib core drift, refusing")
    if n_tensors != CFG["model"]["tensors_expected"]:
        raise RuntimeError(
            f"TimeMixer tensor count {n_tensors} != pinned "
            f"{CFG['model']['tensors_expected']} - TSLib core drift, refusing")
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("non-trainable parameter found (full finetune)")
    return model.to(device)


def forward_pred(model, xt):
    """xt [B,96,1] float32 -> pred [B,12] float32; raises on non-finite."""
    import torch
    if xt.dim() != 3 or xt.shape[1] != CTX or xt.shape[2] != 1:
        raise RuntimeError(f"input shape {tuple(xt.shape)} != (B,{CTX},1)")
    if xt.dtype != torch.float32:
        raise RuntimeError("input must be float32")
    out = model(xt, None, None, None)
    if tuple(out.shape) != (xt.shape[0], PRED, 1):
        raise RuntimeError(f"output shape {tuple(out.shape)} != (B,{PRED},1)")
    if not bool(torch.isfinite(out).all()):
        raise DivergenceError("non-finite model output",
                              epoch=getattr(model, "_tm_epoch", None))
    return out[..., 0]


def param_report(model):
    return {"n_params": sum(p.numel() for p in model.parameters()),
            "n_tensors": len(list(model.parameters()))}
