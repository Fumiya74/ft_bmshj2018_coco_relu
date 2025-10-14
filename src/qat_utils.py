#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qat_utils.py (STE-enabled)

Lightweight QAT helpers (FakeQuant-only) with automatic exclusion of
EntropyBottleneck / GaussianConditional / Hyperprior (h_a, h_s) neighborhood.
Supports *scoped* injection: encoder / decoder / all.

Key points:
- Uses STE (straight-through estimator) in both activation & weight fake-quant.
- Two-phase QAT control:
  * calibration: observers track stats; fake-quant forwards identity (no quant noise)
  * training: observers frozen; fake-quant injects quant/dequant with STE (grads flow)
- Optional final freeze: after `qat_freeze_after`, both observers & fake-quant are hard-frozen.

Public API:
- prepare_qat_inplace_scoped(model, scope="encoder"|"decoder"|"all",
                             act_bits=8, w_bits=8, calib_steps=500,
                             freeze_after=0, exclude_bn=True)
- set_qat_global_step(model, step)
- set_qat_mode(model, mode="calib"|"train"|"freeze")
- is_qatified(module) -> bool
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_QAT_TAG = "_is_qat_prepared"

def _module_path(m: nn.Module) -> str:
    return getattr(m, "_name_in_parent", "")

def _tag_module_tree(model: nn.Module):
    for name, m in model.named_modules():
        setattr(m, "_name_in_parent", name)

def _is_hyper_or_entropy(m: nn.Module, name: str) -> bool:
    """
    Avoid touching EntropyBottleneck, GaussianConditional, and hyperpriors (h_a/h_s) neighborhood.
    """
    lname = name.lower()
    if "entropy" in lname or "gaussianconditional" in lname:
        return True
    # common names users use for hyper analysis/synthesis
    if any(x in lname for x in ("h_a", "h_s", "hyper", "hyperprior")):
        return True
    return False

def is_qatified(module: nn.Module) -> bool:
    return bool(getattr(module, _QAT_TAG, False))

# -----------------------------------------------------------------------------
# Observers
# -----------------------------------------------------------------------------

class EMAMinMaxObserver(nn.Module):
    """
    Per-tensor EMA min/max observer for activations.
    """
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("min_val", torch.zeros(1))
        self.register_buffer("max_val", torch.zeros(1))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x_min = torch.min(x)
        x_max = torch.max(x)
        if not bool(self.initialized):
            self.min_val.copy_(x_min)
            self.max_val.copy_(x_max)
            self.initialized.fill_(True)
        else:
            self.min_val.mul_(self.momentum).add_(x_min * (1 - self.momentum))
            self.max_val.mul_(self.momentum).add_(x_max * (1 - self.momentum))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update(x)
        # numerical guard
        min_v = torch.minimum(self.min_val, self.max_val - self.eps)
        max_v = torch.maximum(self.max_val, min_v + self.eps)
        return min_v, max_v


class PerChannelSymObserver(nn.Module):
    """
    Per-channel symmetric max-abs observer for Conv weights (out_channels axis).
    """
    def __init__(self, ch_axis: int = 0, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.ch_axis = ch_axis
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("max_abs", torch.zeros(1))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, w: torch.Tensor):
        # move channel axis to 0 for reduction
        w_perm = w.transpose(0, self.ch_axis).contiguous()
        cur = w_perm.view(w_perm.shape[0], -1).abs().max(dim=1).values
        if not bool(self.initialized):
            self.max_abs = cur.detach().clone()
            self.initialized.fill_(True)
        else:
            self.max_abs.mul_(self.momentum).add_(cur * (1 - self.momentum))

        # guard against zero
        self.max_abs.clamp_(min=self.eps)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        self.update(w)
        return self.max_abs


# -----------------------------------------------------------------------------
# FakeQuant modules with STE
# -----------------------------------------------------------------------------

class FakeQuantAct(nn.Module):
    """
    Per-tensor fake-quant for activations with STE.
    Two phases:
      - calibration: pass-through (identity), observer collects stats
      - training: quant/dequant w/ STE
    """
    def __init__(self, bits: int = 8, symmetric: bool = False):
        super().__init__()
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.observer = EMAMinMaxObserver()
        # states
        self.register_buffer("calibrating", torch.tensor(True))
        self.register_buffer("frozen", torch.tensor(False))

        if self.symmetric:
            # symmetric int range (e.g., int8: -127..127)
            self.qmin = -(2 ** (self.bits - 1) - 1)
            self.qmax = (2 ** (self.bits - 1) - 1)
        else:
            # asymm uint range (0..255)
            self.qmin = 0
            self.qmax = 2 ** self.bits - 1

    def freeze(self, flag: bool = True):
        self.frozen.fill_(bool(flag))

    def set_calibrating(self, flag: bool = True):
        self.calibrating.fill_(bool(flag))

    def _calc_params(self, x: torch.Tensor):
        min_v, max_v = self.observer(x.detach())
        if self.symmetric:
            max_abs = torch.max(-min_v, max_v)
            scale = max_abs / (self.qmax if self.qmax != 0 else 1)
            zp = torch.zeros_like(scale)
        else:
            scale = (max_v - min_v) / max(self.qmax - self.qmin, 1)
            zp = torch.round(self.qmin - min_v / (scale + 1e-12))
        return scale, zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.frozen):
            return x  # fully frozen: no quant, no observer

        if bool(self.calibrating):
            # calibration phase: observer updates, but return identity
            _ = self.observer(x)
            return x

        # training phase: quant/dequant with STE
        scale, zp = self._calc_params(x)
        q = torch.round(x / (scale + 1e-12) + zp)
        q = torch.clamp(q, self.qmin, self.qmax)
        x_q = (q - zp) * scale
        # STE: forward uses x_q, backward uses identity grad wrt x
        return x + (x_q - x).detach()


class FakeQuantWPerChannelSym(nn.Module):
    """
    Per-channel symmetric fake-quant for Conv weights with STE.
    """
    def __init__(self, bits: int = 8, ch_axis: int = 0):
        super().__init__()
        self.bits = int(bits)
        self.ch_axis = int(ch_axis)
        self.observer = PerChannelSymObserver(ch_axis=ch_axis)
        self.register_buffer("calibrating", torch.tensor(True))
        self.register_buffer("frozen", torch.tensor(False))
        # symmetric range like -127..127 (restricting positive max to match negative)
        self.qmax = (2 ** (self.bits - 1) - 1)

    def freeze(self, flag: bool = True):
        self.frozen.fill_(bool(flag))

    def set_calibrating(self, flag: bool = True):
        self.calibrating.fill_(bool(flag))

    def _calc_scale(self, w: torch.Tensor) -> torch.Tensor:
        max_abs = self.observer(w.detach())  # shape: [Cout]
        scale = max_abs / max(self.qmax, 1)
        return scale.clamp(min=1e-12)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if bool(self.frozen):
            return w

        if bool(self.calibrating):
            _ = self.observer(w)
            return w

        # training phase: per-channel quant/dequant with STE
        scale = self._calc_scale(w)  # [Cout]
        # move Cout axis to dim0 to broadcast scale
        w_perm = w.transpose(0, self.ch_axis).contiguous()
        scale_v = scale.view(-1, *([1] * (w_perm.dim() - 1)))
        q = torch.round(w_perm / scale_v)
        q = torch.clamp(q, -self.qmax, self.qmax)
        w_q = q * scale_v
        w_q = w_q.transpose(0, self.ch_axis).contiguous()
        return w + (w_q - w).detach()


# -----------------------------------------------------------------------------
# Conv wrapper
# -----------------------------------------------------------------------------

class ConvWQ(nn.Conv2d):
    """
    nn.Conv2d with per-channel weight fake-quant (symmetric) and optional act fake-quant.
    """
    def __init__(self, conv: nn.Conv2d, w_bits: int = 8, act_bits: Optional[int] = 8, add_act_fakequant: bool = True):
        super().__init__(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        # copy params
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None and self.bias is not None:
                self.bias.copy_(conv.bias)

        # attach fake-quant
        self.w_fake = FakeQuantWPerChannelSym(bits=w_bits, ch_axis=0)
        self.act_fake = FakeQuantAct(bits=act_bits, symmetric=False) if (add_act_fakequant and act_bits is not None) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_fake(self.weight)
        y = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act_fake is not None:
            y = self.act_fake(y)
        return y


# -----------------------------------------------------------------------------
# QAT preparation (scoped)
# -----------------------------------------------------------------------------

def _should_wrap(name: str, scope: str) -> bool:
    lname = name.lower()
    if scope == "all":
        return True
    if scope == "encoder":
        # common encoder path identifiers
        return any(k in lname for k in ("g_a", "encoder"))
    if scope == "decoder":
        return any(k in lname for k in ("g_s", "decoder"))
    return False


def prepare_qat_inplace_scoped(model: nn.Module, scope: str = "encoder",
                               act_bits: int = 8, w_bits: int = 8,
                               calib_steps: int = 500, freeze_after: int = 0,
                               exclude_bn: bool = True, **kwargs) -> nn.Module:
    """
    Replace Conv2d with ConvWQ (weight + optional activation fake-quant) according to `scope`.
    Excludes EntropyBottleneck/GaussianConditional and hyper modules.

    Extra kwargs accepted for backward-compatibility (ignored if present):
    - encoder_attr, decoder_attr: legacy names for paths; scope-based matching is used here.
    - exclude_entropy (bool): kept for API compatibility; exclusion is always on in this impl.
    - verbose (bool): if True, prints simple summary.
    """
    assert scope in ("encoder", "decoder", "all")
    _tag_module_tree(model)

    def _replace(parent: nn.Module, name: str, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            if exclude_bn:
                # if followed by BN, we usually don't add activation fake-quant here.
                add_act = True
            else:
                add_act = True
            wrapped = ConvWQ(m, w_bits=w_bits, act_bits=act_bits, add_act_fakequant=add_act)
            setattr(parent, name, wrapped)
            return True
        return False

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if _is_hyper_or_entropy(child, full_name):
                continue
            if not _should_wrap(full_name, scope):
                continue
            _replace(parent, child_name, child)

    # attach global QAT controls on model root
    
    if kwargs.get("verbose", False):
        print(f"[qat] prepare: scope={scope} act_bits={act_bits} w_bits={w_bits} calib_steps={calib_steps} freeze_after={freeze_after}")
    setattr(model, _QAT_TAG, True)
    model.register_buffer("_qat_global_step", torch.tensor(0))
    model.register_buffer("_qat_calib_steps", torch.tensor(int(calib_steps)))
    model.register_buffer("_qat_freeze_after", torch.tensor(int(freeze_after)))

    # set initial mode: calibration
    set_qat_mode(model, "calib")
    return model


# -----------------------------------------------------------------------------
# Runtime controls
# -----------------------------------------------------------------------------

def _iter_qat_modules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (FakeQuantAct, FakeQuantWPerChannelSym, EMAMinMaxObserver, PerChannelSymObserver, ConvWQ)):
            yield m

def set_qat_global_step(model: nn.Module, step: int):
    if not is_qatified(model):
        return
    if hasattr(model, "_qat_global_step"):
        model._qat_global_step.fill_(int(step))
    _sync_qat_phase(model)

def step_qat(model: nn.Module, inc: int = 1):
    if not is_qatified(model):
        return
    step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item()) + int(inc)
    set_qat_global_step(model, step)

def set_qat_mode(model: nn.Module, mode: str = "calib"):
    """
    mode: "calib" | "train" | "freeze"
    """
    assert mode in ("calib", "train", "freeze")
    for m in _iter_qat_modules(model):
        if isinstance(m, FakeQuantAct) or isinstance(m, FakeQuantWPerChannelSym):
            if mode == "calib":
                m.set_calibrating(True); m.freeze(False)
            elif mode == "train":
                m.set_calibrating(False); m.freeze(False)
            elif mode == "freeze":
                m.set_calibrating(False); m.freeze(True)

def _sync_qat_phase(model: nn.Module):
    step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
    calib_steps = int(getattr(model, "_qat_calib_steps", torch.tensor(0)).item())
    freeze_after = int(getattr(model, "_qat_freeze_after", torch.tensor(0)).item())

    if freeze_after > 0 and step >= freeze_after:
        set_qat_mode(model, "freeze")
    elif step < calib_steps:
        set_qat_mode(model, "calib")
    else:
        set_qat_mode(model, "train")

def apply_qat_phase_by_step(model: nn.Module, global_step: int):
    """
    Convenience: call once per training step.
    """
    set_qat_global_step(model, int(global_step))


# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------

def print_qat_summary(model: nn.Module):
    if not is_qatified(model):
        print("QAT not prepared.")
        return
    step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
    calib_steps = int(getattr(model, "_qat_calib_steps", torch.tensor(0)).item())
    freeze_after = int(getattr(model, "_qat_freeze_after", torch.tensor(0)).item())
    print(f"[QAT] step={step} calib_steps={calib_steps} freeze_after={freeze_after}")
    print("Modules:")
    for n, m in model.named_modules():
        if isinstance(m, ConvWQ):
            print(f"  {n}: ConvWQ(w_bits={m.w_fake.bits}, act_bits={m.act_fake.bits if m.act_fake else None})")
        elif isinstance(m, FakeQuantAct):
            print(f"  {n}: FakeQuantAct(bits={m.bits}, sym={m.symmetric})")
        elif isinstance(m, FakeQuantWPerChannelSym):
            print(f"  {n}: FakeQuantWPerChannelSym(bits={m.bits}, ch_axis={m.ch_axis})")
        elif isinstance(m, (EMAMinMaxObserver, PerChannelSymObserver)):
            print(f"  {n}: {m.__class__.__name__}")

# Backward-compat wrapper for older train.py calls
def step_qat_schedule(model: nn.Module, global_step: int):
    apply_qat_phase_by_step(model, global_step)
