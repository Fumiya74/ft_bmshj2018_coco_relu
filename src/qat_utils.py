#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight QAT helpers (FakeQuant-only) with automatic exclusion of
EntropyBottleneck / GaussianConditional / Hyperprior (h_a, h_s) neighborhood.
Now supports *scoped* injection: encoder / decoder / all.
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

# =====================
# Observers
# =====================
class EMAMinMaxObserver(nn.Module):
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("frozen", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if int(self.frozen.item()) == 1:
            return
        x = x.detach()
        x_min = torch.min(x)
        x_max = torch.max(x)
        if not torch.isfinite(self.min_val):
            self.min_val.copy_(x_min)
        if not torch.isfinite(self.max_val):
            self.max_val.copy_(x_max)
        self.min_val.mul_(self.momentum).add_(x_min * (1 - self.momentum))
        self.max_val.mul_(self.momentum).add_(x_max * (1 - self.momentum))

    def freeze(self, flag: bool = True):
        self.frozen.fill_(1 if flag else 0)

    def forward(self, x: torch.Tensor):
        self.update(x)
        min_v = torch.minimum(self.min_val, self.max_val - self.eps)
        max_v = torch.maximum(self.max_val, min_v + self.eps)
        return min_v, max_v


class PerChannelSymObserver(nn.Module):
    def __init__(self, ch_axis: int = 0, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.ch_axis = ch_axis
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("max_abs", torch.zeros(1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("frozen", torch.tensor(0, dtype=torch.int32))

    @torch.no_grad()
    def update(self, w: torch.Tensor):
        if int(self.frozen.item()) == 1:
            return
        max_abs_now = torch.amax(torch.abs(w.detach()), dim=[d for d in range(w.dim()) if d != self.ch_axis])
        if int(self.initialized.item()) == 0:
            self.max_abs = max_abs_now.clone()
            self.initialized.fill_(1)
        else:
            self.max_abs.mul_(self.momentum).add_(max_abs_now * (1 - self.momentum))

    def freeze(self, flag: bool = True):
        self.frozen.fill_(1 if flag else 0)

    def forward(self, w: torch.Tensor):
        self.update(w)
        return torch.clamp(self.max_abs, min=self.eps)


# =====================
# FakeQuant modules
# =====================
class FakeQuantAct(nn.Module):
    def __init__(self, num_bits: int = 8, momentum: float = 0.99):
        super().__init__()
        self.observer = EMAMinMaxObserver(momentum=momentum)
        self.qmin = 0
        self.qmax = (1 << num_bits) - 1
        self.enabled = True

    def freeze(self, flag: bool = True):
        self.observer.freeze(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.enabled:
            return x
        min_v, max_v = self.observer(x)
        scale = (max_v - min_v) / max(self.qmax - self.qmin, 1)
        scale = torch.clamp(scale, min=1e-8)
        zp = torch.clamp(torch.round(self.qmin - min_v / scale), self.qmin, self.qmax)
        q = torch.round(x / scale + zp)
        q = torch.clamp(q, self.qmin, self.qmax)
        return (q - zp) * scale


class FakeQuantWPerChannelSym(nn.Module):
    def __init__(self, ch_axis: int = 0, num_bits: int = 8, momentum: float = 0.99):
        super().__init__()
        self.observer = PerChannelSymObserver(ch_axis=ch_axis, momentum=momentum)
        self.qmax = (1 << (num_bits - 1)) - 1
        self.enabled = True
        self.ch_axis = ch_axis

    def freeze(self, flag: bool = True):
        self.observer.freeze(flag)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.enabled:
            return w
        max_abs = self.observer(w)
        scale = torch.clamp(max_abs / self.qmax, min=1e-8)
        view = [1] * w.dim()
        view[self.ch_axis] = -1
        scale_v = scale.view(*view)
        q = torch.round(w / scale_v)
        q = torch.clamp(q, -self.qmax-1, self.qmax)
        return q * scale_v


# =====================
# Exclusion logic
# =====================
EXCLUDE_CLASS_SUBSTR = (
    "EntropyBottleneck",
    "GaussianConditional",
)
EXCLUDE_NAME_SUBSTR = (
    "entropy_bottleneck",
    "gaussian_conditional",
    "h_a",
    "h_s",
    "hyper",
    "entropy",
)

def _is_entropy_neighborhood(name_path: str, module: nn.Module) -> bool:
    cname = module.__class__.__name__
    for s in EXCLUDE_CLASS_SUBSTR:
        if s.lower() in cname.lower():
            return True
    for s in EXCLUDE_NAME_SUBSTR:
        if s.lower() in name_path.lower():
            return True
    return False


# =====================
# Injection helpers
# =====================
def _iter_named_modules(model: nn.Module, prefix: str = ""):
    for name, child in model.named_children():
        path = f"{prefix}.{name}" if prefix else name
        yield path, child
        yield from _iter_named_modules(child, path)

def _wrap_conv_with_wfq(module: nn.Conv2d) -> nn.Module:
    class ConvWQ(nn.Conv2d):
        def __init__(self, base: nn.Conv2d):
            super().__init__(
                in_channels=base.in_channels,
                out_channels=base.out_channels,
                kernel_size=base.kernel_size,
                stride=base.stride,
                padding=base.padding,
                dilation=base.dilation,
                groups=base.groups,
                bias=(base.bias is not None),
                padding_mode=base.padding_mode,
            )
            self.load_state_dict(base.state_dict(), strict=True)
            self._fq = FakeQuantWPerChannelSym(ch_axis=0)
        def freeze(self, flag: bool = True):
            self._fq.freeze(flag)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self._fq(self.weight)
            return nn.functional.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
    return ConvWQ(module)

def _append_act_fq(parent: nn.Module, name: str, act_mod: nn.Module):
    seq = nn.Sequential(act_mod, FakeQuantAct())
    setattr(parent, name, seq)

def _inject_qat_under(module: nn.Module, *, exclude_entropy: bool, verbose: bool = False):
    replaced_conv = 0
    appended_act = 0
    for path, mod in list(_iter_named_modules(module)):
        if exclude_entropy and _is_entropy_neighborhood(path, mod):
            setattr(mod, "_qat_excluded", True)
            continue
        if isinstance(mod, nn.Conv2d):
            parent_path = ".".join(path.split(".")[:-1])
            leaf_name = path.split(".")[-1]
            parent = module
            if parent_path:
                for seg in parent_path.split("."):
                    parent = getattr(parent, seg)
            setattr(parent, leaf_name, _wrap_conv_with_wfq(mod))
            replaced_conv += 1
        elif isinstance(mod, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
            parent_path = ".".join(path.split(".")[:-1])
            leaf_name = path.split(".")[-1]
            parent = module
            if parent_path:
                for seg in parent_path.split("."):
                    parent = getattr(parent, seg)
            _append_act_fq(parent, leaf_name, mod)
            appended_act += 1
    if verbose:
        print(f"[QAT] injected under scope: Conv(w-fq)={replaced_conv}, Act(fq)={appended_act}, exclude_entropy={exclude_entropy}")
    return replaced_conv, appended_act


# =====================
# Public APIs
# =====================
def prepare_qat_inplace(
    model: nn.Module,
    exclude_entropy: bool = True,
    calib_steps: int = 2000,
    freeze_after: int = 8000,
    verbose: bool = True,
) -> nn.Module:
    """Global (model-wide) injection. Kept for backward compatibility."""
    model.register_buffer("_qat_calib_steps", torch.tensor(int(calib_steps)))
    model.register_buffer("_qat_freeze_after", torch.tensor(int(freeze_after)))
    model.register_buffer("_qat_global_step", torch.tensor(0, dtype=torch.long))

    _inject_qat_under(model, exclude_entropy=exclude_entropy, verbose=verbose)
    return model


def prepare_qat_inplace_scoped(
    model: nn.Module,
    scope: str = "all",
    encoder_attr: str = "g_a",
    decoder_attr: str = "g_s",
    exclude_entropy: bool = True,
    calib_steps: int = 2000,
    freeze_after: int = 8000,
    verbose: bool = True,
) -> nn.Module:
    """
    Scoped injection. scope in {"encoder","decoder","all"}
    If encoder_attr/decoder_attr are not present, falls back to whole model.
    """
    scope = scope.lower()
    assert scope in {"encoder", "decoder", "all"}
    model.register_buffer("_qat_calib_steps", torch.tensor(int(calib_steps)))
    model.register_buffer("_qat_freeze_after", torch.tensor(int(freeze_after)))
    model.register_buffer("_qat_global_step", torch.tensor(0, dtype=torch.long))

    total_conv, total_act = 0, 0

    def inject_into(sub: Optional[nn.Module]):
        nonlocal total_conv, total_act
        if sub is None:
            return
        c, a = _inject_qat_under(sub, exclude_entropy=exclude_entropy, verbose=verbose)
        total_conv += c; total_act += a

    has_enc = hasattr(model, encoder_attr) and isinstance(getattr(model, encoder_attr), nn.Module)
    has_dec = hasattr(model, decoder_attr) and isinstance(getattr(model, decoder_attr), nn.Module)

    if scope in {"encoder", "all"}:
        inject_into(getattr(model, encoder_attr) if has_enc else model if not has_dec else getattr(model, encoder_attr))
    if scope in {"decoder", "all"}:
        inject_into(getattr(model, decoder_attr) if has_dec else model if not has_enc else getattr(model, decoder_attr))

    if verbose:
        print(f"[QAT] scoped='{scope}' injected: Conv(w-fq)={total_conv}, Act(fq)={total_act}, exclude_entropy={exclude_entropy}")
    return model


@torch.no_grad()
def step_qat_schedule(model: nn.Module, global_step: int):
    """Calibrate first, then freeze observers; simple 2-phase schedule."""
    if hasattr(model, "_qat_global_step"):
        model._qat_global_step.fill_(int(global_step))
    calib_steps = int(getattr(model, "_qat_calib_steps", torch.tensor(0)).item())
    freeze_after = int(getattr(model, "_qat_freeze_after", torch.tensor(0)).item())

    def _apply(m: nn.Module):
        if isinstance(m, (FakeQuantAct, FakeQuantWPerChannelSym, EMAMinMaxObserver, PerChannelSymObserver)):
            step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
            if step >= freeze_after:
                if hasattr(m, "freeze"): m.freeze(True)
            elif step >= calib_steps:
                if hasattr(m, "freeze"): m.freeze(True)
            else:
                if hasattr(m, "freeze"): m.freeze(False)
    model.apply(_apply)
