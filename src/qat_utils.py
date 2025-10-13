# -*- coding: utf-8 -*-
"""
Lightweight QAT utilities (FakeQuant-only, no FX convert)
- prepare_qat_inplace(model, ...): insert activation & weight fakequants around Conv/Linear
- set_qat_observers_enabled(model, enabled)
- set_qat_fakequant_enabled(model, enabled)
This aims to stabilize training for NPU/INT8 deployment while keeping the graph export simple.
"""

from typing import Optional
import torch
import torch.nn as nn

# --- Simple FakeQuant + Observer modules ---

def _make_act_observer(kind: str = "ema"):
    if kind == "minmax":
        return torch.ao.quantization.MinMaxObserver(quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    # ema (default)
    return torch.ao.quantization.MovingAverageMinMaxObserver(quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

def _make_weight_observer(per_channel: bool = True):
    if per_channel:
        return torch.ao.quantization.PerChannelMinMaxObserver(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
    return torch.ao.quantization.MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

class ActFakeQuant(nn.Module):
    def __init__(self, observer_kind: str = "ema"):
        super().__init__()
        self.observer = _make_act_observer(observer_kind)
        self.fake_q = torch.ao.quantization.FakeQuantize(observer=self.observer, quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    def enable_observer(self, enabled: bool):
        if hasattr(self.fake_q, "enable_observer"):
            if enabled: self.fake_q.enable_observer()
            else: self.fake_q.disable_observer()

    def enable_fake_quant(self, enabled: bool):
        if hasattr(self.fake_q, "enable_fake_quant"):
            if enabled: self.fake_q.enable_fake_quant()
            else: self.fake_q.disable_fake_quant()

    def forward(self, x):
        return self.fake_q(x)

class WeightFakeQuant(nn.Module):
    def __init__(self, per_channel: bool = True):
        super().__init__()
        self.observer = _make_weight_observer(per_channel)
        self.fake_q = torch.ao.quantization.FakeQuantize(observer=self.observer, dtype=torch.qint8,
                                                         quant_min=-128, quant_max=127,
                                                         qscheme=(torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric),
                                                         ch_axis=0)

    def enable_observer(self, enabled: bool):
        if hasattr(self.fake_q, "enable_observer"):
            if enabled: self.fake_q.enable_observer()
            else: self.fake_q.disable_observer()

    def enable_fake_quant(self, enabled: bool):
        if hasattr(self.fake_q, "enable_fake_quant"):
            if enabled: self.fake_q.enable_fake_quant()
            else: self.fake_q.disable_fake_quant()

    def forward(self, w):
        return self.fake_q(w)

# --- Wrappers for Conv/Linear ---

class QATConv2d(nn.Conv2d):
    """Conv2d with input/output/weight fakequant."""
    def __init__(self, m: nn.Conv2d, act_kind: str = "ema", w_per_channel: bool = True):
        super().__init__(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                         m.dilation, m.groups, m.bias is not None, m.padding_mode, device=m.weight.device, dtype=m.weight.dtype)
        # copy weights
        self.weight = nn.Parameter(m.weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())
        # fq
        self.in_fq  = ActFakeQuant(act_kind)
        self.w_fq   = WeightFakeQuant(w_per_channel)
        self.out_fq = ActFakeQuant(act_kind)

    def forward(self, x):
        x = self.in_fq(x)
        w = self.w_fq(self.weight)
        y = nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        y = self.out_fq(y)
        return y

class QATLinear(nn.Linear):
    """Linear with input/output/weight fakequant."""
    def __init__(self, m: nn.Linear, act_kind: str = "ema", w_per_channel: bool = True):
        super().__init__(m.in_features, m.out_features, m.bias is not None, device=m.weight.device, dtype=m.weight.dtype)
        self.weight = nn.Parameter(m.weight.detach().clone())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach().clone())
        self.in_fq  = ActFakeQuant(act_kind)
        # per-channel for linear is equivalent to per-row along out_features
        self.w_fq   = WeightFakeQuant(per_channel=w_per_channel)
        self.out_fq = ActFakeQuant(act_kind)

    def forward(self, x):
        x = self.in_fq(x)
        w = self.w_fq(self.weight)
        y = nn.functional.linear(x, w, self.bias)
        y = self.out_fq(y)
        return y

# --- Public helpers ---

def prepare_qat_inplace(model: nn.Module, act_observer: str = "ema", w_per_channel: bool = True) -> None:
    """Replace Conv2d/Linear by QAT wrapped versions (in-place)."""
    def _recurse(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(parent, name, QATConv2d(child, act_kind=act_observer, w_per_channel=w_per_channel))
            elif isinstance(child, nn.Linear):
                setattr(parent, name, QATLinear(child, act_kind=act_observer, w_per_channel=w_per_channel))
            else:
                _recurse(child)
    _recurse(model)
    print(f"[QAT] inserted FakeQuant around Conv/Linear (act_observer={act_observer}, w_per_channel={w_per_channel})")

def set_qat_observers_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable/disable observers in all fakequant submodules."""
    for m in model.modules():
        if hasattr(m, "enable_observer"):
            m.enable_observer(enabled)
        for attr in ("in_fq","out_fq","w_fq"):
            fq = getattr(m, attr, None)
            if fq is not None and hasattr(fq, "enable_observer"):
                fq.enable_observer(enabled)

def set_qat_fakequant_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable/disable fakequant (quantize/dequantize) effect."""
    for m in model.modules():
        if hasattr(m, "enable_fake_quant"):
            m.enable_fake_quant(enabled)
        for attr in ("in_fq","out_fq","w_fq"):
            fq = getattr(m, attr, None)
            if fq is not None and hasattr(fq, "enable_fake_quant"):
                fq.enable_fake_quant(enabled)
