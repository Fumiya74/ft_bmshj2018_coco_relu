#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantization helpers aligned with Renesas DRP-AI TVM QAT reference.

Key characteristics:
  - Activation fake-quant: MovingAverageMinMaxObserver, uint8 [0,255], per-tensor affine, reduce_range=True
  - Weight fake-quant: MovingAveragePerChannelMinMaxObserver, qint8 [-128,127], per-channel symmetric (ch_axis=0)
  - Uses PyTorch's prepare_qat/enable_observer/enable_fake_quant controls instead of custom STE wrappers.
  - Scoped preparation (encoder / decoder / all) while automatically excluding EntropyBottleneck & hyperpriors.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

try:
    import torch.ao.quantization as tq  # PyTorch >= 1.13
except ModuleNotFoundError:  # pragma: no cover
    import torch.quantization as tq  # type: ignore[no-redef]

_observer = getattr(tq, "observer", None)
_fake_quant_mod = getattr(tq, "fake_quantize", None)

MovingAverageMinMaxObserver = getattr(
    _observer, "MovingAverageMinMaxObserver", getattr(tq, "MovingAverageMinMaxObserver", None)
)
MovingAveragePerChannelMinMaxObserver = getattr(
    _observer, "MovingAveragePerChannelMinMaxObserver", getattr(tq, "MovingAveragePerChannelMinMaxObserver", None)
)
FakeQuantizeCls = getattr(_fake_quant_mod, "FakeQuantize", getattr(tq, "FakeQuantize", None))

if MovingAverageMinMaxObserver is None or MovingAveragePerChannelMinMaxObserver is None or FakeQuantizeCls is None:
    raise ImportError("Required quantization observers/fake-quant classes are unavailable in this PyTorch build.")

_QAT_TAG = "_drp_qat_prepared"
_TARGETS_ATTR = "_drp_qat_targets"


def _is_hyper_or_entropy(module: nn.Module, name: str) -> bool:
    lname = name.lower()
    if "entropy" in lname or "gaussianconditional" in lname:
        return True
    if any(k in lname for k in ("h_a", "h_s", "hyper", "hyperprior")):
        return True
    return False


def is_qatified(module: nn.Module) -> bool:
    return bool(getattr(module, _QAT_TAG, False))


def _default_drpai_qconfig() -> tq.QConfig:
    act_fake = FakeQuantizeCls.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True,
    )
    weight_fake = FakeQuantizeCls.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        ch_axis=0,
    )
    return tq.QConfig(activation=act_fake, weight=weight_fake)


def _collect_scope_targets(model: nn.Module, scope: str) -> Sequence[nn.Module]:
    scope = scope.lower()
    if scope == "all":
        return (model,)
    if scope == "encoder":
        return tuple(filter(None, [getattr(model, "g_a", None)]))
    if scope == "decoder":
        return tuple(filter(None, [getattr(model, "g_s", None)]))
    raise ValueError(f"Unknown scope '{scope}'. Expected one of ['encoder','decoder','all'].")


def _assign_qconfig_recursive(module: nn.Module, qconfig: tq.QConfig, exclude_entropy: bool) -> None:
    for child in module.modules():
        child.qconfig = qconfig  # type: ignore[attr-defined]
    if not exclude_entropy:
        return
    for name, child in module.named_modules():
        if _is_hyper_or_entropy(child, name):
            child.qconfig = None  # type: ignore[attr-defined]

def _limit_qconfig_modules(module: nn.Module, module_limit: int, skip_first_conv: bool) -> None:
    unlimited = module_limit <= 0
    first_skipped = False
    remaining = None
    if skip_first_conv:
        remaining = None if unlimited else max(0, module_limit - 1)
    count = 0
    for _, child in module.named_modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            if getattr(child, "qconfig", None) is None:
                continue
            if skip_first_conv and not first_skipped:
                child.qconfig = None  # type: ignore[attr-defined]
                first_skipped = True
                continue
            if skip_first_conv:
                if remaining is None:
                    continue
                if remaining <= 0:
                    child.qconfig = None  # type: ignore[attr-defined]
                else:
                    remaining -= 1
                continue
            if unlimited:
                continue
            count += 1
            if count > module_limit:
                child.qconfig = None  # type: ignore[attr-defined]


def _prepare_module_for_qat(module: nn.Module, qconfig: tq.QConfig,
                            exclude_entropy: bool, module_limit: int, skip_first_conv: bool) -> None:
    _assign_qconfig_recursive(module, qconfig, exclude_entropy)
    _limit_qconfig_modules(module, module_limit, skip_first_conv)
    tq.prepare_qat(module, inplace=True)


def prepare_qat_inplace_scoped(
    model: nn.Module,
    scope: str = "encoder",
    *,
    calib_steps: int = 2000,
    freeze_after: int = 0,
    exclude_entropy: bool = True,
    module_limit: int = 0,
    range_margin: float = 0.0,
    skip_first_conv: bool = False,
    verbose: bool = False,
    **_: object,
) -> nn.Module:
    """
    Apply DRP-AI style QAT (torch.ao.quantization based) in-place on the selected scope.

    Args:
        model: CompressAI model instance.
        scope: "encoder" | "decoder" | "all"
        calib_steps: Steps before enabling fake-quant (observers stay enabled).
        freeze_after: After this step observers are disabled (and BN optionally frozen).
        exclude_entropy: Always skip EntropyBottleneck / hyper modules.
        module_limit: Limit the number of Conv/Linear layers prepared per scope (0 = no limit).
        range_margin: Expand observer ranges by this fraction once freeze mode activates.
        verbose: Print summary.
        **_: Compatibility kwargs ignored (encoder_attr, decoder_attr, etc.)
    """
    if is_qatified(model) and getattr(model, _TARGETS_ATTR, None) is not None:
        raise RuntimeError("Model already prepared for QAT. Avoid double preparation.")

    qconfig = _default_drpai_qconfig()
    targets = _collect_scope_targets(model, scope)
    if not targets:
        raise RuntimeError(f"Scope '{scope}' did not yield any modules to quantize.")

    for target in targets:
        _prepare_module_for_qat(target, qconfig, exclude_entropy, module_limit, skip_first_conv)

    setattr(model, _TARGETS_ATTR, targets)
    setattr(model, _QAT_TAG, True)

    if not hasattr(model, "_qat_global_step"):
        model.register_buffer("_qat_global_step", torch.tensor(0))
    else:
        model._qat_global_step.fill_(0)  # type: ignore[attr-defined]

    if not hasattr(model, "_qat_calib_steps"):
        model.register_buffer("_qat_calib_steps", torch.tensor(int(calib_steps)))
    else:
        model._qat_calib_steps.fill_(int(calib_steps))  # type: ignore[attr-defined]

    if not hasattr(model, "_qat_freeze_after"):
        model.register_buffer("_qat_freeze_after", torch.tensor(int(freeze_after)))
    else:
        model._qat_freeze_after.fill_(int(freeze_after))  # type: ignore[attr-defined]

    if not hasattr(model, "_qat_range_margin"):
        model.register_buffer("_qat_range_margin", torch.tensor(float(range_margin)))
    else:
        model._qat_range_margin.fill_(float(range_margin))  # type: ignore[attr-defined]
    if not hasattr(model, "_qat_margin_applied"):
        model.register_buffer("_qat_margin_applied", torch.tensor(False))
    else:
        model._qat_margin_applied.fill_(False)  # type: ignore[attr-defined]

    set_qat_mode(model, "calib")

    if verbose:
        names = ", ".join(type(target).__name__ for target in targets)
        print(
            f"[qat] DRP-AI prepare: scope={scope} modules={len(targets)} "
            f"calib_steps={calib_steps} freeze_after={freeze_after} exclude_entropy={exclude_entropy}"
        )
    return model


def _apply_to_fake_quant_modules(model: nn.Module, methods: Iterable[str]) -> None:
    for m in model.modules():
        for method in methods:
            fn = getattr(m, method, None)
            if callable(fn):
                fn()

@torch.no_grad()
def _apply_range_margin(model: nn.Module, margin: float) -> None:
    """
    Expand observer min/max by a fractional margin to soften hard clamps after calibration.
    """
    margin = float(max(0.0, margin))
    if margin == 0.0:
        return
    for module in model.modules():
        obs = getattr(module, "observer", None)
        if obs is None:
            continue
        min_attr = getattr(obs, "min_val", None)
        max_attr = getattr(obs, "max_val", None)
        if min_attr is not None and max_attr is not None:
            span = (max_attr - min_attr).abs()
            delta = span * margin + 1e-6
            obs.min_val.copy_(min_attr - delta)
            obs.max_val.copy_(max_attr + delta)
            continue
        max_abs = getattr(obs, "max_abs", None)
        if max_abs is not None:
            delta = max_abs.abs() * margin + 1e-6
            obs.max_abs.copy_(max_abs + delta)

def set_qat_mode(model: nn.Module, mode: str = "calib") -> None:
    """
    mode: "calib" (observers on, fake-quant off),
          "train" (observers + fake-quant on),
          "freeze" (observers off, fake-quant on, optional range margin applied)
    """
    if not is_qatified(model):
        return
    mode = mode.lower()
    if mode == "calib":
        _apply_to_fake_quant_modules(model, ("disable_fake_quant", "enable_observer"))
    elif mode == "train":
        _apply_to_fake_quant_modules(model, ("enable_fake_quant", "enable_observer"))
    elif mode == "freeze":
        _apply_to_fake_quant_modules(model, ("enable_fake_quant", "disable_observer"))
        freeze_bn_stats(model)
        margin = float(getattr(model, "_qat_range_margin", torch.tensor(0.0)).item())
        applied_buf = getattr(model, "_qat_margin_applied", None)
        already_applied = bool(applied_buf.item()) if isinstance(applied_buf, torch.Tensor) else False
        if margin > 0.0 and not already_applied:
            _apply_range_margin(model, margin)
            if isinstance(applied_buf, torch.Tensor):
                applied_buf.fill_(True)
            else:
                model.register_buffer("_qat_margin_applied", torch.tensor(True))
    else:
        raise ValueError("mode must be one of 'calib','train','freeze'")


def freeze_bn_stats(model: nn.Module) -> None:
    try:
        tq.freeze_bn_stats(model)
    except AttributeError:
        # Older torch.quantization API
        if hasattr(tq, "freeze_bn_stats"):
            tq.freeze_bn_stats(model)  # type: ignore[misc]


def _sync_qat_phase(model: nn.Module) -> None:
    step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
    calib_steps = int(getattr(model, "_qat_calib_steps", torch.tensor(0)).item())
    freeze_after = int(getattr(model, "_qat_freeze_after", torch.tensor(0)).item())

    if freeze_after > 0 and step >= freeze_after:
        set_qat_mode(model, "freeze")
    elif step < calib_steps:
        set_qat_mode(model, "calib")
    else:
        set_qat_mode(model, "train")


def set_qat_global_step(model: nn.Module, step: int) -> None:
    if not is_qatified(model):
        return
    if hasattr(model, "_qat_global_step"):
        model._qat_global_step.fill_(int(step))  # type: ignore[attr-defined]
    _sync_qat_phase(model)


def step_qat(model: nn.Module, inc: int = 1) -> None:
    if not is_qatified(model):
        return
    cur = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
    set_qat_global_step(model, cur + int(inc))


def apply_qat_phase_by_step(model: nn.Module, global_step: int) -> None:
    set_qat_global_step(model, int(global_step))


def step_qat_schedule(model: nn.Module, global_step: int) -> None:
    apply_qat_phase_by_step(model, global_step)


def print_qat_summary(model: nn.Module) -> None:
    if not is_qatified(model):
        print("QAT not prepared.")
        return
    step = int(getattr(model, "_qat_global_step", torch.tensor(0)).item())
    calib_steps = int(getattr(model, "_qat_calib_steps", torch.tensor(0)).item())
    freeze_after = int(getattr(model, "_qat_freeze_after", torch.tensor(0)).item())
    print(f"[QAT] step={step} calib_steps={calib_steps} freeze_after={freeze_after}")

    targets = getattr(model, _TARGETS_ATTR, ())
    if targets:
        names = ", ".join(type(t).__name__ for t in targets)
        print(f"[QAT] targets: {names}")

    fake_quant_modules = []
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizeCls):
            fake_quant_modules.append((name, module))

    print(f"[QAT] fake-quant modules: {len(fake_quant_modules)}")
    for name, fq in fake_quant_modules[:10]:
        qmin = getattr(fq, "quant_min", None)
        qmax = getattr(fq, "quant_max", None)
        dtype = getattr(fq, "dtype", None)
        print(f"  {name}: FakeQuantize(qmin={qmin}, qmax={qmax}, dtype={dtype})")
    if len(fake_quant_modules) > 10:
        print(f"  ... ({len(fake_quant_modules) - 10} more)")
