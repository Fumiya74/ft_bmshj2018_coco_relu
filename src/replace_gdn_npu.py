
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace CompressAI GDN/IGDN with NPU-friendly blocks defined in src.npu_blocks.py.

This version matches the user's GDNishLite signatures exactly:

- GDNishLiteEnc(C: int, t: float=2.0, kdw: int=3, use_residual: bool=True, use_eca: bool=True)
- GDNishLiteDec(C: int, k: int=3, g_min: float=0.5, g_max: float=2.0, use_residual: bool=True)

It also:
- infers channels from compressai.layers.GDN or lookalikes (channels/beta/gamma)
- accepts alias args and ignores unknown kwargs with a warning
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

try:
    from compressai.layers import GDN
except Exception:
    GDN = None

from src.npu_blocks import GDNishLiteEnc, GDNishLiteDec


def _is_gdn(m: nn.Module) -> bool:
    if GDN is not None and isinstance(m, GDN):
        return True
    name = m.__class__.__name__.lower()
    return "gdn" in name or "generalizeddivisivenorm" in name


def _infer_channels_from_gdn(gdn: nn.Module) -> Optional[int]:
    # compressai.GDN exposes channels
    ch = getattr(gdn, "channels", None)
    if isinstance(ch, int) and ch > 0:
        return ch
    # fallback: look for beta/gamma parameters
    beta = getattr(gdn, "beta", None)
    if isinstance(beta, torch.nn.Parameter) and beta.ndim == 1:
        return int(beta.numel())
    gamma = getattr(gdn, "gamma", None)
    if isinstance(gamma, torch.nn.Parameter) and gamma.ndim >= 1:
        return int(gamma.shape[0])
    return None


def _build_replacement(gdn: nn.Module, side_hint: str,
                       enc_kwargs: Dict[str, Any],
                       dec_kwargs: Dict[str, Any]) -> nn.Module:
    inverse = bool(getattr(gdn, "inverse", False))
    C = _infer_channels_from_gdn(gdn)
    if C is None:
        raise ValueError("Cannot infer channel count for GDN module; please pass standard CompressAI modules.")

    if inverse or side_hint == "decoder":
        # Map to user's Dec signature
        ctor = GDNishLiteDec
        ctor_kwargs = dict(C=C,
                           k=dec_kwargs.get("k", 3),
                           g_min=dec_kwargs.get("g_min", 0.5),
                           g_max=dec_kwargs.get("g_max", 2.0),
                           use_residual=dec_kwargs.get("use_residual", True))
    else:
        # Map to user's Enc signature
        ctor = GDNishLiteEnc
        ctor_kwargs = dict(C=C,
                           t=enc_kwargs.get("t", 2.0),
                           kdw=enc_kwargs.get("kdw", enc_kwargs.get("k_dw", 3)),
                           use_residual=enc_kwargs.get("use_residual", enc_kwargs.get("residual", True)),
                           use_eca=enc_kwargs.get("use_eca", enc_kwargs.get("eca", True)))
    return ctor(**ctor_kwargs)


def _replace_in(module: nn.Module, side_hint: str,
                enc_kwargs: Dict[str, Any],
                dec_kwargs: Dict[str, Any]) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if _is_gdn(child):
            new_block = _build_replacement(child, side_hint, enc_kwargs, dec_kwargs)
            setattr(module, name, new_block)
            replaced += 1
        else:
            replaced += _replace_in(child, side_hint, enc_kwargs, dec_kwargs)
    return replaced


def replace_gdn_with_npu(model: nn.Module,
                         mode: str = "all",
                         *,
                         # Encoder knobs (match or map to user's GDNishLiteEnc)
                         enc_t: float = 2.0,
                         enc_kdw: int = 3,
                         enc_use_eca: Optional[bool] = None,
                         enc_residual: Optional[bool] = None,
                         # Decoder knobs (match or map to user's GDNishLiteDec)
                         dec_k: int = 3,
                         dec_gmin: float = 0.5,
                         dec_gmax: float = 2.0,
                         dec_residual: Optional[bool] = None,
                         verbose: bool = True,
                         **extra_kwargs) -> nn.Module:
    """
    Replace GDN/IGDN with NPU-friendly GDNishLite blocks.
    Unknown kwargs are ignored with a single-line warning.
    """
    if extra_kwargs and verbose:
        print(f"[replace_gdn_with_npu] Ignoring extra kwargs: {list(extra_kwargs.keys())}")

    mode = mode.lower()
    assert mode in {"all", "encoder", "decoder"}

    enc_kwargs = dict(t=enc_t, kdw=enc_kdw)
    if enc_use_eca is not None:
        enc_kwargs["use_eca"] = bool(enc_use_eca)
    if enc_residual is not None:
        enc_kwargs["use_residual"] = bool(enc_residual)

    dec_kwargs = dict(k=dec_k, g_min=dec_gmin, g_max=dec_gmax)
    if dec_residual is not None:
        dec_kwargs["use_residual"] = bool(dec_residual)

    total = 0
    has_ga = hasattr(model, "g_a") and isinstance(getattr(model, "g_a"), nn.Module)
    has_gs = hasattr(model, "g_s") and isinstance(getattr(model, "g_s"), nn.Module)

    if mode in {"all", "encoder"}:
        if has_ga:
            total += _replace_in(model.g_a, "encoder", enc_kwargs, dec_kwargs)
        else:
            total += _replace_in(model, "encoder", enc_kwargs, dec_kwargs)

    if mode in {"all", "decoder"}:
        if has_gs:
            total += _replace_in(model.g_s, "decoder", enc_kwargs, dec_kwargs)
        else:
            total += _replace_in(model, "decoder", enc_kwargs, dec_kwargs)

    if verbose:
        print(f"[replace_gdn_with_npu] Replaced {total} GDN/IGDN layer(s) with GDNishLite blocks (mode={mode}).")
    return model