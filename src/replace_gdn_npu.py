
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to replace CompressAI GDN/IGDN modules with NPU-friendly blocks.
"""
from typing import Tuple
import torch.nn as nn

from .npu_blocks import GDNishLiteEnc, GDNishLiteDec

def _is_gdn_module(m: nn.Module) -> Tuple[bool, bool]:
    """
    Returns (is_gdn, is_inverse). When unknown, infer from class name.
    """
    try:
        from compressai.layers import GDN
        if isinstance(m, GDN):
            return True, bool(getattr(m, "inverse", False))
    except Exception:
        pass
    name = m.__class__.__name__.lower()
    is_gdn = ("gdn" in name) or ("generalizeddivisive" in name)
    is_inv = ("inverse" in name) or ("igdn" in name)
    return is_gdn, is_inv

def _replace_in(module: nn.Module, mode: str, enc_kwargs: dict, dec_kwargs: dict) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        is_gdn, is_inv = _is_gdn_module(child)
        if is_gdn:
            # determine channels
            C = None
            # many GDN keep "channels" or "num_features"
            for attr in ("channels", "num_features", "ch", "C", "c"):
                if hasattr(child, attr):
                    C = int(getattr(child, attr))
                    break
            if C is None:
                # try conv weight shape in neighbors is complex; fallback to guess via bias/weight if present.
                # As a safe default, require caller to pass correct mode in models with standard CompressAI: g_a/g_s path.
                raise ValueError("Cannot infer channel count for GDN module; please pass standard CompressAI modules.")
            if (mode in ("encoder", "all")) and not is_inv:
                repl = GDNishLiteEnc(C, **enc_kwargs)
            elif (mode in ("decoder", "all")) and is_inv:
                repl = GDNishLiteDec(C, **dec_kwargs)
            else:
                # leave untouched
                _recurse(child)  # still search deeper
                continue
            setattr(module, name, repl)
            replaced += 1
        else:
            replaced += _replace_in(child, mode, enc_kwargs, dec_kwargs)
    return replaced

def _recurse(module: nn.Module):
    for _, child in list(module.named_children()):
        _recurse(child)

def replace_gdn_with_npu(
    model: nn.Module,
    mode: str = "all",
    enc_t: float = 2.0,
    enc_kdw: int = 3,
    enc_use_eca: bool = True,
    enc_use_res: bool = True,
    dec_k: int = 3,
    dec_gmin: float = 0.5,
    dec_gmax: float = 2.0,
    dec_use_res: bool = True,
) -> nn.Module:
    """
    Replace GDN/IGDN with GDNishLiteEnc/Dec.
    mode: 'encoder' | 'decoder' | 'all'
    """
    mode = mode.lower()
    if mode not in {"encoder", "decoder", "all"}:
        raise ValueError("mode must be 'encoder', 'decoder', or 'all'")

    enc_kwargs = dict(t=enc_t, kdw=enc_kdw, use_residual=enc_use_res, use_eca=enc_use_eca)
    dec_kwargs = dict(k=dec_k, g_min=dec_gmin, g_max=dec_gmax, use_residual=dec_use_res)

    # Prefer replacing within g_a / g_s when present (CompressAI standard modules)
    replaced = 0
    has_ga = hasattr(model, "g_a") and isinstance(getattr(model, "g_a"), nn.Module)
    has_gs = hasattr(model, "g_s") and isinstance(getattr(model, "g_s"), nn.Module)

    if mode in ("encoder", "all"):
        if has_ga:
            replaced += _replace_in(model.g_a, "encoder", enc_kwargs, dec_kwargs)
        else:
            replaced += _replace_in(model, "encoder", enc_kwargs, dec_kwargs)
    if mode in ("decoder", "all"):
        if has_gs:
            replaced += _replace_in(model.g_s, "decoder", enc_kwargs, dec_kwargs)
        else:
            replaced += _replace_in(model, "decoder", enc_kwargs, dec_kwargs)

    print(f"[replace_gdn_with_npu] Replaced {replaced} module(s) (mode='{mode}') "
          f"enc_t={enc_t}, enc_kdw={enc_kdw}, enc_eca={enc_use_eca}, dec_k={dec_k}, g=[{dec_gmin},{dec_gmax}]")
    return model
