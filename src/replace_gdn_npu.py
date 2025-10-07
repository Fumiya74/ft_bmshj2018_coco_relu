
import torch
import torch.nn as nn
from typing import Tuple

try:
    from compressai.layers import GDN
except Exception:
    GDN = None

# Import the proposed NPU-friendly blocks (must exist in your repo)
from src.npu_blocks import GDNishLiteEnc, GDNishLiteDec


def _is_gdn(m: nn.Module) -> bool:
    if GDN is not None and isinstance(m, GDN):
        return True
    name = m.__class__.__name__.lower()
    return "gdn" in name or "generalizeddivisivenorm" in name


def _infer_channels_from_gdn(gdn: nn.Module) -> int | None:
    """
    Try multiple strategies to infer the channel count from a GDN-like module.
    Works with compressai.layers.GDN and lookalikes.
    """
    # official compressai GDN attribute
    ch = getattr(gdn, "channels", None)
    if isinstance(ch, int) and ch > 0:
        return ch

    # beta/gamma parameters
    beta = getattr(gdn, "beta", None)
    if isinstance(beta, torch.nn.Parameter) and beta.ndim == 1:
        return int(beta.numel())

    gamma = getattr(gdn, "gamma", None)
    if isinstance(gamma, torch.nn.Parameter) and gamma.ndim >= 1:
        return int(gamma.shape[0])

    return None


def _build_replacement(gdn: nn.Module, side_hint: str,
                       enc_kwargs: dict, dec_kwargs: dict) -> nn.Module:
    inverse = bool(getattr(gdn, "inverse", False))
    channels = _infer_channels_from_gdn(gdn)
    if channels is None:
        raise ValueError("Cannot infer channel count for GDN module; please pass standard CompressAI modules.")

    if inverse or side_hint == "decoder":
        return GDNishLiteDec(channels=channels, **dec_kwargs)
    else:
        return GDNishLiteEnc(channels=channels, **enc_kwargs)


def _replace_in(module: nn.Module, side_hint: str,
                enc_kwargs: dict, dec_kwargs: dict) -> int:
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
                         enc_t: float = 2.0,
                         enc_kdw: int = 3,
                         enc_eca: bool = True,
                         enc_residual: bool = True,
                         enc_act: str = "relu6",
                         enc_bn: bool = False,
                         dec_k: int = 3,
                         dec_gmin: float = 0.5,
                         dec_gmax: float = 2.0,
                         dec_residual: bool = True,
                         dec_act: str = "relu6",
                         dec_bn: bool = False,
                         verbose: bool = True) -> nn.Module:
    mode = mode.lower()
    assert mode in {"all", "encoder", "decoder"}

    enc_kwargs = dict(t=enc_t, k_dw=enc_kdw, use_eca=enc_eca, residual=enc_residual,
                      act_type=enc_act, bn=enc_bn)
    dec_kwargs = dict(k=dec_k, g_min=dec_gmin, g_max=dec_gmax, residual=dec_residual,
                      act_type=dec_act, use_bn=dec_bn)

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
