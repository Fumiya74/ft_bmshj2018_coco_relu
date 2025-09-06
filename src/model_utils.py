
import torch
import torch.nn as nn
from typing import Tuple, Sequence

def _is_gdn(m: nn.Module) -> bool:
    # compressai.layers.GDN を優先しつつ、名前ベースでも判定（独自GDN対策）
    try:
        from compressai.layers import GDN
        if isinstance(m, GDN):
            return True
    except Exception:
        pass
    name = m.__class__.__name__.lower()
    return "gdn" in name or "generalizeddivisivenorm" in name

def _replace_gdn_in_module(module: nn.Module) -> int:
    """
    module 直下の submodule を走査し、GDNを ReLU(inplace=False) に置換。
    戻り値: 置換件数
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if _is_gdn(child):
            # ★ インプレースは禁止（勾配保存が壊れて encoder に勾配が戻らない/NAN の一因になる）
            setattr(module, name, nn.ReLU(inplace=False))
            replaced += 1
        else:
            replaced += _replace_gdn_in_module(child)
    return replaced

def replace_gdn_with_relu(model: nn.Module, mode: str = "decoder") -> nn.Module:
    """
    GDN -> ReLU 置換を、学習対象パートだけに限定して実行。
    mode in {'decoder', 'encoder', 'decoder+encoder', 'all'}
    """
    mode = mode.lower()
    replaced_total = 0

    # モデルが g_a/g_s を持つ（CompressAI系の典型）場合は部位限定で置換
    has_ga = hasattr(model, "g_a") and isinstance(getattr(model, "g_a"), nn.Module)
    has_gs = hasattr(model, "g_s") and isinstance(getattr(model, "g_s"), nn.Module)

    if has_ga or has_gs:
        if mode in ("all", "decoder+encoder", "encoder"):
            if has_ga:
                replaced_total += _replace_gdn_in_module(model.g_a)
        if mode in ("all", "decoder+encoder", "decoder"):
            if has_gs:
                replaced_total += _replace_gdn_in_module(model.g_s)
    else:
        # g_a / g_s が無い特殊モデルの場合は全体置換
        if mode in ("all", "decoder+encoder", "encoder", "decoder"):
            replaced_total += _replace_gdn_in_module(model)

    print(f"[replace_gdn_with_relu] Replaced {replaced_total} GDN(s) -> ReLU (mode='{mode}')")
    return model

def set_trainable_parts(model: nn.Module, mode: str = "decoder"):
    """mode in {'decoder', 'encoder', 'decoder+encoder', 'all'}"""
    mode = mode.lower()
    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    # CompressAI の一般的な命名: g_a (encoder), g_s (decoder)
    if mode == "decoder" and hasattr(model, "g_s"):
        for p in model.g_s.parameters():
            p.requires_grad = True

    elif mode == "encoder" and hasattr(model, "g_a"):
        for p in model.g_a.parameters():
            p.requires_grad = True

    elif mode == "decoder+encoder":
        if hasattr(model, "g_s"):
            for p in model.g_s.parameters():
                p.requires_grad = True
        if hasattr(model, "g_a"):
            for p in model.g_a.parameters():
                p.requires_grad = True

@torch.no_grad()
def _is_tensor_safe(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all()

def _pick_x_hat(out, key_candidates: Sequence[str]) -> torch.Tensor:
    """
    CompressAI 系 forward の戻り値 out から再構成画像 x_hat を取り出す。
    - dict: 優先キー順に探索して取得
    - Tensor: そのまま返す
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in key_candidates:
            if k in out:
                return out[k]
    raise KeyError(
        f"forward_reconstruction: could not find x_hat in output. "
        f"Tried keys={list(key_candidates)}; got type={type(out)} with keys={list(out.keys()) if isinstance(out, dict) else None}"
    )

def forward_reconstruction(
    model: nn.Module,
    x: torch.Tensor,
    *,
    clamp: bool = False,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    key_candidates: Sequence[str] = ("x_hat", "x", "recon", "x_dec", "x_out"),
    return_all: bool = False,
):
    """
    モデルの forward から再構成 x_hat を安全に取り出すユーティリティ。

    - CompressAI の bmshj2018_factorized は dict を返し、その中の 'x_hat' が再構成画像。
    - 学習・推論の両方で使用可能（学習時に勾配を切らない）。
    - clamp=False が既定（学習の勾配を阻害しない）。保存/メトリクス時は外側で clamp するか、
      clamp=True を指定してください。
    """
    out = model(x)  # 勾配を保持
    x_hat = _pick_x_hat(out, key_candidates)

    if clamp:
        x_hat = x_hat.clamp(clamp_min, clamp_max)

    return (x_hat, out) if return_all else x_hat
