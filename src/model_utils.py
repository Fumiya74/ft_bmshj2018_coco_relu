import torch
import torch.nn as nn
from typing import Tuple
from typing import Iterable

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
    module 直下の submodule を走査し、GDNを ReLU(inplace=True) に置換。
    戻り値: 置換件数
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if _is_gdn(child):
            setattr(module, name, nn.ReLU(inplace=True))
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
        # g_a / g_s が無い特殊モデルの場合：
        # - 'all' / 'decoder+encoder' はモデル全体を対象
        # - 'encoder' / 'decoder' は部位特定ができないため全体置換かスキップを選ぶ
        #   → 実運用では全体置換のほうが分かりやすいので全体置換にしています。
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

def forward_reconstruction(model, x):
    """CompressAI モデルの出力から再構成画像 x_hat を取り出す。"""
    out = model(x)
    if isinstance(out, dict):
        x_hat = out.get("x_hat", None)
    else:
        x_hat = getattr(out, "x_hat", None)
    if x_hat is None:
        # 互換性フォールバック
        if hasattr(model, "g_s") and hasattr(model, "g_a"):
            # Entropy bottleneck等を無視して解析/合成のみ（学習外用途）
            y = model.g_a(x)
            x_hat = model.g_s(y)
        else:
            raise RuntimeError("Cannot extract x_hat from model output.")
    return x_hat
