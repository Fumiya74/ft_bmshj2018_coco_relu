import torch
import torch.nn as nn
from typing import Tuple

def _try_import_gdn():
    try:
        from compressai.layers import GDN
        return GDN
    except Exception:
        # 古いバージョン互換
        try:
            from compressai.layers.gdn import GDN
            return GDN
        except Exception:
            return None

def replace_gdn_with_relu(module: nn.Module) -> nn.Module:
    """CompressAI モデル内部の GDN/IGDN を ReLU に差し替える（inplace）。"""
    GDN = _try_import_gdn()

    def _replace(m: nn.Module, prefix=""):
        for name, child in list(m.named_children()):
            # 再帰的に探索
            _replace(child, prefix + name + ".")
            if GDN is not None and isinstance(child, GDN):
                # child.inverse で IGDN かどうかが分かるが、いずれも ReLU へ
                setattr(m, name, nn.ReLU(inplace=True))
    _replace(module)
    return module

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
