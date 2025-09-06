import torch
import torch.nn as nn
from typing import Sequence

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
            # インプレースは禁止（勾配保存を壊す可能性）
            setattr(module, name, nn.ReLU(inplace=False))
            replaced += 1
        else:
            replaced += _replace_gdn_in_module(child)
    return replaced

def replace_gdn_with_relu(model: nn.Module, mode: str = "encoder") -> nn.Module:
    """
    GDN -> ReLU 置換。
    mode in {'encoder', 'decoder', 'all'}
    """
    mode = mode.lower()
    if mode not in {"encoder", "decoder", "all"}:
        raise ValueError(f"replace_gdn_with_relu: mode must be one of 'encoder','decoder','all', got {mode}")

    replaced_total = 0
    has_ga = hasattr(model, "g_a") and isinstance(getattr(model, "g_a"), nn.Module)
    has_gs = hasattr(model, "g_s") and isinstance(getattr(model, "g_s"), nn.Module)

    if mode in ("encoder", "all"):
        if has_ga:
            replaced_total += _replace_gdn_in_module(model.g_a)
        else:
            replaced_total += _replace_gdn_in_module(model)
    if mode in ("decoder", "all"):
        if has_gs:
            replaced_total += _replace_gdn_in_module(model.g_s)
        else:
            replaced_total += _replace_gdn_in_module(model)

    print(f"[replace_gdn_with_relu] Replaced {replaced_total} GDN(s) -> ReLU (mode='{mode}')")
    return model

def set_trainable_parts(model: nn.Module, replaced_block: str = "encoder", train_scope: str = "replaced"):
    """
    再学習範囲の設定。
    replaced_block in {'encoder','decoder','all'}  … GDN置換を行ったブロックの指定
    train_scope    in {'replaced','replaced+hyper','all'} … 再学習する範囲
      - 'replaced'           : 置換したブロックのみ（encoder→g_a / decoder→g_s / all→g_a+g_s）
      - 'replaced+hyper'     : 上に加えて hyperprior 系（h_a, h_s, entropy_bottleneck）も更新
      - 'all'                : 全層を更新
    """
    replaced_block = replaced_block.lower()
    train_scope = train_scope.lower()
    if replaced_block not in {"encoder","decoder","all"}:
        raise ValueError("set_trainable_parts: replaced_block must be 'encoder','decoder','all'")
    if train_scope not in {"replaced","replaced+hyper","all"}:
        raise ValueError("set_trainable_parts: train_scope must be 'replaced','replaced+hyper','all'")

    # まず全て停止
    for p in model.parameters():
        p.requires_grad = False

    if train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    # ユーティリティ：存在する場合に requires_grad=True を立てる
    def _enable(module: nn.Module | None):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    # 置換ブロック
    if replaced_block in {"encoder","all"} and hasattr(model, "g_a"):
        _enable(model.g_a)
    if replaced_block in {"decoder","all"} and hasattr(model, "g_s"):
        _enable(model.g_s)

    if train_scope == "replaced+hyper":
        # hyperprior 系（存在すれば）
        _enable(getattr(model, "h_a", None))
        _enable(getattr(model, "h_s", None))
        _enable(getattr(model, "entropy_bottleneck", None))  # Module だが .parameters() を持つ

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
    - 学習時に勾配を切らない（clamp=False が既定）。
    """
    out = model(x)  # 勾配を保持
    x_hat = _pick_x_hat(out, key_candidates)

    if clamp:
        x_hat = x_hat.clamp(clamp_min, clamp_max)

    return (x_hat, out) if return_all else x_hat
