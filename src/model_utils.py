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
) -> torch.Tensor | Tuple[torch.Tensor, object]:
    """
    モデルの forward から再構成 x_hat を安全に取り出すユーティリティ。

    - CompressAI の bmshj2018_factorized は dict を返し、その中の 'x_hat' が再構成画像。
    - 学習・推論の両方で使用可能（学習時に勾配を切らない）。
    - clamp=False が既定（学習の勾配を阻害しない）。保存/メトリクス時は外側で clamp するか、
      clamp=True を指定してください。

    Args:
        model: nn.Module
        x: 入力画像 (B, C, H, W), 0..1 期待
        clamp: 0..1 に丸めたい場合 True（既定 False）
        clamp_min, clamp_max: clamp 範囲
        key_candidates: dict 出力時に x_hat 候補として探索するキー名の優先順
        return_all: True のとき (x_hat, raw_out) を返す

    Returns:
        x_hat もしくは (x_hat, raw_out)
    """
    out = model(x)  # 勾配を保持
    x_hat = _pick_x_hat(out, key_candidates)

    if clamp:
        x_hat = x_hat.clamp(clamp_min, clamp_max)

    # 早期に致命的な値を検出（デバッグ向け、学習の邪魔はしない）
    # コメントアウトしたい場合は下2行を消してください
    # if not _is_tensor_safe(x_hat):
    #     raise RuntimeError("forward_reconstruction: x_hat contains NaN/Inf")

    return (x_hat, out) if return_all else x_hat

"""
def forward_reconstruction(model, x):
    #CompressAI モデルの出力から再構成画像 x_hat を取り出す。
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
"""
