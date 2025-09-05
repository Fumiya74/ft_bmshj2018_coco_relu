# src/model_utils.py の中に置く想定
from typing import Tuple, Optional, Sequence
import torch
import torch.nn as nn

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
