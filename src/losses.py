
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from typing import Dict, Iterable

def recon_loss(x_hat, x, alpha_l1: float = 0.4):
    """
    学習安定化のため、損失計算は常に FP32 で実施し、MS-SSIM は 0..1 範囲で評価。
    """
    x_hat_f = x_hat.float().clamp(0, 1)
    x_f = x.float().clamp(0, 1)

    l1 = (x_hat_f - x_f).abs().mean()
    msssim = ms_ssim(x_hat_f, x_f, data_range=1.0, size_average=True)
    loss = alpha_l1 * l1 + (1.0 - alpha_l1) * (1.0 - msssim)
    return loss, {"l1": l1.detach(), "ms_ssim": msssim.detach()}

def _bpp_from_likelihoods(likelihoods: Dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    """
    CompressAI の likelihoods(dict of tensors) から bpp を計算。
    bpp = total_bits / (N*H*W), total_bits = sum(-log2 p)
    """
    total_bits = 0.0
    for k, lik in likelihoods.items():
        # lik: (N, C, H, W) など
        total_bits = total_bits + (-torch.log2(lik + 1e-9)).sum()
    return total_bits / float(num_pixels)

def rd_loss(x_hat, x, likelihoods: Dict[str, torch.Tensor], *, alpha_l1: float = 0.4, lambda_bpp: float = 0.01):
    """
    RD 最適化: loss = D + lambda * R
      - D: recon_loss と同じ定義（L1 と 1-MS-SSIM の混合）
      - R: likelihoods から計算した bpp
    """
    N, _, H, W = x.shape
    num_pixels = N * H * W

    recon, logs = recon_loss(x_hat, x, alpha_l1=alpha_l1)
    bpp = _bpp_from_likelihoods(likelihoods, num_pixels)

    loss = recon + lambda_bpp * bpp
    logs = dict(logs)
    logs["bpp"] = bpp.detach()
    return loss, logs

def psnr(x_hat, x, eps=1e-8):
    x_hat = x_hat.float()
    x = x.float()
    mse = torch.mean((x_hat - x) ** 2)
    return -10.0 * torch.log10(mse + eps)