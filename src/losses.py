import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def recon_loss(x_hat, x, alpha_l1=0.4):
    l1 = (x_hat - x).abs().mean()
    msssim = ms_ssim(x_hat.clamp(0,1), x.clamp(0,1), data_range=1.0, size_average=True)
    loss = alpha_l1 * l1 + (1.0 - alpha_l1) * (1.0 - msssim)
    return loss, {"l1": l1.detach(), "ms_ssim": msssim.detach()}

def psnr(x_hat, x, eps=1e-8):
    mse = torch.mean((x_hat - x)**2)
    return -10.0 * torch.log10(mse + eps)
