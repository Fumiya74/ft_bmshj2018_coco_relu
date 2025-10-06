
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU-friendly replacements for GDN/IGDN:
- GDNishLiteEnc: 1x1 -> ReLU6 -> DW3x3 -> ReLU6 -> 1x1 -> (ECA) [+ residual]
- GDNishLiteDec: 1x1 -> ReLU6 -> per-channel 1x1 -> HardSigmoid (scaled) -> mul -> kxk [+ residual]
All ops are ONNX/NPU-friendly and quantization-tolerant.
"""

from typing import Optional
import torch
import torch.nn as nn

# -------- ECA (Efficient Channel Attention) --------
class ECA(nn.Module):
    """GlobalAvgPool -> 1D conv (k=3..7) -> Sigmoid -> per-channel multiplication"""
    def __init__(self, C: int, k: int = 3):
        super().__init__()
        assert k % 2 == 1, "ECA kernel size k must be odd"
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,C,H,W]
        w = x.mean((2,3), keepdim=True)                 # [N,C,1,1]
        w = w.squeeze(-1).transpose(1,2)                # [N,1,C]
        w = self.conv1d(w).transpose(1,2).unsqueeze(-1) # [N,C,1,1]
        g = torch.sigmoid(w)
        return x * g

# -------- Scaled HardSigmoid (quantization-friendly gating) --------
class ScaledHardSigmoid(nn.Module):
    def __init__(self, g_min: float = 0.5, g_max: float = 2.0):
        super().__init__()
        self.g_min = float(g_min)
        self.g_span = float(g_max - g_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hard-sigmoid: clamp(x/6 + 0.5, 0, 1)
        h = torch.clamp(x * (1.0/6.0) + 0.5, 0.0, 1.0)
        return self.g_min + self.g_span * h

# -------- Encoder-side replacement (GDN alternative) --------
class GDNishLiteEnc(nn.Module):
    """
    1x1(C->tC) -> ReLU6 -> DW3x3(tC) -> ReLU6 -> 1x1(tC->C) -> [ECA] -> (+ residual)
    """
    def __init__(self, C: int, t: float = 2.0, kdw: int = 3, use_residual: bool = True, use_eca: bool = True):
        super().__init__()
        Ct = max(1, int(round(C * t)))
        self.pw1 = nn.Conv2d(C, Ct, 1, bias=True)
        self.act1 = nn.ReLU6(inplace=True)
        self.dw  = nn.Conv2d(Ct, Ct, kdw, padding=kdw//2, groups=Ct, bias=True)
        self.act2 = nn.ReLU6(inplace=True)
        self.pw2 = nn.Conv2d(Ct, C, 1, bias=True)
        self.eca = ECA(C, k=3) if use_eca else nn.Identity()
        self.use_res = use_residual

        self._init_identity_like(C, Ct)

    def _init_identity_like(self, C: int, Ct: int) -> None:
        # Initialize close to identity to keep pre-trained features stable
        nn.init.zeros_(self.pw1.weight); nn.init.zeros_(self.pw1.bias)
        with torch.no_grad():
            # first C output channels pass-through
            eye = torch.zeros_like(self.pw1.weight)    # [Ct,C,1,1]
            for i in range(min(C, Ct)):
                eye[i, i, 0, 0] = 1.0
            self.pw1.weight.copy_(eye)
        nn.init.zeros_(self.dw.bias); nn.init.zeros_(self.pw2.bias)
        nn.init.zeros_(self.dw.weight)
        # small center weight in DW to avoid breaking identity
        with torch.no_grad():
            k = self.dw.kernel_size[0]
            c = k // 2
            for ch in range(self.dw.out_channels):
                self.dw.weight[ch, 0, c, c] = 1e-3
        nn.init.zeros_(self.pw2.weight)
        with torch.no_grad():
            # take only first C rows from expanded rep to come back
            back = torch.zeros_like(self.pw2.weight)   # [C,Ct,1,1]
            for i in range(min(C, Ct)):
                back[i, i, 0, 0] = 1.0
            self.pw2.weight.copy_(back)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act2(self.dw(self.act1(self.pw1(x)))))
        y = self.eca(y)
        return x + y if self.use_res else y

# -------- Decoder-side replacement (IGDN alternative) --------
class GDNishLiteDec(nn.Module):
    """
    1x1 -> ReLU6 -> per-channel 1x1(groups=C) -> ScaledHardSigmoid -> mul -> kxk Conv [+ residual]
    Allows gain > 1 to mimic IGDN's "amplification".
    """
    def __init__(self, C: int, k: int = 3, g_min: float = 0.5, g_max: float = 2.0, use_residual: bool = True):
        super().__init__()
        self.mix   = nn.Conv2d(C, C, 1, bias=True)
        self.act   = nn.ReLU6(inplace=True)
        self.g_lin = nn.Conv2d(C, C, 1, groups=C, bias=True)  # per-channel linear
        self.g_act = ScaledHardSigmoid(g_min, g_max)
        self.kconv = nn.Conv2d(C, C, k, padding=k//2, bias=True)
        self.use_res = use_residual
        self._init_safe()

    def _init_safe(self):
        nn.init.zeros_(self.mix.bias); nn.init.zeros_(self.g_lin.bias); nn.init.zeros_(self.kconv.bias)
        # small weights to start near identity after residual
        nn.init.kaiming_uniform_(self.mix.weight, a=1.0)
        nn.init.zeros_(self.g_lin.weight)
        nn.init.zeros_(self.kconv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.mix(x))
        g = self.g_act(self.g_lin(z))   # [N,C,1,1] effectively
        y = self.kconv(z * g)
        return x + y if self.use_res else y
