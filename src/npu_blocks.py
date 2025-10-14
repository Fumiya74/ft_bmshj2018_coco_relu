"""
DRP-AI–safe GDNishLite blocks (full file)

This version avoids Conv1d and tanh so it translates cleanly to DRP‑AI via ONNX.
All ops are Conv2d (1×1 / depthwise / 3×3), Add, Mul, GlobalAveragePool2d,
Clamp (for ReLU6 / Hardtanh / HardSigmoid), plus simple arithmetic.

Key changes vs. previous GDNishLite:
- ECA → SE‑Lite (GAP + 1×1 Conv2d + HardSigmoid implemented as clamp/scale)
- Decoder gain: tanh removed; uses hardtanh centered at 1 (1 + clamp(·, −a, a))
- Careful, near‑identity initializations to prevent early overflow
- No in‑place activations (better for ONNX export)

Tested export target: ONNX opset 13.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "SELite",
    "GDNishLiteEnc",
    "GDNishLiteDec",
]


class SELite(nn.Module):
    """Squeeze-and-Excitation (lite, DRP-AI friendly).

    SE(x) = x * hsigmoid(Conv2d(GAP(x)))
    where hsigmoid(u) := clamp(u + 3, 0, 6) / 6  (built from Add + Clamp + Div)

    Parameters
    ----------
    channels: int
        Number of channels in/out.
    reduce: int
        Reduction ratio for the hidden bottleneck (set to 1 to keep 1×1 only).
        Using reduce=1 keeps a single Conv2d(C→C).
    """

    def __init__(self, channels: int, reduce: int = 1) -> None:
        super().__init__()
        assert reduce >= 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        c_mid = max(1, channels // reduce)
        # two-layer MLP with 1×1 convs is often used; to stay simple and DRP‑AI friendly,
        # we keep a single 1×1 conv. If you want two layers, add another Conv2d+ReLU.
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc.bias)
        # small weight init to avoid early over-scaling
        nn.init.zeros_(self.fc.weight)

    @staticmethod
    def _hsigmoid(u: torch.Tensor) -> torch.Tensor:
        # HardSigmoid(u) = clamp(u+3, 0, 6)/6
        return torch.clamp(u + 3.0, 0.0, 6.0) / 6.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)               # [N, C, 1, 1]
        y = self.fc(y)                # [N, C, 1, 1]
        y = self._hsigmoid(y)         # [N, C, 1, 1]
        return x * y                  # scale per channel


class GDNishLiteEnc(nn.Module):
    """Encoder-side DRP-AI–friendly surrogate for GDN.

    Structure: 1×1 (C→Ct) → ReLU6 → DWConv(Ct groups) → ReLU6 → 1×1 (Ct→C)
               → optional SE‑Lite → Clamp([−enc_out_clip, +enc_out_clip])
               → residual add with learnable scale α (init 0.1)

    Notes
    -----
    * All activations are implemented with Clamp/linear ops (no GELU/Swish).
    * Depthwise conv starts almost zero (center tap 1e−4) to keep numerics tame.
    * 1×1 convs are identity when shapes match; otherwise start at zeros.
    """

    def __init__(
        self,
        C: int,
        Ct: Optional[int] = None,
        *,
        # compatibility aliases / extras from replacement utility
        t: Optional[int] = None,                 # alias of Ct
        use_residual: bool = True,
        use_se: bool = False,
        use_eca: Optional[bool] = None,          # alias (if provided)
        dw_kernel_size: int = 3,
        enc_out_clip: float = 6.0,
        alpha_init: float = 0.1,
        se_reduce: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dw_kernel_size % 2 == 1, "dw_kernel_size must be odd"
        # resolve aliases
        if Ct is None:
            Ct = t if t is not None else C
        if use_eca is not None:
            use_se = bool(use_eca)  # map ECA flag to SE-lite on/off

        # Layers
        self.pw1 = nn.Conv2d(C, Ct, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(Ct, Ct, kernel_size=dw_kernel_size,
                            padding=dw_kernel_size // 2, groups=Ct, bias=True)
        self.pw2 = nn.Conv2d(Ct, C, kernel_size=1, bias=True)
        self.se = SELite(C, reduce=se_reduce) if use_se else nn.Identity()

        # Residual control
        self.use_res = use_residual
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Output clamp for EB input safety
        self.enc_out_clip = float(enc_out_clip)

        # Initialization
        self._init_identity_like(C=C, Ct=Ct, dw_kernel_size=dw_kernel_size)

    def _init_identity_like(self, C: int, Ct: int, dw_kernel_size: int) -> None:
        nn.init.zeros_(self.pw1.bias)
        nn.init.zeros_(self.pw2.bias)
        nn.init.zeros_(self.dw.bias)

        if Ct == C:
            eye = torch.zeros((Ct, C, 1, 1))
            for i in range(C):
                eye[i, i, 0, 0] = 1.0
            with torch.no_grad():
                self.pw1.weight.copy_(eye)
                self.pw2.weight.copy_(eye)
        else:
            nn.init.zeros_(self.pw1.weight)
            nn.init.zeros_(self.pw2.weight)

        with torch.no_grad():
            self.dw.weight.zero_()
            c = dw_kernel_size // 2
            for ch in range(self.dw.weight.shape[0]):
                self.dw.weight[ch, 0, c, c] = 1e-4

    @staticmethod
    def _relu6(u: torch.Tensor) -> torch.Tensor:
        # ReLU6 via clamp(·, 0, 6)
        return torch.clamp(u, 0.0, 6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(x)
        y = self._relu6(y)
        y = self.dw(y)
        y = self._relu6(y)
        y = self.pw2(y)
        y = self.se(y)
        y = torch.clamp(y, -self.enc_out_clip, self.enc_out_clip)
        if self.use_res:
            return x + self.alpha * y
        return y


class GDNishLiteDec(nn.Module):
    """Decoder-side DRP-AI–friendly surrogate for IGDN.

    Output = (x + kconv(x)) * g
    with g = clamp(1 + clamp(g_lin(z), −a, +a), gmin, gmax),  z = mix(x)

    Parameters
    ----------
    C: int
        Channels in/out.
    use_se: bool
        Whether to enable SE‑Lite scaling after the residual branch.
    gmin, gmax: float
        Final gain bounds (0.5–1.3 recommended). Centered around 1 at init.
    a: float
        Inner hardtanh half‑width: g_pre = 1 + clamp(g_lin(z), −a, +a)
    kkernel: int
        Kernel size for the optional refinement conv. Defaults to 3.
    use_refine: bool
        If True, include `kconv`; otherwise Identity.
    """

    def __init__(
        self,
        C: int,
        *,
        use_se: bool = False,
        gmin: float = 0.5,
        gmax: float = 1.3,
        a: float = 0.3,
        kkernel: int = 3,
        use_refine: bool = True,
        se_reduce: int = 1,
        # compatibility aliases / extras from replacement utility
        dec_gmax: Optional[float] = None,    # alias of gmax
        use_eca: Optional[bool] = None,      # alias of use_se
        **kwargs,
    ) -> None:
        super().__init__()
        if dec_gmax is not None:
            gmax = float(dec_gmax)
        if use_eca is not None:
            use_se = bool(use_eca)
        assert gmin > 0 and gmax > gmin, "Invalid gain bounds"
        assert kkernel % 2 == 1, "kkernel must be odd"
        assert a > 0.0

        self.gmin = float(gmin)
        self.gmax = float(gmax)
        self.a = float(a)

        self.mix = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.g_lin = nn.Conv2d(C, C, kernel_size=1, bias=True)

        if use_refine:
            self.kconv = nn.Conv2d(C, C, kernel_size=kkernel, padding=kkernel // 2, bias=True)
        else:
            self.kconv = nn.Identity()

        self.se = SELite(C, reduce=se_reduce) if use_se else nn.Identity()

        self._init_safe()

    def _init_safe(self) -> None:
        # Zero biases
        nn.init.zeros_(self.mix.bias)
        nn.init.zeros_(self.g_lin.bias)
        if isinstance(self.kconv, nn.Conv2d):
            nn.init.zeros_(self.kconv.bias)
        # Start strictly identity‑like: z≈0, g_pre≈1, kconv≈0
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.g_lin.weight)
        if isinstance(self.kconv, nn.Conv2d):
            nn.init.zeros_(self.kconv.weight)

    def _gain(self, z: torch.Tensor) -> torch.Tensor:
        # g_pre = 1 + clamp(g_lin(z), −a, +a)
        g_pre = 1.0 + torch.clamp(self.g_lin(z), -self.a, self.a)
        return torch.clamp(g_pre, self.gmin, self.gmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mix(x)
        g = self._gain(z)
        if isinstance(self.kconv, nn.Conv2d):
            y = x + self.kconv(x)
        else:
            y = x
        y = self.se(y)
        return y * g
