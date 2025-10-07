
import torch
import torch.nn as nn
from typing import Sequence, Dict, Any, Optional, List

def _is_gdn(m: nn.Module) -> bool:
    try:
        from compressai.layers import GDN
        if isinstance(m, GDN):
            return True
    except Exception:
        pass
    name = m.__class__.__name__.lower()
    return "gdn" in name or "generalizeddivisivenorm" in name

def _replace_gdn_in_module(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if _is_gdn(child):
            setattr(module, name, nn.ReLU(inplace=False))
            replaced += 1
        else:
            replaced += _replace_gdn_in_module(child)
    return replaced

def replace_gdn_with_relu(model: nn.Module, mode: str = "encoder") -> nn.Module:
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
    replaced_block = replaced_block.lower()
    train_scope = train_scope.lower()
    if replaced_block not in {"encoder","decoder","all"}:
        raise ValueError("set_trainable_parts: replaced_block must be 'encoder','decoder','all'")
    if train_scope not in {"replaced","replaced+hyper","all"}:
        raise ValueError("set_trainable_parts: train_scope must be 'replaced','replaced+hyper','all'")

    for p in model.parameters():
        p.requires_grad = False

    if train_scope == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    def _enable(module: Optional[nn.Module]):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    if replaced_block in {"encoder","all"} and hasattr(model, "g_a"):
        _enable(model.g_a)
    if replaced_block in {"decoder","all"} and hasattr(model, "g_s"):
        _enable(model.g_s)

    if train_scope == "replaced+hyper":
        _enable(getattr(model, "h_a", None))
        _enable(getattr(model, "h_s", None))
        _enable(getattr(model, "entropy_bottleneck", None))

@torch.no_grad()
def _is_tensor_safe(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all()

def _pick_x_hat(out, key_candidates: Sequence[str]) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in key_candidates:
            if k in out:
                return out[k]
    if hasattr(out, "_fields"):
        for k in list(out._fields):
            if k in ("x_hat","x","recon","x_dec","x_out"):
                return getattr(out, k)
    for k in ("x_hat","x","recon","x_dec","x_out"):
        if hasattr(out, k):
            return getattr(out, k)
    raise KeyError(
        f"forward_reconstruction: could not find x_hat in output. type={type(out)}"
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
    out = model(x)
    x_hat = _pick_x_hat(out, key_candidates)
    if clamp:
        x_hat = x_hat.clamp(clamp_min, clamp_max)
    return (x_hat, out) if return_all else x_hat

class _ForceFP32(nn.Module):
    """
    Wrap a submodule to run in FP32 even under AMP.
    Also *casts inputs to float32* to avoid dtype mismatch inside the wrapped module,
    then casts outputs back to the original input dtype (if it was floating).
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def _to_fp32(self, obj):
        if torch.is_tensor(obj):
            return obj.float()
        elif isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._to_fp32(o) for o in obj)
        elif isinstance(obj, dict):
            return {k: self._to_fp32(v) for k, v in obj.items()}
        return obj

    def _cast_like(self, out, ref_dtype: torch.dtype):
        if torch.is_tensor(out):
            if out.is_floating_point():
                return out.to(ref_dtype)
            return out
        elif isinstance(out, (list, tuple)):
            t = type(out)
            return t(self._cast_like(o, ref_dtype) for o in out)
        elif isinstance(out, dict):
            return {k: self._cast_like(v, ref_dtype) for k, v in out.items()}
        return out

    def forward(self, *args, **kwargs):
        # pick a reference floating dtype from inputs (bf16/fp16/fp32)
        ref_dtype = None
        def _find_ref(o):
            nonlocal ref_dtype
            if torch.is_tensor(o) and o.is_floating_point():
                ref_dtype = ref_dtype or o.dtype
        for a in args: _find_ref(a)
        for v in kwargs.values(): _find_ref(v)

        # upcast inputs to fp32
        args32 = self._to_fp32(args)
        kwargs32 = self._to_fp32(kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            out = self.module(*args32, **kwargs32)

        # cast back to the reference dtype to keep graph consistent
        if ref_dtype is not None and ref_dtype != torch.float32:
            out = self._cast_like(out, ref_dtype)
        return out

def _wrap_if_match(parent: nn.Module, name: str, child: nn.Module, match_names: List[str]) -> bool:
    cname = child.__class__.__name__
    for m in match_names:
        if m and m.lower() in cname.lower():
            setattr(parent, name, _ForceFP32(child))
            return True
    return False

def wrap_modules_for_local_fp32(model: nn.Module, *, policy: str = "none", custom: str = "") -> None:
    """
    指定ポリシーに合致するモジュールを _ForceFP32 でラップして、該当部だけAMPを無効にする。
      - none: 何もしない
      - entropy: EntropyBottleneck / GaussianConditional 系のみ
      - entropy+decoder: 上に加えて decoder 側（g_s配下）の Norm/Softmax/Exp など
      - all_normexp: Softmax/LogSoftmax/Exp/Log/LayerNorm/BatchNorm を全体で
      - custom: カンマ区切りのクラス名部分一致で指定
    """
    if policy == "none":
        return
    if policy == "custom":
        targets = [s.strip() for s in custom.split(",") if s.strip()]
    elif policy == "entropy":
        targets = ["EntropyBottleneck", "GaussianConditional"]
    elif policy == "entropy+decoder":
        targets = ["EntropyBottleneck", "GaussianConditional", "Softmax", "LogSoftmax", "Exp", "Log"]
    elif policy == "all_normexp":
        targets = ["Softmax","LogSoftmax","Exp","Log","LayerNorm","BatchNorm","InstanceNorm","GroupNorm"]
    else:
        targets = []

    def _recurse(mod: nn.Module):
        for name, child in list(mod.named_children()):
            replaced = _wrap_if_match(mod, name, child, targets)
            if not replaced:
                _recurse(child)
    _recurse(model)
    print(f"[local-fp32] policy='{policy}', custom='{custom}' applied (inputs force-cast to fp32).")

def extract_likelihoods(raw_out: Any) -> Dict[str, torch.Tensor]:
    """
    CompressAI forward の戻り値から likelihoods を取り出す。
    """
    if isinstance(raw_out, dict):
        if "likelihoods" in raw_out:
            return raw_out["likelihoods"]
    if hasattr(raw_out, "likelihoods"):
        return getattr(raw_out, "likelihoods")
    if isinstance(raw_out, dict):
        for k in raw_out.keys():
            if "likelihood" in k:
                return raw_out[k]
    raise KeyError("extract_likelihoods: 'likelihoods' not found in model output.")
