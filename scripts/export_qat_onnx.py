#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export helper for QAT checkpoints (FakeQuant-inclusive) to ONNX.

Outputs:
  - factorized_relu_{W}x{H}_qat.onnx       : full autoencoder
  - factorized_relu_enc_ae_{W}x{H}_qat.onnx : encoder (g_a)
  - factorized_relu_dec_ae_{W}x{H}_qat.onnx : decoder (g_s)
  - relu_entropy_params.npz                : entropy bottleneck buffers (quantized_cdf, offset, cdf_length, quantiles)
"""
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

from compressai.zoo import bmshj2018_factorized

from src.model_utils import replace_gdn_with_relu
from src.replace_gdn_npu import replace_gdn_with_npu
from src.qat_utils import prepare_qat_inplace_scoped


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint {path} does not contain a 'model' state dict.")
    return ckpt


def _build_model(train_args: Dict[str, Any]) -> nn.Module:
    quality = train_args.get("quality", 8)
    act = train_args.get("act", "relu")
    replace_parts = train_args.get("replace_parts", "encoder")

    model = bmshj2018_factorized(quality=quality, pretrained=False)

    if act == "relu":
        model = replace_gdn_with_relu(model, mode=replace_parts)
    elif act == "gdnish":
        enc_kwargs = dict(
            enc_t=float(train_args.get("enc_t", 2.0)),
            enc_kdw=int(train_args.get("enc_kdw", 3)),
            enc_use_eca=str(train_args.get("enc_eca", "true")).lower() == "true",
            enc_residual=str(train_args.get("enc_residual", "true")).lower() == "true",
        )
        dec_kwargs = dict(
            dec_k=int(train_args.get("dec_k", 3)),
            dec_gmin=float(train_args.get("dec_gmin", 0.5)),
            dec_gmax=float(train_args.get("dec_gmax", 2.0)),
            dec_residual=str(train_args.get("dec_residual", "true")).lower() == "true",
        )
        model = replace_gdn_with_npu(
            model,
            mode=replace_parts,
            verbose=False,
            **enc_kwargs,
            **dec_kwargs,
        )
    else:
        raise ValueError(f"Unsupported act '{act}' in checkpoint args.")

    return model


def _maybe_prepare_qat(model: nn.Module, train_args: Dict[str, Any]) -> None:
    if str(train_args.get("qat", "false")).lower() != "true":
        return
    scope = train_args.get("qat_scope", "all")
    exclude_entropy = str(train_args.get("qat_exclude_entropy", "true")).lower() == "true"
    calib_steps = int(train_args.get("qat_calib_steps", 2000))
    freeze_after = int(train_args.get("qat_freeze_after", 0))
    module_limit = int(train_args.get("qat_module_limit", 0) or 0)
    range_margin = float(train_args.get("qat_range_margin", 0.0) or 0.0)
    prepare_qat_inplace_scoped(
        model,
        scope=scope,
        calib_steps=calib_steps,
        freeze_after=freeze_after,
        exclude_entropy=exclude_entropy,
        module_limit=module_limit,
        range_margin=range_margin,
        verbose=False,
    )


def _sanitize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, val in state_dict.items():
        if key.startswith("_qat_"):
            continue
        if key.endswith(".scale") or key.endswith(".zero_point"):
            if "fake_quant" in key or "activation_post_process" in key:
                clean[key] = val
                continue
            continue
        if isinstance(val, torch.Tensor) and val.is_quantized:
            val = val.dequantize()
        clean[key] = val
    return clean


_copy_symbolic_registered = False


@parse_args("v", "v", "b")
def _copy_symbolic(g, self, src, non_blocking=False):
    return g.op("Identity", src)


def _ensure_copy_symbolic() -> None:
    global _copy_symbolic_registered
    if _copy_symbolic_registered:
        return
    for opset in range(13, 18):
        register_custom_op_symbolic("aten::copy", _copy_symbolic, opset)
        register_custom_op_symbolic("aten::copy_", _copy_symbolic, opset)
    _copy_symbolic_registered = True


class _FullWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.model.g_a(x)
        recon = self.model.g_s(latent)
        return recon


class _EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.g_a(x)


class _DecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.model.g_s(y)


def _save_entropy_params(model: nn.Module, out_path: Path) -> None:
    if not hasattr(model, "entropy_bottleneck"):
        raise RuntimeError("Model does not have an entropy_bottleneck attribute.")
    eb = model.entropy_bottleneck
    arrays = {
        "quantized_cdf": getattr(eb, "_quantized_cdf").cpu().numpy(),
        "offset": getattr(eb, "_offset").cpu().numpy(),
        "cdf_length": getattr(eb, "_cdf_length").cpu().numpy(),
        "quantiles": eb.quantiles.detach().cpu().numpy(),
    }
    np.savez(out_path, **arrays)


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, out_file: Path, input_name: str, output_name: str) -> None:
    _ensure_copy_symbolic()
    torch.onnx.export(
        model,
        dummy_input,
        out_file.as_posix(),
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
        input_names=[input_name],
        output_names=[output_name],
    )


def main():
    parser = argparse.ArgumentParser(description="Export QAT checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fake-quant (.pt) checkpoint")
    parser.add_argument("--width", type=int, required=True, help="Input width")
    parser.add_argument("--height", type=int, required=True, help="Input height")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save ONNX/NPZ (default: checkpoint dir)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = _load_checkpoint(ckpt_path)
    train_args = ckpt.get("args", {})

    model = _build_model(train_args)
    _maybe_prepare_qat(model, train_args)

    if str(train_args.get("freeze_entropy", "false")).lower() == "true" and hasattr(model, "entropy_bottleneck"):
        for p in model.entropy_bottleneck.parameters():
            p.requires_grad = False

    sanitized_state = _sanitize_state_dict(ckpt["model"])
    missing, unexpected = model.load_state_dict(sanitized_state, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys ignored from checkpoint: {unexpected}")
    if hasattr(model, "update"):
        try:
            model.update()
        except Exception as exc:
            print(f"[warn] model.update() failed: {exc}")
    model.eval().cpu()

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = int(args.height), int(args.width)
    dummy_input = torch.zeros(1, 3, h, w)

    full_wrapper = _FullWrapper(model)
    full_path = out_dir / f"factorized_relu_{w}x{h}_qat.onnx"
    export_onnx(full_wrapper, dummy_input, full_path, "x", "x_hat")

    with torch.no_grad():
        latent = model.g_a(dummy_input).detach()

    enc_path = out_dir / f"factorized_relu_enc_ae_{w}x{h}_qat.onnx"
    export_onnx(_EncoderWrapper(model), dummy_input, enc_path, "x", "y")

    dec_path = out_dir / f"factorized_relu_dec_ae_{w}x{h}_qat.onnx"
    export_onnx(_DecoderWrapper(model), latent, dec_path, "y", "x_hat")

    eb_npz = out_dir / "relu_entropy_params.npz"
    _save_entropy_params(model, eb_npz)

    print(f"[export] Saved full model to {full_path}")
    print(f"[export] Saved encoder to {enc_path}")
    print(f"[export] Saved decoder to {dec_path}")
    print(f"[export] Saved entropy parameters to {eb_npz}")


if __name__ == "__main__":
    main()
