
"""
Export model (full / encoder / decoder) to ONNX.
Usage:
  python -m scripts.export_onnx --ckpt checkpoints/last.pt --out model_full.onnx --part full --input_size 224 --batch 1
  python -m scripts.export_onnx --ckpt checkpoints/last.pt --out encoder.onnx --part encoder --input_size 224 --batch 1
"""
import argparse, os
from pathlib import Path
import torch
import torch.nn as nn
from compressai.zoo import bmshj2018_factorized
from src.model_utils import replace_gdn_with_relu

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="学習済みチェックポイント（省略可）")
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--replace_parts", type=str, default="encoder", choices=["encoder","decoder","all"])
    ap.add_argument("--part", type=str, default="full", choices=["full","encoder","decoder"])
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamic", action="store_true", help="動的軸を有効化（H,W を可変に）")
    return ap.parse_args()

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.g_a = model.g_a
    def forward(self, x):
        # latent y
        return self.g_a(x)

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.g_s = model.g_s
    def forward(self, y):
        return self.g_s(y)

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = bmshj2018_factorized(quality=args.quality, pretrained=True)
    model = replace_gdn_with_relu(model, mode=args.replace_parts)
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[load] {args.ckpt} loaded.")
    model.eval().to(device)

    if args.part == "encoder":
        export_model = EncoderWrapper(model).to(device)
        dummy = torch.randn(args.batch, 3, args.input_size, args.input_size, device=device)
        input_names = ["x"]
        output_names = ["y"]
        dyn_axes = { "x": {0:"N", 2:"H", 3:"W"}, "y": {0:"N", 2:"H", 3:"W"} } if args.dynamic else None
    elif args.part == "decoder":
        # 推定 latent チャネルはモデルの g_s 入力チャネル数に合わせる
        c = model.g_s[0].in_channels if hasattr(model.g_s[0], "in_channels") else 192
        export_model = DecoderWrapper(model).to(device)
        dummy = torch.randn(args.batch, c, args.input_size//16, args.input_size//16, device=device)
        input_names = ["y"]
        output_names = ["x_hat"]
        dyn_axes = { "y": {0:"N", 2:"H", 3:"W"}, "x_hat": {0:"N", 2:"H", 3:"W"} } if args.dynamic else None
    else:
        export_model = model.to(device)
        dummy = torch.randn(args.batch, 3, args.input_size, args.input_size, device=device)
        input_names = ["x"]
        output_names = ["x_hat"]  # 実際は複数出力だが、最初のテンソルを主出力と想定
        dyn_axes = { "x": {0:"N", 2:"H", 3:"W"}, "x_hat": {0:"N", 2:"H", 3:"W"} } if args.dynamic else None

    onnx_path = Path(args.out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            export_model, dummy, onnx_path.as_posix(),
            input_names=input_names, output_names=output_names,
            opset_version=args.opset, dynamic_axes=dyn_axes, do_constant_folding=True
        )
    print(f"[exported] {onnx_path}")

if __name__ == "__main__":
    main()
