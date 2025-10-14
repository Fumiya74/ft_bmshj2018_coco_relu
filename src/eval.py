import argparse, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai.zoo import bmshj2018_factorized
from src.dataset_coco import ImageFolder224
from src.losses import recon_loss, psnr
from src.model_utils import replace_gdn_with_relu, forward_reconstruction, extract_likelihoods


def compute_bpp(likelihoods, x):
    """bit per pixel (bpp) を計算"""
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    total_bits = 0.0
    for v in likelihoods.values():
        total_bits += torch.sum(-torch.log2(v)).item()
    return total_bits / num_pixels


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """
    既存の学習済み model と validation DataLoader を受け取り、
    MS-SSIM / PSNR / BPP の平均値を返すユーティリティ。
    """
    model.eval()
    mss, psn, bpps = [], [], []
    for x in tqdm(val_loader, desc="eval"):
        x = x.to(device)
        # raw_out も取得して likelihoods を算出（BPP計算用）
        x_hat, raw_out = forward_reconstruction(model, x, clamp=True, return_all=True)
        likelihoods = extract_likelihoods(raw_out)
        bpp_val = compute_bpp(likelihoods, x)
        bpps.append(bpp_val)

        _, logs = recon_loss(x_hat, x)
        mss.append(float(logs["ms_ssim"]))
        psn.append(float(psnr(x_hat, x)))
    mean_ms_ssim = sum(mss)/len(mss) if mss else 0.0
    mean_psnr    = sum(psn)/len(psn) if psn else 0.0
    mean_bpp     = sum(bpps)/len(bpps) if bpps else 0.0
    return mean_ms_ssim, mean_psnr, mean_bpp


# 既存の CLI 実行（単体評価）も従来どおり可能
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", type=str, required=True)
    ap.add_argument("--use_prepared", type=str, default="true")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", type=str, required=True)
    return ap.parse_args()


def main():
    args = get_args()
    use_prepared = args.use_prepared.lower() == "true"

    ds = ImageFolder224(args.coco_dir, "val", use_prepared=use_prepared, input_size=args.input_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = bmshj2018_factorized(quality=args.quality, pretrained=True)
    model = replace_gdn_with_relu(model)  # 必要に応じてNPU版などに差し替えてOK
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device)

    mss, psn, bpp = evaluate_model(model, dl, args.device)
    print(f"MS-SSIM: {mss:.4f}")
    print(f"PSNR   : {psn:.2f} dB")
    print(f"BPP    : {bpp:.4f}")


if __name__ == "__main__":
    main()