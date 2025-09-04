import argparse, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai.zoo import bmshj2018_factorized
from src.dataset_coco import ImageFolder224
from src.losses import recon_loss, psnr
from src.model_utils import replace_gdn_with_relu, forward_reconstruction

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
    model = replace_gdn_with_relu(model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    mss, psn = [], []
    with torch.no_grad():
        for x in tqdm(dl, desc="eval"):
            x = x.to(args.device)
            x_hat = forward_reconstruction(model, x).clamp(0,1)
            _, logs = recon_loss(x_hat, x)
            mss.append(float(logs["ms_ssim"]))
            psn.append(float(psnr(x_hat, x)))

    print(f"MS-SSIM: {sum(mss)/len(mss):.4f}")
    print(f"PSNR   : {sum(psn)/len(psn):.2f} dB")

if __name__ == "__main__":
    main()
