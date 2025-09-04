import argparse, os, json, time, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from compressai.zoo import bmshj2018_factorized
from src.dataset_coco import ImageFolder224
from src.losses import recon_loss, psnr
from src.model_utils import replace_gdn_with_relu, set_trainable_parts, forward_reconstruction

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", type=str, required=True, help="COCO 224 データ or COCO ルート")
    ap.add_argument("--use_prepared", type=str, default="true")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=8, help="bmshj2018 factorized quality [0..8]")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha_l1", type=float, default=0.4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--recon_dir", type=str, default="./recon")
    ap.add_argument("--recon_every", type=int, default=2)
    ap.add_argument("--recon_count", type=int, default=16)
    ap.add_argument("--train_parts", type=str, default="decoder", choices=["decoder","decoder+encoder","all"])
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--wandb", type=str, default="false")
    return ap.parse_args()

def main():
    args = get_args()
    use_prepared = args.use_prepared.lower() == "true"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.recon_dir, exist_ok=True)

    if args.wandb.lower() == "true":
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "bmshj2018_relu"),
                   config=vars(args))
    else:
        wandb = None

    # Data
    tr = ImageFolder224(args.coco_dir, "train", use_prepared=use_prepared, input_size=args.input_size)
    va = ImageFolder224(args.coco_dir, "val",   use_prepared=use_prepared, input_size=args.input_size)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = bmshj2018_factorized(quality=args.quality, pretrained=True)
    model = replace_gdn_with_relu(model)
    model.to(args.device)
    set_trainable_parts(model, args.train_parts)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda")))

    best_msssim = 0.0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_msssim = ckpt.get("best_msssim", 0.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch+1}/{args.epochs}")
        avg_loss = 0.0
        for x in pbar:
            x = x.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.device.startswith("cuda"))):
                x_hat = forward_reconstruction(model, x)
                loss, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            avg_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "msssim": f"{logs['ms_ssim'].item():.4f}"})
            if wandb: 
                wandb.log({"train/loss": loss.item(), "train/msssim": logs["ms_ssim"].item()})
        avg_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        mss_list, psnr_list = [], []
        with torch.no_grad():
            for i, x in enumerate(val_loader):
                x = x.to(args.device)
                x_hat = forward_reconstruction(model, x)
                _, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
                mss_list.append(logs["ms_ssim"].item())
                psnr_list.append(psnr(x_hat, x).item())
                # Recon dump
                if (epoch+1) % args.recon_every == 0 and i == 0:
                    # 0..1 にクリップして保存
                    n = min(args.recon_count, x.size(0))
                    grid = torch.stack([x[:n], x_hat[:n].clamp(0,1)], dim=1).reshape(2*n, *x.shape[1:])
                    save_image(grid, os.path.join(args.recon_dir, f"e{epoch+1:03d}_recon.png"), nrow=2, padding=2)
        mean_mss = sum(mss_list)/len(mss_list) if mss_list else 0.0
        mean_ps  = sum(psnr_list)/len(psnr_list) if psnr_list else 0.0
        if wandb:
            wandb.log({"val/ms_ssim": mean_mss, "val/psnr": mean_ps, "epoch": epoch+1})
        print(f"[val] epoch={epoch+1}  MS-SSIM={mean_mss:.4f}  PSNR={mean_ps:.2f}dB")

        # Save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_msssim": max(best_msssim, mean_mss),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"last.pt"))
        if mean_mss > best_msssim:
            best_msssim = mean_mss
            torch.save(ckpt, os.path.join(args.save_dir, f"best_msssim.pt"))

    print("Done.")

if __name__ == "__main__":
    main()
