
import argparse, os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as vutils
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

    # 置換対象と再学習スコープ
    ap.add_argument("--replace_parts", type=str, default="encoder",
                    choices=["encoder","decoder","all"],
                    help="GDN→ReLU 置換を行うブロック")
    ap.add_argument("--train_scope", type=str, default="replaced+hyper",
                    choices=["replaced","replaced+hyper","all"],
                    help="再学習範囲：置換ブロックのみ / 置換+hyperprior+entropy_bottleneck / 全層")

    # 安定化
    ap.add_argument("--amp_warmup_steps", type=int, default=100,
                    help="このステップ数までは AMP を無効化（FP32 で安定化）")
    ap.add_argument("--lr_warmup_steps", type=int, default=500,
                    help="線形ウォームアップのステップ数（最大学習率に到達するまで）")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="勾配クリッピングのしきい値。0 以下で無効")
    ap.add_argument("--overflow_check", type=str, default="true",
                    help="true のとき forward 出力に NaN/Inf/過大値があれば即停止")
    ap.add_argument("--overflow_tol", type=float, default=1e8,
                    help="過大値（abs>この値）検知で停止")

    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--wandb", type=str, default="false", help="true で W&B を有効化")
    return ap.parse_args()

def maybe_init_wandb(args):
    use = args.wandb.lower() == "true" and (os.environ.get("WANDB_DISABLED","false").lower() != "true")
    if not use:
        return None
    try:
        import wandb
        run_id_file = Path(args.save_dir) / "wandb_run_id.txt"
        if args.resume and run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            resume_mode = "allow"
            print(f"[wandb] Resuming previous run (id={run_id})")
        else:
            base_id = wandb.util.generate_id()
            run_id = f"{base_id}_{args.replace_parts}_{args.train_scope}"
            run_id_file.write_text(run_id)
            resume_mode = None
            print(f"[wandb] Starting new run (id={run_id})")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "bmshj2018_relu"),
            config=vars(args),
            id=run_id,
            resume=resume_mode
        )
        return wandb
    except Exception as e:
        print(f"W&B 無効化: {e}")
        return None

@torch.no_grad()
def save_val_recons(model, loader, device, save_root, tag, wb=None, grid_max=16):
    model.eval()
    out_dir = Path(save_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    grid_samples = []
    for x in loader:
        x = x.to(device)
        x_hat = forward_reconstruction(model, x, clamp=True)
        bs = x_hat.size(0)
        for b in range(bs):
            save_image(x_hat[b], out_dir / f"{saved:05d}.png")
            if len(grid_samples) < grid_max:
                grid_samples.append(x_hat[b].cpu())
            saved += 1
    if wb and len(grid_samples) > 0:
        grid = vutils.make_grid(grid_samples, nrow=4, padding=2)
        wb.log({f"recon/{tag}": [wb.Image(grid, caption=tag)]})
    print(f"[recon] Saved {saved} recon images to {out_dir}")

def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)

def _check_overflow(raw_out: Any, tol: float):
    for t in _iter_tensors(raw_out):
        if not torch.isfinite(t).all():
            bad = t[~torch.isfinite(t)]
            raise RuntimeError(f"[overflow-check] NaN/Inf detected: shape={t.shape}, example={bad.flatten()[:4]}")
        if (t.abs() > tol).any():
            big = t[t.abs() > tol]
            raise RuntimeError(f"[overflow-check] overly large values (>|{tol}|): shape={t.shape}, "
                               f"max_abs={t.abs().max().item()}, example={big.flatten()[:4]}")

def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, factor: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * factor

def main():
    args = get_args()
    use_prepared = args.use_prepared.lower() == "true"
    do_overflow_check = args.overflow_check.lower() == "true"

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.recon_dir).mkdir(parents=True, exist_ok=True)
    wb = maybe_init_wandb(args)

    # Data
    tr = ImageFolder224(args.coco_dir, "train", use_prepared=use_prepared, input_size=args.input_size)
    va = ImageFolder224(args.coco_dir, "val",   use_prepared=use_prepared, input_size=args.input_size)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 再構成保存用サブセット
    recon_n = max(1, min(args.recon_count, len(va))) if len(va) > 0 else 0
    recon_loader = None
    if recon_n > 0:
        recon_subset = Subset(va, list(range(recon_n)))
        recon_loader = DataLoader(recon_subset, batch_size=min(8, args.batch_size),
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = bmshj2018_factorized(quality=args.quality, pretrained=True)
    model = replace_gdn_with_relu(model, mode=args.replace_parts)
    model.to(args.device)
    set_trainable_parts(model, replaced_block=args.replace_parts, train_scope=args.train_scope)

    # Optimizer
    base_lr = args.lr
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=base_lr)

    # AMP / Scaler
    cuda_available = args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=cuda_available)

    best_msssim = 0.0
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_msssim = ckpt.get("best_msssim", 0.0)
        print(f"Resumed from {args.resume} (next epoch = {start_epoch})")

    # 事前の再構成保存
    if recon_loader is not None:
        tag = f"pre_e{start_epoch:03d}"
        save_val_recons(model, recon_loader, args.device, args.recon_dir, tag, wb=wb,
                        grid_max=min(16, args.recon_count))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch+1}/{args.epochs}")
        avg_loss = 0.0
        for x in pbar:
            x = x.to(args.device)

            # LR warmup: 線形
            if args.lr_warmup_steps > 0:
                factor = min(1.0, (global_step + 1) / float(args.lr_warmup_steps))
            else:
                factor = 1.0
            _set_lr(opt, base_lr, factor)

            # AMP warmup
            amp_enabled = cuda_available and (global_step + 1 >= args.amp_warmup_steps)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                x_hat, raw_out = forward_reconstruction(model, x, clamp=False, return_all=True)
                if do_overflow_check:
                    _check_overflow(raw_out, tol=args.overflow_tol)

            # 損失は常に FP32
            with torch.cuda.amp.autocast(enabled=False):
                loss, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)

            scaler.scale(loss).backward()

            # 勾配クリッピング
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(opt)
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)

            scaler.step(opt)
            scaler.update()

            avg_loss += float(loss.detach().cpu())
            cur_msssim = float(logs["ms_ssim"])
            cur_psnr   = psnr(x_hat.detach().clamp(0, 1), x).item()
            pbar.set_postfix({
                "loss": f"{float(loss):.4f}",
                "msssim": f"{cur_msssim:.4f}",
                "psnr": f"{cur_psnr:.2f}",
                "lr": f"{opt.param_groups[0]['lr']:.2e}",
                "amp": int(amp_enabled),
            })
            if wb:
                wb.log({
                    "train/loss": float(loss),
                    "train/msssim": cur_msssim,
                    "train/psnr": cur_psnr,
                    "train_param/lr": opt.param_groups[0]["lr"],
                    "train_param/amp_enabled": int(amp_enabled),
                })

            global_step += 1

        avg_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        mss_list, psnr_list = [], []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(args.device)
                x_hat = forward_reconstruction(model, x, clamp=True)
                _, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
                mss_list.append(float(logs["ms_ssim"]))
                psnr_list.append(psnr(x_hat, x).item())

        mean_mss = sum(mss_list)/len(mss_list) if mss_list else 0.0
        mean_ps  = sum(psnr_list)/len(psnr_list) if psnr_list else 0.0

        print(f"[val] epoch={epoch+1}  MS-SSIM={mean_mss:.4f}  PSNR={mean_ps:.2f}dB  avg_loss={avg_loss:.4f}")
        if wb:
            wb.log({
                "val/ms_ssim": mean_mss,
                "val/psnr": mean_ps,
                "epoch": epoch+1
            })
            
        # ★ 既定の間隔で、再構成画像を PNG で保存（ディレクトリ分割）＋ W&B にグリッドをログ
        if recon_loader is not None and args.recon_every > 0 and ((epoch + 1) % args.recon_every == 0):
            tag = f"e{epoch+1:03d}"
            save_val_recons(model, recon_loader, args.device, args.recon_dir, tag, wb=wb,
                            grid_max=min(16, args.recon_count))
        # Save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_msssim": max(best_msssim, mean_mss),
            "args": vars(args),
        }
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        if mean_mss > best_msssim:
            best_msssim = mean_mss
            torch.save(ckpt, os.path.join(args.save_dir, "best_msssim.pt"))

    print("Done.")

if __name__ == "__main__":
    main()