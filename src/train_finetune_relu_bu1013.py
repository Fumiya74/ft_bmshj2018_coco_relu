#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script (bmshj2018_factorized with GDN→ReLU) with:
- RD-loss detailed logging (tqdm + Weights & Biases) when --loss_type rd
- Validation recon PNGs per epoch AND original validation images saved once to ./recon/origin
  so you can always tell which originals correspond to reconstructions.
- (NEW) EntropyBottleneck auxiliary loss optimizer (--eb_aux / --eb_aux_lr)
Drop this file over your existing script and run as usual.
"""

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
from src.losses import recon_loss, psnr, rd_loss
from src.model_utils import (
    replace_gdn_with_relu,
    set_trainable_parts,
    forward_reconstruction,
    wrap_modules_for_local_fp32,
    extract_likelihoods,
)

# ========= EB aux helpers =========
def _collect_eb_aux_params(model):
    params = []
    for m in model.modules():
        if hasattr(m, "entropy_bottleneck"):
            params += list(m.entropy_bottleneck.parameters())
    return params

def _compute_eb_aux_loss(model):
    loss = 0.0
    for m in model.modules():
        if hasattr(m, "entropy_bottleneck"):
            eb = m.entropy_bottleneck
            if hasattr(eb, "loss"):
                loss = loss + eb.loss()
    return loss

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

    # AMP 設定
    ap.add_argument("--amp", type=str, default="true", help="true で AMP を有効化")
    ap.add_argument("--amp_dtype", type=str, choices=["fp16","bf16"], default="bf16")

    # 局所FP32ポリシー
    ap.add_argument("--local_fp32", type=str, default="none",
                    choices=["none","entropy","entropy+decoder","all_normexp","custom"],
                    help="特定モジュールを局所的にFP32で実行して数値安定化")
    ap.add_argument("--local_fp32_custom", type=str, default="",
                    help="custom の場合: カンマ区切りでクラス名の部分一致（例: 'EntropyBottleneck,Softmax'）")

    # 学習率最適化（スケジューラ/オプティマイザ）
    ap.add_argument("--optimizer", type=str, choices=["adam","adamw"], default="adam")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--sched", type=str, choices=["none","cosine","onecycle","plateau"], default="cosine")
    ap.add_argument("--lr_warmup_steps", type=int, default=500,
                    help="線形ウォームアップのステップ数（最大学習率に到達するまで）")
    ap.add_argument("--onecycle_pct_start", type=float, default=0.1, help="OneCycleLRのpct_start")

    # 安定化
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="勾配クリッピングのしきい値。0 以下で無効")
    ap.add_argument("--overflow_check", type=str, default="true",
                    help="true のとき forward 出力に NaN/Inf/過大値があれば即停止")
    ap.add_argument("--overflow_tol", type=float, default=1e8,
                    help="過大値（abs>この値）検知で停止")

    # 損失
    ap.add_argument("--loss_type", type=str, choices=["recon","rd"], default="recon",
                    help="'recon' は再構成誤差のみ、'rd' はレート歪み最適化")
    ap.add_argument("--lambda_bpp", type=float, default=0.01, help="RD のレート項の係数（bpp に掛ける）")

    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--wandb", type=str, default="false", help="true で W&B を有効化")

    # ===== EB aux optimizer options =====
    ap.add_argument("--eb_aux", type=str, default="true", help="true で EntropyBottleneck の aux_loss を最適化")
    ap.add_argument("--eb_aux_lr", type=float, default=1e-3, help="EB aux optimizer の学習率（推奨: 1e-3）")
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

def _wb_log(wb, payload: dict):
    """Safe W&B logging (ignores None values, handles wb=None)."""
    if wb is None:
        return
    clean = {k: v for k, v in payload.items() if v is not None}
    if clean:
        wb.log(clean)

@torch.no_grad()
def save_val_recons(model, loader, device, save_root, tag, wb=None, grid_max=16):
    """
    Save reconstructed images under {save_root}/{tag}/NNNNN.png and
    save the corresponding original validation images (once) under {save_root}/origin/NNNNN.png.
    Indexing is consistent across tags: origin/00000.png corresponds to {tag}/00000.png.
    """
    model.eval()
    out_dir = Path(save_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    origin_dir = Path(save_root) / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    grid_samples = []
    for x in loader:
        x = x.to(device)
        # keep unclamped original tensor for saving origin
        x_for_save = x.detach().cpu().clamp(0, 1)
        x_hat = forward_reconstruction(model, x, clamp=True)

        bs = x_hat.size(0)
        for b in range(bs):
            recon_path = out_dir / f"{saved:05d}.png"
            save_image(x_hat[b], recon_path)

            # Save origin once (if not existing); indexing stays consistent even if file already present
            origin_path = origin_dir / f"{saved:05d}.png"
            if not origin_path.exists():
                save_image(x_for_save[b], origin_path)

            if len(grid_samples) < grid_max:
                grid_samples.append(x_hat[b].cpu())
            saved += 1

    if wb and len(grid_samples) > 0:
        grid = vutils.make_grid(grid_samples, nrow=4, padding=2)
        wb.log({f"recon/{tag}": [wb.Image(grid, caption=tag)]})
    print(f"[recon] Saved {saved} recon images to {out_dir}")
    print(f"[recon] Originals stored in {origin_dir} (saved once; existing files are reused)")

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

def _build_optimizer(args, params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

def _build_scheduler(args, optimizer, steps_per_epoch: int):
    sched = None
    if args.sched == "cosine":
        # CosineAnnealingLR: エポック単位で更新
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(1e-8, args.lr*0.01))
    elif args.sched == "onecycle":
        # ステップ単位で更新
        total_steps = max(1, steps_per_epoch * args.epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.onecycle_pct_start, anneal_strategy="cos"
        )
    elif args.sched == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    return sched

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
    wrap_modules_for_local_fp32(model, policy=args.local_fp32, custom=args.local_fp32_custom)
    model.to(args.device)
    set_trainable_parts(model, replaced_block=args.replace_parts, train_scope=args.train_scope)

    # Optimizer / Scheduler
    base_lr = args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(args, params)
    steps_per_epoch = max(1, len(train_loader))
    sched = _build_scheduler(args, opt, steps_per_epoch)

    # ===== EB aux optimizer (optional) =====
    use_eb_aux = args.eb_aux.lower() == "true"
    aux_opt = None
    if use_eb_aux:
        aux_params = _collect_eb_aux_params(model)
        if len(aux_params) == 0:
            print("[warn] No EntropyBottleneck params found for aux optimizer. --eb_aux will be ignored.")
            use_eb_aux = False
        else:
            aux_opt = torch.optim.Adam(aux_params, lr=args.eb_aux_lr)

    # AMP / Scaler
    cuda_available = args.device.startswith("cuda")
    amp_enabled = cuda_available and (args.amp.lower() == "true")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_msssim = 0.0
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "aux_opt" in ckpt and aux_opt is not None:
            try:
                aux_opt.load_state_dict(ckpt["aux_opt"])
                print("[resume] Restored aux optimizer state.")
            except Exception as e:
                print(f"[resume] Skip restoring aux_opt: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_msssim = ckpt.get("best_msssim", 0.0)
        print(f"Resumed from {args.resume} (next epoch = {start_epoch})")

    # 事前の再構成保存（origin も保存）
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

            # LR warmup (linear, batch-wise) — only during warmup
            if args.lr_warmup_steps > 0 and (global_step < args.lr_warmup_steps):
                factor = (global_step + 1) / float(args.lr_warmup_steps)
                _set_lr(opt, base_lr, factor)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                x_hat, raw_out = forward_reconstruction(model, x, clamp=False, return_all=True)
                if do_overflow_check:
                    _check_overflow(raw_out, tol=args.overflow_tol)

            # 損失は必ず FP32
            with torch.cuda.amp.autocast(enabled=False):
                if args.loss_type == "recon":
                    loss, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
                    rd_components = {
                        "l1": logs.get("l1", None),
                        "one_minus_msssim": float(1.0 - float(logs.get("ms_ssim", 0.0))) if logs.get("ms_ssim", None) is not None else None,
                        "recon_total": float(loss.detach().cpu()),
                        "bpp": None,
                        "lambda_bpp": None,
                        "rd_total": None,
                    }
                    ms_ssim_cur = float(logs.get("ms_ssim", 0.0))
                else:
                    likelihoods = extract_likelihoods(raw_out)
                    loss, logs = rd_loss(x_hat, x, likelihoods, alpha_l1=args.alpha_l1, lambda_bpp=args.lambda_bpp)
                    # 再構成内訳（RDとは別に計算し直して表示用に使う）
                    recon_only_loss, recon_logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
                    rd_components = {
                        "l1": float(recon_logs.get("l1", 0.0)),
                        "one_minus_msssim": float(1.0 - float(recon_logs.get("ms_ssim", 0.0))) if recon_logs.get("ms_ssim", None) is not None else None,
                        "recon_total": float(recon_only_loss.detach().cpu()),
                        "bpp": float(logs.get("bpp", 0.0)) if logs.get("bpp", None) is not None else None,
                        "lambda_bpp": float(args.lambda_bpp),
                        "rd_total": float(loss.detach().cpu()),
                    }
                    ms_ssim_cur = float(recon_logs.get("ms_ssim", 0.0))

            scaler.scale(loss).backward()

            # 勾配クリッピング
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(opt)
                clip_grad_norm_(params, args.max_grad_norm)

            scaler.step(opt)
            scaler.update()

            # scheduler update
            if args.sched == "onecycle":
                sched.step()

            cur_psnr = psnr(x_hat.detach().clamp(0, 1), x).item()

            # tqdm 表示：RD時は内訳も表示
            if args.loss_type == "rd":
                pbar.set_postfix({
                    "RD": f"{rd_components['rd_total']:.4f}",
                    "L1": f"{rd_components['l1']:.4f}" if rd_components['l1'] is not None else "n/a",
                    "1-MSS": f"{rd_components['one_minus_msssim']:.4f}" if rd_components['one_minus_msssim'] is not None else "n/a",
                    "BPP": f"{rd_components['bpp']:.4f}" if rd_components["bpp"] is not None else "n/a",
                    "λ": f"{rd_components['lambda_bpp']:.3g}" if rd_components["lambda_bpp"] is not None else "n/a",
                    "MSSSIM": f"{ms_ssim_cur:.4f}",
                    "PSNR": f"{cur_psnr:.2f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                    "amp": int(amp_enabled),
                })
            else:
                pbar.set_postfix({
                    "loss": f"{float(loss):.4f}",
                    "MSSSIM": f"{ms_ssim_cur:.4f}",
                    "PSNR": f"{cur_psnr:.2f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                    "amp": int(amp_enabled),
                })

            # W&B ログ
            _wb_log(wb, {
                "train/psnr": cur_psnr,
                "train/msssim": ms_ssim_cur,
                "train_param/lr": opt.param_groups[0]["lr"],
                "train_param/amp_enabled": int(amp_enabled),
            })
            if args.loss_type == "rd":
                _wb_log(wb, {
                    "train/rd/total": rd_components["rd_total"],
                    "train/rd/recon_total": rd_components["recon_total"],
                    "train/rd/l1": rd_components["l1"],
                    "train/rd/1-msssim": rd_components["one_minus_msssim"],
                    "train/rd/bpp": rd_components["bpp"],
                    "train/rd/lambda_bpp": rd_components["lambda_bpp"],
                })
            else:
                _wb_log(wb, {
                    "train/loss": float(loss),
                })

            avg_loss += float(loss.detach().cpu())
            global_step += 1

            # ===== EB aux step (separate optimizer) =====
            if use_eb_aux and aux_opt is not None:
                aux_opt.zero_grad(set_to_none=True)
                # aux は常に FP32 でOK（数値安定のため）
                with torch.cuda.amp.autocast(enabled=False):
                    aux_loss = _compute_eb_aux_loss(model)
                aux_loss.backward()
                aux_opt.step()
                _wb_log(wb, {"train/aux_loss": float(aux_loss)})

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
        _wb_log(wb, {
            "val/ms_ssim": mean_mss,
            "val/psnr": mean_ps,
            "epoch": epoch+1
        })

        # scheduler (epoch-wise)
        if args.sched in ["cosine"]:
            sched.step()
        elif args.sched == "plateau":
            # 監視指標は MS-SSIM（大きいほど良い）
            sched.step(mean_mss)

        if recon_loader is not None and args.recon_every > 0 and ((epoch + 1) % args.recon_every == 0):
            tag = f"e{epoch+1:03d}"
            save_val_recons(model, recon_loader, args.device, args.recon_dir, tag, wb=wb,
                            grid_max=min(16, args.recon_count))

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_msssim": mean_mss,
            "args": vars(args),
        }
        if aux_opt is not None:
            ckpt["aux_opt"] = aux_opt.state_dict()
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        # best by MS-SSIM
        if mean_mss > getattr(main, "_best_mss_", 0.0):
            main._best_mss_ = mean_mss
            torch.save(ckpt, os.path.join(args.save_dir, "best_msssim.pt"))

    # ===== finalize EB CDF tables =====
    if use_eb_aux and hasattr(model, "update"):
        print("[post] Updating entropy bottleneck CDF tables...")
        model.update()

    print("Done.")

if __name__ == "__main__":
    main()