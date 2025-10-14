#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training script for bmshj2018_factorized (CompressAI) with GDN replacement:
- --act relu   : replace GDN/IGDN with ReLU
- --act gdnish : replace GDN/IGDN with NPU-friendly GDNishLiteEnc/Dec blocks (from src.npu_blocks.py)
- Optional QAT (fake-quant) with automatic exclusion of EntropyBottleneck/Hyperprior vicinity
  *and* scope selection: --qat_scope {encoder,decoder,all}
- Optional EntropyBottleneck auxiliary loss optimizer (--eb_aux / --eb_aux_lr)
- Validation recon PNGs per epoch AND original validation images saved once to ./recon/origin
"""
import argparse, os
from pathlib import Path
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as vutils
from torchvision.utils import save_image
from tqdm import tqdm

from compressai.zoo import bmshj2018_factorized
from src.dataset_coco import ImageFolder224
from src.losses import recon_loss, psnr, rd_loss
from src.eval import evaluate_model
from src.model_utils import (
    replace_gdn_with_relu,
    set_trainable_parts,
    forward_reconstruction,
    wrap_modules_for_local_fp32,
    extract_likelihoods,
)
from src.replace_gdn_npu import replace_gdn_with_npu  # << refactored import

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

    ap.add_argument("--act", type=str, choices=["relu","gdnish"], default="relu", help="GDN replacement: relu or gdnish")
    ap.add_argument("--replace_parts", type=str, default="encoder", choices=["encoder","decoder","all"])
    ap.add_argument("--train_scope", type=str, default="replaced+hyper", choices=["replaced","replaced+hyper","all"])

    # NPU-friendly GDNishLite knobs (mapped to src.replace_gdn_npu)
    ap.add_argument("--enc_t", type=float, default=2.0)
    ap.add_argument("--enc_kdw", type=int, default=3)
    ap.add_argument("--enc_eca", type=str, default="true")
    ap.add_argument("--enc_residual", type=str, default="true")
    ap.add_argument("--dec_k", type=int, default=3)
    ap.add_argument("--dec_gmin", type=float, default=0.5)
    ap.add_argument("--dec_gmax", type=float, default=2.0)
    ap.add_argument("--dec_residual", type=str, default="true")

    ap.add_argument("--amp", type=str, default="true")
    ap.add_argument("--amp_dtype", type=str, choices=["fp16","bf16"], default="bf16")
    ap.add_argument("--local_fp32", type=str, default="none", choices=["none","entropy","entropy+decoder","all_normexp","custom"])
    ap.add_argument("--local_fp32_custom", type=str, default="")

    ap.add_argument("--optimizer", type=str, choices=["adam","adamw"], default="adam")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--sched", type=str, choices=["none","cosine","onecycle","plateau"], default="cosine")
    ap.add_argument("--lr_warmup_steps", type=int, default=500)
    ap.add_argument("--onecycle_pct_start", type=float, default=0.1)

    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--overflow_check", type=str, default="true")
    ap.add_argument("--overflow_tol", type=float, default=1e8)

    ap.add_argument("--loss_type", type=str, choices=["recon","rd"], default="recon")
    ap.add_argument("--lambda_bpp", type=float, default=0.01)

    ap.add_argument("--eb_aux", type=str, default="true")
    ap.add_argument("--eb_aux_lr", type=float, default=1e-3)

    # QAT options (+ scope selection)
    ap.add_argument("--qat", type=str, default="false")
    ap.add_argument("--qat_scope", type=str, choices=["encoder","decoder","all"], default="all",
                    help="Where to insert FakeQuant: encoder(g_a), decoder(g_s), or all.")
    ap.add_argument("--qat_exclude_entropy", type=str, default="true")
    ap.add_argument("--qat_calib_steps", type=int, default=2000)
    ap.add_argument("--qat_freeze_after", type=int, default=8000)

    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--wandb", type=str, default="true")

    return ap.parse_args()

def maybe_init_wandb(args):
    use = args.wandb.lower() == "true" and (os.environ.get("WANDB_DISABLED","false").lower() != "true")
    if not use: return None
    try:
        import wandb
        run_id_file = Path(args.save_dir) / "wandb_run_id.txt"
        if args.resume and run_id_file.exists():
            run_id = run_id_file.read_text().strip(); resume_mode = "allow"
            print(f"[wandb] Resuming previous run (id={run_id})")
        else:
            base_id = wandb.util.generate_id()
            run_id = f"{base_id}_{args.act}_{args.replace_parts}_{args.train_scope}"
            run_id_file.write_text(run_id); resume_mode = None
            print(f"[wandb] Starting new run (id={run_id})")
        wandb.init(project=os.environ.get("WANDB_PROJECT", f"bmshj2018_{args.act}"),
                   config=vars(args), id=run_id, resume=resume_mode)
        return wandb
    except Exception as e:
        print(f"W&B 無効化: {e}"); return None

def _wb_log(wb, payload: dict):
    if wb is None: return
    clean = {k: v for k, v in payload.items() if v is not None}
    if clean: wb.log(clean)

@torch.no_grad()
def save_val_recons(model, loader, device, save_root, tag, wb=None, grid_max=16):
    model.eval()
    out_dir = Path(save_root) / tag; out_dir.mkdir(parents=True, exist_ok=True)
    origin_dir = Path(save_root) / "origin"; origin_dir.mkdir(parents=True, exist_ok=True)
    saved = 0; grid_samples = []
    for x in loader:
        x = x.to(device)
        x_for_save = x.detach().cpu().clamp(0, 1)
        x_hat = forward_reconstruction(model, x, clamp=True)
        bs = x_hat.size(0)
        for b in range(bs):
            save_image(x_hat[b], out_dir / f"{saved:05d}.png")
            op = origin_dir / f"{saved:05d}.png"
            if not op.exists(): save_image(x_for_save[b], op)
            if len(grid_samples) < grid_max: grid_samples.append(x_hat[b].cpu())
            saved += 1
    if wb and grid_samples:
        grid = vutils.make_grid(grid_samples, nrow=4, padding=2); wb.log({f"recon/{tag}":[wb.Image(grid, caption=tag)]})
    print(f"[recon] Saved {saved} recon images to {out_dir}")
    print(f"[recon] Originals stored in {origin_dir}")

def _iter_tensors(obj: Any):
    if torch.is_tensor(obj): yield obj
    elif isinstance(obj, dict):
        for v in obj.values(): yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj: yield from _iter_tensors(v)

def _check_overflow(raw_out: Any, tol: float):
    for t in _iter_tensors(raw_out):
        if not torch.isfinite(t).all():
            bad = t[~torch.isfinite(t)]
            raise RuntimeError(f"[overflow] NaN/Inf: shape={t.shape}, example={bad.flatten()[:4]}")
        if (t.abs() > tol).any():
            big = t[t.abs() > tol]
            raise RuntimeError(f"[overflow] too large (>|{tol}|): max_abs={t.abs().max().item()}, ex={big.flatten()[:4]}")

def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, factor: float):
    for pg in optimizer.param_groups: pg["lr"] = base_lr * factor

def _build_optimizer(args, params):
    if args.optimizer == "adamw": return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

def _build_scheduler(args, optimizer, steps_per_epoch: int):
    sched = None
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(1e-8, args.lr*0.01))
    elif args.sched == "onecycle":
        total = max(1, steps_per_epoch * args.epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total,
                                                    pct_start=args.onecycle_pct_start, anneal_strategy="cos")
    elif args.sched == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    return sched

def main():
    args = get_args()
    use_prepared = args.use_prepared.lower() == "true"
    do_overflow = args.overflow_check.lower() == "true"
    enc_eca = args.enc_eca.lower() == "true"
    enc_res = args.enc_residual.lower() == "true"
    dec_res = args.dec_residual.lower() == "true"

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.recon_dir).mkdir(parents=True, exist_ok=True)
    wb = maybe_init_wandb(args)

    tr = ImageFolder224(args.coco_dir, "train", use_prepared=use_prepared, input_size=args.input_size)
    va = ImageFolder224(args.coco_dir, "val",   use_prepared=use_prepared, input_size=args.input_size)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    recon_n = max(1, min(args.recon_count, len(va))) if len(va) > 0 else 0
    recon_loader = None
    if recon_n > 0:
        recon_subset = Subset(va, list(range(recon_n)))
        recon_loader = DataLoader(recon_subset, batch_size=min(8, args.batch_size),
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = bmshj2018_factorized(quality=args.quality, pretrained=True)

    # === GDN replacement ===
    if args.act == "relu":
        model = replace_gdn_with_relu(model, mode=args.replace_parts)
    else:
        # Uses refactored NPU-friendly replacer
        model = replace_gdn_with_npu(
            model,
            mode=args.replace_parts,
            enc_t=args.enc_t, enc_kdw=args.enc_kdw, enc_use_eca=enc_eca, enc_residual=enc_res,
            dec_k=args.dec_k, dec_gmin=args.dec_gmin, dec_gmax=args.dec_gmax, dec_residual=dec_res,
            verbose=True,
        )

    wrap_modules_for_local_fp32(model, policy=args.local_fp32, custom=args.local_fp32_custom)
    set_trainable_parts(model, replaced_block=args.replace_parts, train_scope=args.train_scope)

    base_lr = args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(args, params)
    steps_per_epoch = max(1, len(train_loader))
    sched = _build_scheduler(args, opt, steps_per_epoch)

    # EB aux optimizer (optional)
    use_eb_aux = args.eb_aux.lower() == "true"
    aux_opt = None
    if use_eb_aux:
        eb_params = []
        for m in model.modules():
            if hasattr(m, "entropy_bottleneck"):
                eb_params += list(m.entropy_bottleneck.parameters())
        if len(eb_params) == 0:
            print("[warn] No EntropyBottleneck params found for aux optimizer. --eb_aux will be ignored.")
            use_eb_aux = False
        else:
            aux_opt = torch.optim.Adam(eb_params, lr=args.eb_aux_lr)

    # QAT (optional + scope)
    use_qat = args.qat.lower() == "true"
    if use_qat:
        from src.qat_utils import prepare_qat_inplace_scoped
        prepare_qat_inplace_scoped(
            model,
            scope=args.qat_scope,
            encoder_attr="g_a",
            decoder_attr="g_s",
            exclude_entropy=(args.qat_exclude_entropy.lower() == "true"),
            calib_steps=args.qat_calib_steps,
            freeze_after=args.qat_freeze_after,
            verbose=True,
        )
        print(f"[QAT] enabled scope={args.qat_scope} exclude_entropy={args.qat_exclude_entropy} "
              f"calib_steps={args.qat_calib_steps} freeze_after={args.qat_freeze_after}")
    model.to(args.device)

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

            if args.lr_warmup_steps > 0 and (global_step < args.lr_warmup_steps) and (args.sched in ["none","cosine","plateau"]):
                factor = (global_step + 1) / float(args.lr_warmup_steps)
                _set_lr(opt, base_lr, factor)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                x_hat, raw_out = forward_reconstruction(model, x, clamp=False, return_all=True)
                if do_overflow: _check_overflow(raw_out, tol=args.overflow_tol)

            with torch.cuda.amp.autocast(enabled=False):
                if args.loss_type == "recon":
                    loss, logs = recon_loss(x_hat, x, alpha_l1=args.alpha_l1)
                    rd_components = {
                        "l1": logs.get("l1", None),
                        "one_minus_msssim": float(1.0 - float(logs.get("ms_ssim", 0.0))) if logs.get("ms_ssim", None) is not None else None,
                        "recon_total": float(loss.detach().cpu()),
                        "bpp": None, "lambda_bpp": None, "rd_total": None,
                    }
                    ms_ssim_cur = float(logs.get("ms_ssim", 0.0))
                else:
                    likelihoods = extract_likelihoods(raw_out)
                    loss, logs = rd_loss(x_hat, x, likelihoods, alpha_l1=args.alpha_l1, lambda_bpp=args.lambda_bpp)
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
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(opt); clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(opt); scaler.update()
            if args.sched == "onecycle": sched.step()

            cur_psnr = psnr(x_hat.detach().clamp(0, 1), x).item()

            # QAT observer scheduling
            if args.qat.lower() == "true":
                from src.qat_utils import step_qat_schedule
                step_qat_schedule(model, global_step)

            if args.loss_type == "rd":
                pbar.set_postfix({
                    "RD": f"{rd_components['rd_total']:.4f}",
                    "L1": f"{rd_components['l1']:.4f}" if rd_components['l1'] is not None else "n/a",
                    "1-MSS": f"{rd_components['one_minus_msssim']:.4f}" if rd_components['one_minus_msssim'] is not None else "n/a",
                    "BPP": f"{rd_components['bpp']:.4f}" if rd_components["bpp"] is not None else "n/a",
                    "λ": f"{rd_components['lambda_bpp']:.3g}" if rd_components["lambda_bpp"] is not None else "n/a",
                    "MSSSIM": f"{ms_ssim_cur:.4f}", "PSNR": f"{cur_psnr:.2f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}", "amp": int(amp_enabled),
                })
            else:
                pbar.set_postfix({
                    "loss": f"{float(loss):.4f}", "MSSSIM": f"{ms_ssim_cur:.4f}", "PSNR": f"{cur_psnr:.2f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}", "amp": int(amp_enabled),
                })

            if wb is not None:
                wb.log({
                    "train/psnr": cur_psnr, "train/msssim": ms_ssim_cur,
                    "train_param/lr": opt.param_groups[0]["lr"], "train_param/amp_enabled": int(amp_enabled),
                })
                if args.loss_type == "rd":
                    wb.log({
                        "train/rd/total": rd_components["rd_total"],
                        "train/rd/recon_total": rd_components["recon_total"],
                        "train/rd/l1": rd_components["l1"],
                        "train/rd/1-msssim": rd_components["one_minus_msssim"],
                        "train/rd/bpp": rd_components["bpp"],
                        "train/rd/lambda_bpp": rd_components["lambda_bpp"],
                    })
                else:
                    wb.log({"train/loss": float(loss)})

            avg_loss += float(loss.detach().cpu()); global_step += 1

            if use_eb_aux and aux_opt is not None:
                aux_opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=False):
                    aux_loss = 0.0
                    for m in model.modules():
                        if hasattr(m, "entropy_bottleneck"):
                            eb = m.entropy_bottleneck
                            if hasattr(eb, "loss"):
                                aux_loss = aux_loss + eb.loss()
                aux_loss.backward()
                aux_opt.step()
                if wb is not None:
                    wb.log({"train/aux_loss": float(aux_loss)})

        avg_loss /= max(1, len(train_loader))

        # ===== Validation（eval.pyの関数を呼び出し） =====
        mean_mss, mean_ps, mean_bpp = evaluate_model(model, val_loader, args.device)
        print(f"[val] epoch={epoch+1}  MS-SSIM={mean_mss:.4f}  PSNR={mean_ps:.2f}dB  BPP={mean_bpp:.4f}  avg_loss={avg_loss:.4f}")

        if wb is not None:
            wb.log({"val/ms_ssim": mean_mss, "val/psnr": mean_ps, "val/bpp": mean_bpp, "epoch": epoch+1})

        if args.sched in ["cosine"]:
            sched.step()
        elif args.sched == "plateau":
            sched.step(mean_mss)

        if recon_loader is not None and args.recon_every > 0 and ((epoch + 1) % args.recon_every == 0):
            tag = f"e{epoch+1:03d}"
            save_val_recons(model, recon_loader, args.device, args.recon_dir, tag, wb=wb,
                            grid_max=min(16, args.recon_count))

        # --------- Save checkpoints (training-state) ---------
        ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                "best_msssim": mean_mss, "args": vars(args)}
        if aux_opt is not None:
            ckpt["aux_opt"] = aux_opt.state_dict()
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        # Best score handling
        if mean_mss > getattr(main, "_best_mss_", 0.0):
            main._best_mss_ = mean_mss
            torch.save(ckpt, os.path.join(args.save_dir, "best_msssim.pt"))

            # ---- Deploy snapshot with CDF updated ----
            snap = deepcopy(model).eval()
            if hasattr(snap, "update"):
                print("[best] Building CDF tables for deploy snapshot...")
                snap.update()  # build CDF buffers
            deploy_ckpt = {
                "epoch": epoch,
                "model": snap.state_dict(),  # includes updated CDF buffers
                "best_msssim": mean_mss,
                "args": vars(args),
            }
            out_path = os.path.join(args.save_dir, "best_msssim_cdf.pt")
            torch.save(deploy_ckpt, out_path)
            print(f"[best] Saved deploy snapshot: {out_path}")

    # ===== Post training: finalize CDF and save final_updated.pt =====
    if (args.eb_aux.lower() == "true") and hasattr(model, "update"):
        print("[post] Updating entropy bottleneck CDF tables...")
        model.update()
        final_ckpt = {"model": model.state_dict(), "args": vars(args)}
        out_path = os.path.join(args.save_dir, "final_updated.pt")
        torch.save(final_ckpt, out_path)
        print(f"[post] Saved final model (after CDF update): {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()