#!/usr/bin/env python3
import argparse, os, glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def iter_images(d):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    for e in exts:
        for p in Path(d).glob(e):
            yield p

def resize_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Resize image while keeping aspect ratio so that both dimensions are
    at least target size, then center crop to (target_w, target_h).
    This handles small images by upscaling before cropping.
    """
    w, h = img.size
    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(int(round(w * scale)), target_w)
    new_h = max(int(round(h * scale)), target_h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    return img.crop((left, top, left + target_w, top + target_h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", type=str, required=True, help="COCO ルート（train2017/ val2017/ がある場所）")
    ap.add_argument("--out_dir", type=str, required=True, help="224 クロップ出力先（train/ val/ を作成）")
    ap.add_argument("--size", type=int, default=224, help="出力サイズ（正方形）。--width/--height が未指定の場合に使用")
    ap.add_argument("--width", type=int, default=0, help="出力幅（未指定時は --size）")
    ap.add_argument("--height", type=int, default=0, help="出力高さ（未指定時は --size）")
    ap.add_argument("--include_val", type=str, default="true")
    args = ap.parse_args()

    target_w = args.width if args.width > 0 else args.size
    target_h = args.height if args.height > 0 else args.size

    out_train = Path(args.out_dir)/"train"
    out_val   = Path(args.out_dir)/"val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # train
    src_train = Path(args.coco_dir)/"train2017"
    paths = list(iter_images(src_train))
    for p in tqdm(paths, desc="train"):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = resize_and_center_crop(im, target_w, target_h)
        im.save(out_train/f"{p.stem}.jpg", quality=95)

    # val
    if args.include_val.lower() == "true":
        src_val = Path(args.coco_dir)/"val2017"
        paths = list(iter_images(src_val))
        for p in tqdm(paths, desc="val"):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = resize_and_center_crop(im, target_w, target_h)
            im.save(out_val/f"{p.stem}.jpg", quality=95)

if __name__ == "__main__":
    main()
