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

def center_crop_resize(img, size=224):
    w,h = img.size
    s = min(w,h)
    left = (w - s)//2
    top  = (h - s)//2
    img = img.crop((left, top, left+s, top+s))
    return img.resize((size,size), Image.BICUBIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", type=str, required=True, help="COCO ルート（train2017/ val2017/ がある場所）")
    ap.add_argument("--out_dir", type=str, required=True, help="224 クロップ出力先（train/ val/ を作成）")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--include_val", type=str, default="true")
    args = ap.parse_args()

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
            im = center_crop_resize(im, args.size)
        im.save(out_train/f"{p.stem}.jpg", quality=95)

    # val
    if args.include_val.lower() == "true":
        src_val = Path(args.coco_dir)/"val2017"
        paths = list(iter_images(src_val))
        for p in tqdm(paths, desc="val"):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im = center_crop_resize(im, args.size)
            im.save(out_val/f"{p.stem}.jpg", quality=95)

if __name__ == "__main__":
    main()
