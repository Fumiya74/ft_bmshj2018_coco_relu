import os, glob, random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

def _resize_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    scale = max(target_w / float(w), target_h / float(h))
    new_w = max(int(round(w * scale)), target_w)
    new_h = max(int(round(h * scale)), target_h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    return img.crop((left, top, left + target_w, top + target_h))


class ImageFolder224(Dataset):
    def __init__(self, root, split="train", use_prepared=True, input_size=224,
                 input_width=None, input_height=None):
        self.root = Path(root)
        self.split = split
        self.use_prepared = use_prepared
        self.input_width = int(input_width) if input_width is not None else int(input_size)
        self.input_height = int(input_height) if input_height is not None else int(input_size)

        if use_prepared:
            # train/ 以下のサブディレクトリも含めて探索
            self.paths = [p for p in (self.root/split).rglob("*") if p.suffix.lower() in IMG_EXTS]
            self.transform = T.Compose([
                T.Lambda(lambda im: _resize_center_crop(im, self.input_width, self.input_height)),
                T.ToTensor(),
            ])
        else:
            # オンザフライで CenterCrop+Resize
            if split == "train":
                base = self.root/"train2017"
            else:
                base = self.root/"val2017"
            self.paths = [p for p in base.rglob("*") if p.suffix.lower() in IMG_EXTS]
            self.transform = T.Compose([
                T.Lambda(lambda im: _resize_center_crop(im, self.input_width, self.input_height)),
                T.ToTensor(),
            ])

        # 読み込み枚数を表示
        print(f"[{self.split}] {len(self.paths)} images found")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        return x
