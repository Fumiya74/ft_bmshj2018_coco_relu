import os, glob, random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

class ImageFolder224(Dataset):
    def __init__(self, root, split="train", use_prepared=True, input_size=224):
        self.root = Path(root)
        self.split = split
        self.use_prepared = use_prepared
        self.input_size = input_size

        if use_prepared:
            # train/ 以下のサブディレクトリも含めて探索
            self.paths = [p for p in (self.root/split).rglob("*") if p.suffix.lower() in IMG_EXTS]
            self.transform = T.ToTensor()
        else:
            # オンザフライで CenterCrop+Resize
            if split == "train":
                base = self.root/"train2017"
            else:
                base = self.root/"val2017"
            self.paths = [p for p in base.rglob("*") if p.suffix.lower() in IMG_EXTS]
            self.transform = T.Compose([
                T.CenterCrop(min(480, 0x7FFFFFFF)),
                T.Resize((input_size,input_size)),
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
