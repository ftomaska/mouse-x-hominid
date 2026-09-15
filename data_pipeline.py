"""
Data loading utilities for the mammals (mouse vs hominid sketch) project.

Key change from the original 2024 notebook: images are kept as continuous
grayscale in [0,1] for training/attacks (no hard binarize-to-{0,1} step in
the main pipeline). The original notebook thresholded every pixel to
exactly 0 or 1 with `gray > 1` after ImageNet-normalizing the image first,
which is why FGSM perturbations later just produced flat noise -- there
was no continuous intensity information left to perturb, and epsilon=0.5
is huge relative to a binary signal anyway.
"""
import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 128
CLASSES = ["hominids", "mouse"]  # alphabetical, matches ImageFolder convention


def load_image_gray(path, size=IMG_SIZE):
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr  # HxW in [0,1], background ~1 (white), ink ~0 (dark)


CACHE_SIZE = 160  # cache a bit larger than IMG_SIZE so augmentation has room


def build_cache(root, cache_root):
    """Pre-resize the (huge, 3024x4032) source photos once to a small
    on-disk cache so training doesn't re-decode+resize full-res JPEGs every
    epoch (that alone made a 60-epoch run take >10 minutes)."""
    os.makedirs(cache_root, exist_ok=True)
    for split in ("train", "val"):
        for cls in CLASSES:
            src_dir = os.path.join(root, split, cls)
            dst_dir = os.path.join(cache_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(src_dir, "*.jpg"))):
                name = os.path.splitext(os.path.basename(p))[0] + ".png"
                dst = os.path.join(dst_dir, name)
                if os.path.exists(dst):
                    continue
                im = Image.open(p).convert("L").resize((CACHE_SIZE, CACHE_SIZE), Image.LANCZOS)
                im.save(dst)


class SketchDataset(Dataset):
    def __init__(self, root, split, size=IMG_SIZE, augment=False, ext="jpg"):
        self.size = size
        self.augment = augment
        self.samples = []
        for label_idx, cls in enumerate(CLASSES):
            for p in sorted(glob.glob(os.path.join(root, split, cls, f"*.{ext}"))):
                self.samples.append((p, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        im_full = Image.open(path).convert("L")
        if im_full.size != (self.size, self.size):
            im_full = im_full.resize((self.size, self.size), Image.LANCZOS)
        img = im_full

        if self.augment:
            import random
            angle = random.uniform(-20, 20)
            img = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # small random translation
            max_shift = int(0.08 * self.size)
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            img = img.transform(
                img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
                fillcolor=255, resample=Image.BILINEAR,
            )
            # mild scale jitter
            if random.random() < 0.5:
                scale = random.uniform(0.85, 1.15)
                new_size = int(self.size * scale)
                img = img.resize((new_size, new_size), Image.BILINEAR)
                if new_size >= self.size:
                    left = (new_size - self.size) // 2
                    img = img.crop((left, left, left + self.size, left + self.size))
                else:
                    canvas = Image.new("L", (self.size, self.size), 255)
                    off = (self.size - new_size) // 2
                    canvas.paste(img, (off, off))
                    img = canvas

        arr = np.asarray(img, dtype=np.float32) / 255.0
        # replicate to 3 channels for pretrained ImageNet backbones
        t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return t, label
