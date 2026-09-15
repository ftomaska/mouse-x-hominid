"""Small display helpers shared by the webviz scripts (extracted from the
original gradcam_saliency.py so this trimmed repo doesn't need to carry
the from-scratch custom-CNN training code just for three utility
functions)."""
import numpy as np
from PIL import Image


def to_display(img_tensor):
    return img_tensor[0].numpy()  # single channel, already [0,1]


def content_bbox(img01, ink_thresh=0.92, pad_frac=0.12):
    """Bounding box of the actual drawing (non-white pixels), with a
    margin, so the crisp sketch fills the frame instead of sitting tiny
    in a mostly-blank 128x128 canvas."""
    mask = img01 < ink_thresh
    ys, xs = np.where(mask)
    h, w = img01.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = int((x1 - x0) * pad_frac) + 2
    pad_y = int((y1 - y0) * pad_frac) + 2
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    return x0, y0, x1, y1


def upsample_for_display(arr, scale=4):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0
