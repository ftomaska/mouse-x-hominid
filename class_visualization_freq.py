"""De-novo class synthesis, take 3: stroke-parameterized (continuous-line
prior, unchanged from class_visualization_strokes*.py) but the ink-budget
match is replaced with a spatial-frequency match -- rewarding the
synthesized image for having the same radial power-spectrum shape as
real drawings of that class, rather than just the same total ink
fraction. Targets the EfficientNet-B0 classifier. Generates several
(N_SAMPLES) independent examples per class rather than one "best" pick.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_effnet import imagenet_normalize as normalize
from effnet_common import load_model as _load_effnet_model
from data_pipeline import SketchDataset, CLASSES, IMG_SIZE
from freq_prior import radial_spectrum, reference_spectrum

CACHE_ROOT = "data/dataset_cache"

DEVICE = torch.device("cpu")
N_SAMPLES_PER_CLASS = 16
N_SHOWN_PER_CLASS = 8
WIDTH_PX = 2.2


def load_model():
    model = _load_effnet_model()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def bezier_points(ctrl, n_samples=24):
    t = torch.linspace(0, 1, n_samples).view(1, n_samples, 1)
    p0, p1, p2, p3 = ctrl[:, 0:1], ctrl[:, 1:2], ctrl[:, 2:3], ctrl[:, 3:4]
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + \
          3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return pts


def render_strokes(ctrl, widths, size=IMG_SIZE, sharpness=2500.0):
    n_strokes = ctrl.shape[0]
    pts = bezier_points(ctrl)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
    )
    grid = torch.stack([xs, ys], dim=-1).view(1, 1, size, size, 2)
    pts_ = pts.view(n_strokes, pts.shape[1], 1, 1, 2)
    d2 = ((grid - pts_) ** 2).sum(-1)
    min_d2, _ = d2.min(dim=1)
    w = widths.view(n_strokes, 1, 1) / size
    ink_s = torch.sigmoid((w ** 2 - min_d2) * sharpness)
    ink = 1 - torch.prod(1 - ink_s, dim=0)
    return ink.view(1, 1, size, size)


def random_affine(img, max_rot_deg=8, scale_range=(0.94, 1.06), max_shift_frac=0.04):
    n = img.shape[0]
    angle = (torch.rand(n) * 2 - 1) * max_rot_deg * np.pi / 180
    scale = torch.empty(n).uniform_(*scale_range)
    shift = (torch.rand(n, 2) * 2 - 1) * max_shift_frac
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = shift[:, 0]
    theta[:, 1, 2] = shift[:, 1]
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border")


def synthesize(model, target_class, ref_spectrum, n_strokes=4, steps=500, lr=0.015,
               seed=0, freq_weight=0.6, min_chord=0.28, chord_weight=40.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ctrl = torch.zeros(n_strokes, 4, 2)
    for s in range(n_strokes):
        cx, cy = np.random.uniform(0.3, 0.7, size=2)
        ang = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0.15, 0.3)
        dx, dy = np.cos(ang) * length, np.sin(ang) * length
        p0 = np.array([cx - dx / 2, cy - dy / 2])
        p3 = np.array([cx + dx / 2, cy + dy / 2])
        p1 = p0 + (p3 - p0) * 0.33 + np.random.uniform(-0.05, 0.05, size=2)
        p2 = p0 + (p3 - p0) * 0.66 + np.random.uniform(-0.05, 0.05, size=2)
        ctrl[s] = torch.tensor(np.array([p0, p1, p2, p3]), dtype=torch.float32)
    ctrl.requires_grad_(True)

    widths = torch.full((n_strokes,), WIDTH_PX)
    opt = torch.optim.Adam([ctrl], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        ink = render_strokes(ctrl, widths)
        img = 1 - ink

        transformed = random_affine(img)
        rgb = transformed.repeat(1, 3, 1, 1)
        logits = model(normalize(rgb))
        class_score = logits[0, target_class]

        spec = radial_spectrum(ink)
        freq_loss = F.mse_loss(spec, ref_spectrum)

        bounds_penalty = (F.relu(-ctrl) + F.relu(ctrl - 1)).pow(2).sum()
        chord = (ctrl[:, 3] - ctrl[:, 0]).norm(dim=-1)
        chord_penalty = F.relu(min_chord - chord).pow(2).sum()

        loss = -class_score + freq_weight * freq_loss + 5.0 * bounds_penalty \
            + chord_weight * chord_penalty
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_ink = render_strokes(ctrl, widths)
        final_img = 1 - final_ink
        final_logits = model(normalize(final_img.repeat(1, 3, 1, 1)))
        final_probs = F.softmax(final_logits, dim=1)[0]

    return final_img.detach()[0, 0].numpy(), final_probs.numpy()


def upsample_for_display(arr, scale=4):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def main():
    out_dir = "webviz/class_viz_freq"
    os.makedirs(out_dir, exist_ok=True)
    model = load_model()

    train_ds = SketchDataset(CACHE_ROOT, "train", augment=False, ext="png")
    by_class_imgs = {0: [], 1: []}
    for i in range(len(train_ds)):
        img, label = train_ds[i]
        by_class_imgs[label].append(img[0])

    refs = {cls_name: reference_spectrum(by_class_imgs[cls_idx])
            for cls_idx, cls_name in enumerate(CLASSES)}

    for cls_idx, cls_name in enumerate(CLASSES):
        ref_spectrum = refs[cls_name]
        results = []
        for seed in range(N_SAMPLES_PER_CLASS):
            img, probs = synthesize(model, cls_idx, ref_spectrum, seed=seed)
            ink_frac = 1 - img.mean()
            print(f"{cls_name} seed={seed}  p(target)={probs[cls_idx]:.3f}  ink={ink_frac:.4f}")
            results.append((img, probs, ink_frac))

        # keep a handful of the most confidently-classified examples to display
        results.sort(key=lambda r: -r[1][cls_idx])
        shown = results[:N_SHOWN_PER_CLASS]

        ncols = 4
        nrows = (len(shown) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, (img, probs, ink_frac) in zip(axes, shown):
            big = upsample_for_display(img, scale=5)
            ax.imshow(big, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f'"{cls_name}"  p={probs[cls_idx]:.2f}\n(ink={ink_frac:.3f}, freq-matched)', fontsize=11)
            ax.axis("off")
        for ax in axes[len(shown):]:
            ax.axis("off")
        plt.tight_layout()
        fname = f"{out_dir}/{cls_name}_grid.png"
        plt.savefig(fname, dpi=160)
        plt.close(fig)
        print("saved", fname)


if __name__ == "__main__":
    main()
