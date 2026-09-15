"""Blackout occlusion, capped at 32px, run over the FULL dataset (train+val
combined) per class, for the EfficientNet-B0 classifier -- reports the
flip rate (fraction of drawings that can be misclassified with a single
black square of at most 32px) alongside 10 illustrative example panels
per class, honestly labeled as flip or no-flip."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import SketchDataset, CLASSES
from effnet_common import load_model, predict, occlude, sensitivity_map
from display_utils import content_bbox, upsample_for_display

CACHE_ROOT = "data/dataset_cache"

OUT_DIR = "webviz/occlusion_effnet"
os.makedirs(OUT_DIR, exist_ok=True)
UP = 4
MAX_SIZE = 32
STEP = 16


def rank_candidates(heat, patch, stride, img01, top_k=6, ink_thresh=0.92, min_ink_frac=0.15):
    ink_mask = (img01[0].numpy() < ink_thresh)
    flat_idx = np.argsort(heat.flatten())
    coords = []
    for fi in flat_idx:
        iy, ix = np.unravel_index(fi, heat.shape)
        y0, x0 = iy * stride, ix * stride
        patch_ink = ink_mask[y0:y0 + patch, x0:x0 + patch].mean()
        if patch_ink >= min_ink_frac:
            coords.append((y0, x0))
        if len(coords) >= top_k:
            break
    return coords


def try_flip_black(model, img_tensor, label, candidates, base_size=16, max_size=MAX_SIZE, step=STEP):
    img01 = img_tensor[0:1]
    H, W = img01.shape[-2:]
    best = None

    for (cy0, cx0) in candidates:
        cy, cx = cy0 + base_size // 2, cx0 + base_size // 2
        for size in range(base_size, max_size + 1, step):
            yy0 = max(0, min(H - size, cy - size // 2))
            xx0 = max(0, min(W - size, cx - size // 2))
            blacked = occlude(img01, yy0, xx0, size, fill=0.0)
            probs = predict(model, blacked.repeat(3, 1, 1))
            pred = probs.argmax().item()
            if pred != label:
                return size, yy0, xx0, blacked, probs, True
            if best is None or probs[label].item() < best[0]:
                best = (probs[label].item(), size, yy0, xx0, blacked, probs)

    _, size, yy0, xx0, blacked, probs = best
    return size, yy0, xx0, blacked, probs, False


def main():
    model = load_model()
    train_ds = SketchDataset(CACHE_ROOT, "train", augment=False, ext="png")
    val_ds = SketchDataset(CACHE_ROOT, "val", augment=False, ext="png")

    pool = {0: [], 1: []}
    for ds, tag in [(val_ds, "val"), (train_ds, "train")]:
        for i in range(len(ds)):
            _, label = ds.samples[i]
            pool[label].append((ds, tag, i))

    stats = {}
    examples = {0: [], 1: []}  # up to 10 per class, mix of flip/no-flip in dataset order

    for cls_idx, cls_name in enumerate(CLASSES):
        n_total = len(pool[cls_idx])
        n_flipped = 0
        for (ds, tag, i) in pool[cls_idx]:
            img_tensor, label = ds[i]
            img01 = img_tensor[0:1]
            heat = sensitivity_map(model, img_tensor, label, patch=16, stride=8, fill=0.0)
            candidates = rank_candidates(heat, patch=16, stride=8, img01=img01, top_k=6)
            if not candidates:
                # no on-ink candidate at all -- counts as not flipped
                continue
            size, y0, x0, blacked, probs, flipped = try_flip_black(model, img_tensor, label, candidates)
            if flipped:
                n_flipped += 1
            probs_orig = predict(model, img_tensor.unsqueeze(0))
            if len(examples[cls_idx]) < 10:
                examples[cls_idx].append({
                    "tag": tag, "i": i, "size": size, "y0": y0, "x0": x0,
                    "blacked": blacked, "probs": probs, "probs_orig": probs_orig,
                    "img01": img01, "flipped": flipped,
                })
        stats[cls_name] = {"n_total": n_total, "n_flipped": n_flipped,
                            "pct": 100 * n_flipped / n_total}
        print(f"{cls_name}: {n_flipped}/{n_total} flipped ({100*n_flipped/n_total:.1f}%) at <= {MAX_SIZE}px")

    with open("webviz/occlusion_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    examples_meta = {
        CLASSES[cls_idx]: [
            {"rank": r, "size": ex["size"], "flipped": ex["flipped"]}
            for r, ex in enumerate(examples[cls_idx], start=1)
        ]
        for cls_idx in (0, 1)
    }
    with open("webviz/occlusion_examples.json", "w") as f:
        json.dump(examples_meta, f, indent=2)

    # render example panels
    for cls_idx, cls_name in enumerate(CLASSES):
        for rank, ex in enumerate(examples[cls_idx], start=1):
            img01 = ex["img01"]
            base_big = upsample_for_display(img01[0].numpy(), UP)
            x0c, y0c, x1c, y1c = content_bbox(base_big)
            blacked_big = upsample_for_display(ex["blacked"][0].numpy(), UP)

            pred = ex["probs"].argmax().item()
            fig, axes = plt.subplots(1, 2, figsize=(9, 5.0))
            axes[0].imshow(base_big[y0c:y1c, x0c:x1c], cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"{cls_name} #{rank}  orig: {cls_name} ({ex['probs_orig'][cls_idx]:.2f})", fontsize=12)
            axes[0].axis("off")
            axes[1].imshow(blacked_big[y0c:y1c, x0c:x1c], cmap="gray", vmin=0, vmax=1)
            status = "MISCLASSIFIED" if ex["flipped"] else "did not flip"
            axes[1].set_title(f"{ex['size']}px blackout (EfficientNet-B0, max 32px)\npred: {CLASSES[pred]} ({ex['probs'][pred]:.2f}) -- {status}", fontsize=12)
            axes[1].axis("off")
            plt.tight_layout()
            fname = f"{OUT_DIR}/{cls_name}_{rank:02d}.png"
            plt.savefig(fname, dpi=140)
            plt.close(fig)
            print("saved", fname)

    print("STATS", stats)


if __name__ == "__main__":
    main()
