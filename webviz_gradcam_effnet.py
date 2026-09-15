"""GradCAM examples (10/class) for the EfficientNet-B0 transfer-learning
classifier, same format as webviz_gradcam.py but targeting effnet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os

from data_pipeline import SketchDataset, CLASSES
from effnet_common import load_model, gradcam
from display_utils import to_display, content_bbox, upsample_for_display

CACHE_ROOT = "data/dataset_cache"

OUT_DIR = "webviz/gradcam_effnet"
os.makedirs(OUT_DIR, exist_ok=True)
UPSCALE = 4


def main():
    val_ds = SketchDataset(CACHE_ROOT, "val", augment=False, ext="png")
    model = load_model()

    by_class = {0: [], 1: []}
    for i in range(len(val_ds)):
        _, label = val_ds.samples[i]
        by_class[label].append(i)

    for cls_idx, cls_name in enumerate(CLASSES):
        idxs = by_class[cls_idx][:10]
        for rank, idx in enumerate(idxs):
            img_tensor, label = val_ds[idx]
            cam, pred_cam, logits = gradcam(model, img_tensor)
            base = to_display(img_tensor)

            base_big = upsample_for_display(base, UPSCALE)
            cam_big = upsample_for_display(cam, UPSCALE)
            x0, y0, x1, y1 = content_bbox(base_big)
            base_c = base_big[y0:y1, x0:x1]
            cam_c = cam_big[y0:y1, x0:x1]

            probs = torch.softmax(logits, dim=1)[0]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
            axes[0].imshow(base_c, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"{cls_name} #{rank+1}", fontsize=13)
            axes[0].axis("off")
            axes[1].imshow(base_c, cmap="gray", vmin=0, vmax=1)
            axes[1].imshow(cam_c, cmap="jet", alpha=0.45)
            axes[1].set_title(f"GradCAM (EfficientNet-B0)  pred: {CLASSES[pred_cam]} ({probs[pred_cam]:.2f})", fontsize=12)
            axes[1].axis("off")
            plt.tight_layout()
            fname = f"{OUT_DIR}/{cls_name}_{rank+1:02d}.png"
            plt.savefig(fname, dpi=140)
            plt.close(fig)
            print("saved", fname)


if __name__ == "__main__":
    main()
