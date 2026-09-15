# Mouse vs. Hominid

A classifier trained on my own hand-drawn mouse and hominid sketches,
fine-tuned from a real ImageNet-pretrained EfficientNet-B0 backbone and
interrogated with GradCAM, blackout occlusion, and de-novo class
synthesis to see what it actually learned to look for.

**Live write-up:** the full results (dataset GIFs, architecture,
GradCAM, occlusion statistics, synthesized examples, and a conclusion)
are published as an interactive page — open
[`webviz/page.html`](webviz/page.html) directly in a browser, or enable
GitHub Pages on this repo to host it at a URL.

This repo covers the current EfficientNet-B0 round of the project. An
earlier round trained a small custom CNN from scratch (this environment
initially couldn't reach `download.pytorch.org`/`huggingface.co`) and
also explored FGSM adversarial examples and a class-conditional DCGAN;
that code isn't included here.

## What's in here

- `data_pipeline.py` — dataset loading/caching (`SketchDataset`, `CLASSES`, `IMG_SIZE`).
- `train_effnet.py` — fine-tunes EfficientNet-B0 (ImageNet weights, backbone mostly frozen) on the 125/32 train/val split. Reaches 100% val accuracy by epoch 18/25.
- `effnet_common.py` — shared inference helpers: `load_model`, `predict`, `gradcam`, `occlude`, `sensitivity_map`.
- `display_utils.py` — small image-display helpers (crop-to-content, upsampling) used by the webviz scripts.
- `webviz_gradcam_effnet.py` — renders 10 GradCAM examples per class.
- `webviz_occlusion_stats.py` — blackout-occlusion robustness test, capped at a 32px square, run over the *full* dataset (not just until N flips are found); reports the true flip rate per class alongside honestly-labeled example panels.
- `freq_prior.py` — radial power-spectrum prior: matches a synthesized drawing's spatial-frequency content to real drawings of that class, rather than just matching total ink coverage.
- `class_visualization_freq.py` — de-novo class synthesis: starts from a blank canvas and runs gradient ascent on the classifier's logit, parameterizing the image as a small number of fixed-width Bezier strokes (so it can only spend "ink" on continuous lines, never scattered points) and using the frequency prior above instead of an ink budget. Generates several independent examples per class.
- `webviz/` — the published page (`page.html`) and every image/gif it embeds.
- `data/dataset_cache/` — the preprocessed 128×128 grayscale PNGs actually used for training (77 hominid + 80 mouse drawings, cropped to ink and cached). The larger raw source photos aren't included here.
- `models_effnet/classifier_best.pth` — the trained classifier weights (epoch-18 checkpoint, 100% val accuracy).

## Key findings

- **Occlusion asymmetry:** capped at a 32px black square and run over the whole dataset, 72.5% of mouse drawings (58/80) can be flipped to "hominid," but only 3.9% of hominid drawings (3/77) can be flipped to "mouse."
- **What each class seems to be "about":** hominid reads as defined mainly by the core/crotch region plus a limb ("limby"); mouse reads as defined by prominent ears plus a close rounded back (a compact aggregation of features). This shows up consistently across GradCAM, occlusion, and synthesis.
- **De-novo synthesis:** frequency-matched synthesis (vs. plain ink-budget matching) produces wobbly, scalloped lines with real hand-drawn-like fine-scale texture, and the class-conditional shape difference (hominid: loosely closed torso/limb loops; mouse: busier, tangled marks) survives the change in prior.

## Running it

```bash
pip install -r requirements.txt

# The ImageNet-pretrained EfficientNet-B0 weights aren't committed here
# (~20MB third-party download). download.pytorch.org / huggingface.co
# may be blocked on some networks -- this GitHub release mirror isn't:
mkdir -p models_effnet
curl -L -o models_effnet/efficientnet-b0-355c32eb.pth \
  https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth

# Train (optional -- models_effnet/classifier_best.pth is already included)
python train_effnet.py

# Regenerate the webviz assets
python webviz_gradcam_effnet.py
python webviz_occlusion_stats.py
python class_visualization_freq.py
```

Then open `webviz/page.html` in a browser.
