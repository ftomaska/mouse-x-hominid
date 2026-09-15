"""Shared helpers for the EfficientNet-B0 transfer-learning classifier --
load, normalize, GradCAM, occlusion -- mirroring gradcam_saliency.py /
occlusion.py but for the pretrained backbone instead of the from-scratch
SketchCNN."""
import numpy as np
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from train_effnet import build_model, imagenet_normalize, WEIGHTS

DEVICE = torch.device("cpu")
CKPT = "models_effnet/classifier_best.pth"


def load_model():
    model = EfficientNet.from_name("efficientnet-b0")
    import torch.nn as nn
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    return model


def forward_with_features(model, x):
    """Runs the classifier and also returns the final conv feature map
    (post conv_head+bn+swish, pre pooling) with its grad retained, for
    GradCAM -- EfficientNet's own extract_features() gives us this
    directly instead of needing a forward/backward hook on a Sequential
    index like the custom CNN version does."""
    features = model.extract_features(x)
    features.retain_grad()
    pooled = model._avg_pooling(features).flatten(1)
    logits = model._fc(model._dropout(pooled))
    return logits, features


@torch.no_grad()
def predict(model, rgb01):
    """rgb01: (3,H,W) or (1,3,H,W) tensor in [0,1] RGB (grayscale sketch
    replicated to 3 channels, same convention as the rest of the repo)."""
    if rgb01.dim() == 3:
        rgb01 = rgb01.unsqueeze(0)
    logits = model(imagenet_normalize(rgb01))
    return F.softmax(logits, dim=1)[0]


def gradcam(model, img_tensor, target_class=None):
    """img_tensor: (3,H,W) in [0,1]. Returns (cam as HxW numpy in [0,1],
    predicted class, raw logits)."""
    img = img_tensor.unsqueeze(0).clone().requires_grad_(True)
    logits, features = forward_with_features(model, imagenet_normalize(img))
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1
    logits.backward(gradient=one_hot)

    acts = features[0].detach()       # C,H,W
    grads = features.grad[0]          # C,H,W
    weights = grads.mean(dim=(1, 2))  # C
    cam = F.relu((weights[:, None, None] * acts).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().numpy(), target_class, logits.detach()


def occlude(img01, y0, x0, size, fill):
    out = img01.clone()
    out[:, y0:y0 + size, x0:x0 + size] = fill
    return out


@torch.no_grad()
def sensitivity_map(model, img_tensor, true_label, patch=16, stride=8, fill=0.0):
    img01 = img_tensor[0:1]
    H, W = img01.shape[-2:]
    n_y = (H - patch) // stride + 1
    n_x = (W - patch) // stride + 1
    heat = np.zeros((n_y, n_x))
    for iy in range(n_y):
        for ix in range(n_x):
            y0, x0 = iy * stride, ix * stride
            occluded = occlude(img01, y0, x0, patch, fill)
            probs = predict(model, occluded.repeat(3, 1, 1))
            heat[iy, ix] = probs[true_label].item()
    return heat
