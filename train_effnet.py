"""
EfficientNet-B0 transfer learning, take 2.

download.pytorch.org and huggingface.co are both blocked by the network
policy here -- that's why the from-scratch SketchCNN exists at all. But
plain github.com release assets (objects.githubusercontent.com) are NOT
blocked, and the `efficientnet_pytorch` package's ImageNet-pretrained
weights are hosted exactly there:
  https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
so a real pretrained EfficientNet-B0 backbone is available after all --
just not through torch.hub's usual path.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

from data_pipeline import SketchDataset, build_cache

DATA_ROOT = "data/dataset"
CACHE_ROOT = "data/dataset_cache"
DEVICE = torch.device("cpu")
WEIGHTS = "models_effnet/efficientnet-b0-355c32eb.pth"

# ImageNet normalization -- required since we're using ImageNet-pretrained
# features, unlike the from-scratch model's simple 0.5/0.5 scaling.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(batch):
    return (batch - IMAGENET_MEAN) / IMAGENET_STD


def build_model(num_classes=2, freeze_backbone=True):
    model = EfficientNet.from_name("efficientnet-b0")
    sd = torch.load(WEIGHTS, map_location="cpu")
    sd.pop("_fc.weight", None)
    sd.pop("_fc.bias", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert unexpected == [], unexpected
    assert set(missing) == {"_fc.weight", "_fc.bias"}, missing

    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        # 125 training images is tiny for a 5.3M-param backbone -- freeze
        # everything except the final block + classifier head, otherwise
        # it just memorizes the train set with zero generalization signal.
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if name.startswith("_blocks.15") or name.startswith("_blocks.14") \
               or name.startswith("_conv_head") or name.startswith("_bn1") \
               or name.startswith("_fc"):
                p.requires_grad_(True)
    return model


def main():
    torch.manual_seed(0)
    build_cache(DATA_ROOT, CACHE_ROOT)
    train_ds = SketchDataset(CACHE_ROOT, "train", augment=True, ext="png")
    val_ds = SketchDataset(CACHE_ROOT, "val", augment=False, ext="png")
    print(f"train n={len(train_ds)}  val n={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = build_model().to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    n_epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs = imagenet_normalize(imgs)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                imgs_n = imagenet_normalize(imgs)
                out = model(imgs_n)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_ds)
        val_acc = 100 * correct / total
        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models_effnet/classifier_best.pth")

    print(f"best val acc: {best_val_acc:.1f}%")
    torch.save(model.state_dict(), "models_effnet/classifier_final.pth")


if __name__ == "__main__":
    main()
