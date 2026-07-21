"""
Train a mini ship classifier (Naval vs Civilian) using transfer learning.

Usage:
    pip install torch torchvision pillow --break-system-packages
    python scripts/train.py --data_dir data --epochs 15 --out model/ship_classifier.pth

Expected folder layout:
    data/train/naval/*.jpg
    data/train/civilian/*.jpg
    data/val/naval/*.jpg
    data/val/civilian/*.jpg
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


class ShipDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/ship_classifier.pth")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Skip downloading ImageNet weights (random init) — useful for offline smoke tests.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ShipDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_ds = ShipDataset(os.path.join(args.data_dir, "val"), val_tf)
    print(f"Classes: {train_ds.classes}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = build_model(num_classes=len(train_ds.classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_dl, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
            }, args.out)
            print(f"  -> saved new best model (val_acc={val_acc:.3f}) to {args.out}")

    print(f"Training complete. Best val_acc={best_val_acc:.3f}. Model saved to {args.out}")


if __name__ == "__main__":
    main()
