"""
PyTorch benchmark version of the same CNN.

Built to compare against the Keras model from src/model.py — same architecture,
different framework. Useful for understanding how training loops, autograd,
and DataLoaders work under the hood compared to Keras's higher-level API.

Usage:
    python -m src.pytorch_benchmark --epochs 25
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_CLASSES, SEED
)


class VGGStyleCNN(nn.Module):
    """Same architecture as the Keras model, written in PyTorch."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # After 3 pools on 224 -> 28
        flat_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_dataloaders(batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args.batch_size)

    model = VGGStyleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/pytorch_best.pt")
    print(f"\nBest val accuracy (PyTorch): {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
