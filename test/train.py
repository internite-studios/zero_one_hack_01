"""
train.py — MNIST CNN Classifier
================================
Trains a simple CNN on MNIST (0-9 digit classification).
Data must already exist in test/data/ (run download_data.py first).

This script runs on a GPU compute node via SLURM (no internet needed).
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "mnist_cnn.pth"

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CNN Architecture
# ---------------------------------------------------------------------------
class MNISTCNN(nn.Module):
    """Simple CNN for 28×28 MNIST digits → 10 classes (0–9)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Fully connected: 64 channels × 7×7 after two poolings → 128 → 10
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14→7
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def get_dataloaders(data_dir: Path, batch_size: int):
    """Load MNIST from pre-downloaded local files (no internet needed)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean & std
    ])

    train_ds = datasets.MNIST(
        root=str(data_dir), train=True, download=False, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(data_dir), train=False, download=False, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, len(train_ds), len(test_ds)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1:>4d}/{len(loader)}  "
                  f"Loss: {running_loss / (batch_idx + 1):.4f}  "
                  f"Acc: {100.0 * correct / total:.2f}%")

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  MNIST CNN CLASSIFIER — Training on Leonardo")
    print("=" * 60)
    print(f"  Device:      {DEVICE}")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Model path:  {MODEL_PATH}")
    print(f"  Epochs:      {NUM_EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LEARNING_RATE}")
    print("-" * 60)

    # --- Data ---
    print("\n[1/3] Loading MNIST from local files...")
    t0 = time.time()
    train_loader, test_loader, n_train, n_test = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"       Train samples: {n_train:,}  |  Test samples: {n_test:,}")
    print(f"       Load time: {time.time() - t0:.1f}s")

    # --- Model ---
    print("\n[2/3] Building CNN & starting training...")
    model = MNISTCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Parameters: {total_params:,} total  |  {trainable_params:,} trainable")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch, NUM_EPOCHS
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%  "
              f"[{elapsed:.1f}s]")

    # --- Save model ---
    print(f"\n[3/3] Saving model weights to {MODEL_PATH} ...")
    torch.save(model.state_dict(), str(MODEL_PATH))
    print("       Done.")

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE — Final test accuracy: {test_acc:.2f}%")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
