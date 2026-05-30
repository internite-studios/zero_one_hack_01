"""
infer.py — MNIST CNN Inference Runner
=======================================
Loads the trained model, extracts 10 random digits from the MNIST test set,
saves them as PNGs to test/input/, runs inference, and writes per-digit
confidence distributions to test/output/.

Usage:
    pixi run --manifest-path test/pixi.toml --environment gpu python3 test/infer.py
"""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "mnist_cnn.pth"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# ---------------------------------------------------------------------------
# Model (must match train.py architecture exactly)
# ---------------------------------------------------------------------------
import torch.nn as nn


class MNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """Return softmax probabilities for a single (or batched) image."""
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1,28,28) → (1,1,28,28)
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)
    return probs.squeeze(0).cpu()  # shape (10,)


def save_digit_png(tensor: torch.Tensor, path: Path):
    """Save a single (1,28,28) tensor as a PNG file."""
    # tensor values are normalized (~N(0,1)); denormalize back to [0,1]
    img = tensor.clone()
    img = img * 0.3081 + 0.1307  # reverse MNIST normalization
    img = img.clamp(0, 1)
    pil_img = to_pil_image(img)
    pil_img.save(str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("  MNIST CNN — INFERENCE RUNNER")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Input dir:   {INPUT_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print("-" * 60)

    # --- Ensure directories exist ---
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("\n[1/4] Loading trained model...")
    model = MNISTCNN(num_classes=10).to(device)
    state = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("       Model loaded.")

    # --- Load MNIST test set ---
    print("\n[2/4] Loading MNIST test set & sampling 10 digits...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_test = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=False, transform=transform
    )

    # Deterministic sample: pick 10 fixed indices spread across digits 0–9
    # We'll hand-pick one of each digit to guarantee diversity
    digit_to_idx = {}
    for idx in range(len(full_test)):
        _, label = full_test[idx]
        label = int(label)
        if label not in digit_to_idx:
            digit_to_idx[label] = idx
        if len(digit_to_idx) == 10:
            break

    sample_indices = [digit_to_idx[d] for d in range(10)]
    print(f"       Selected indices: {sample_indices}")

    # --- Save input PNGs & collect tensors ---
    print("\n[3/4] Saving digit images to test/input/ ...")
    images = []   # list of (tensor, true_label)
    for i, idx in enumerate(sample_indices):
        tensor, label = full_test[idx]
        label = int(label)
        fname = f"digit_{i:02d}_true{label}.png"
        save_digit_png(tensor, INPUT_DIR / fname)
        images.append((tensor, label, fname))
        print(f"       {fname}  (true={label})")

    # Save a grid of all 10 digits for quick visual check
    grid_tensor = torch.stack([t for t, _, _ in images])  # (10,1,28,28)
    grid_img = make_grid(grid_tensor, nrow=5, padding=2, normalize=True)
    save_image(grid_img, INPUT_DIR / "digits_grid.png")
    print(f"       digits_grid.png  (5×2 overview)")

    # --- Run inference ---
    print("\n[4/4] Running inference & writing results...")
    results = []

    for i, (tensor, true_label, fname) in enumerate(images):
        probs = predict_probs(model, tensor, device)  # shape (10,)
        pred_label = int(probs.argmax().item())
        confidence = float(probs[pred_label].item())

        # Top-3
        topk = probs.topk(3)
        top3_labels = [int(x.item()) for x in topk.indices]
        top3_confs = [float(x.item()) for x in topk.values]

        result = {
            "image": fname,
            "true_label": true_label,
            "predicted": pred_label,
            "confidence": round(confidence, 6),
            "correct": pred_label == true_label,
            "probabilities": {
                str(d): round(float(probs[d].item()), 6) for d in range(10)
            },
            "top3": [
                {"digit": lbl, "confidence": round(cf, 6)}
                for lbl, cf in zip(top3_labels, top3_confs)
            ],
        }
        results.append(result)

        # Print per-digit breakdown
        bar = "█" * int(confidence * 40)
        status = "✓" if result["correct"] else f"✗ (expected {true_label})"
        print(f"\n  ┌─ {fname}")
        print(f"  │  True: {true_label}  →  Predicted: {pred_label}  ({confidence*100:.2f}%)  {status}")
        print(f"  │  {bar}")
        print(f"  │  Confidences:")
        for d in range(10):
            marker = "←" if d == pred_label else "  "
            pct = result["probabilities"][str(d)]
            dbar = "█" * int(pct * 25)
            print(f"  │    {d}: {pct:.4f}  {dbar} {marker}")

    # --- Write output files ---
    # JSON: full probability distributions
    json_path = OUTPUT_DIR / "confidences.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n       Wrote {json_path}")

    # CSV: summary table
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "true_label", "predicted", "confidence",
            "correct", "top1", "top1_conf", "top2", "top2_conf", "top3", "top3_conf"
        ])
        for r in results:
            writer.writerow([
                r["image"],
                r["true_label"],
                r["predicted"],
                r["confidence"],
                r["correct"],
                r["top3"][0]["digit"], r["top3"][0]["confidence"],
                r["top3"][1]["digit"], r["top3"][1]["confidence"],
                r["top3"][2]["digit"], r["top3"][2]["confidence"],
            ])
    print(f"       Wrote {csv_path}")

    # --- Summary ---
    n_correct = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 60)
    print(f"  INFERENCE COMPLETE  —  {n_correct}/10 correct")
    print(f"  Input PNGs:   {INPUT_DIR}")
    print(f"  Output data:  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
