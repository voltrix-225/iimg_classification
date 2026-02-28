"""
main.py - Full pipeline for multilabel classification.

Steps:
    1. Download data from Google Drive
    2. Train model on images + labels
    3. Save model weights (model.pth)
    4. Save loss curve (loss_curve.png)
    5. Run inference on all images and print results

Usage:
    python main.py --drive_url "https://drive.google.com/drive/folders/YOUR_ID"

Or if data is already downloaded:
    python main.py --skip_download --images_dir images --labels labels.txt
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

from dataset import MultilabelDataset, parse_labels, compute_pos_weights, get_transforms
from model import MultilabelResNet

ATTR_NAMES = ['Attr1', 'Attr2', 'Attr3', 'Attr4']


# ---------------------------------------------------------------------------
# Path auto-detection
# ---------------------------------------------------------------------------

def detect_paths(root):
    """Walk root to find the images folder and labels.txt automatically."""
    images_dir = None
    labels_path = None

    for dirpath, dirnames, filenames in os.walk(root):
        if labels_path is None:
            for f in filenames:
                if f == 'labels.txt':
                    labels_path = os.path.join(dirpath, f)

        if images_dir is None:
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in filenames):
                images_dir = dirpath

        if images_dir and labels_path:
            break

    if not images_dir:
        raise FileNotFoundError(f"No images found under '{root}'. Check folder structure.")
    if not labels_path:
        raise FileNotFoundError(f"labels.txt not found under '{root}'.")

    print(f"Auto-detected images dir : {images_dir}")
    print(f"Auto-detected labels file: {labels_path}")
    return images_dir, labels_path


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

def step_download(drive_url, output_dir):
    print("\n=== STEP 1: Downloading data from Google Drive ===")
    try:
        import gdown
    except ImportError:
        raise ImportError("Run: pip install gdown")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(drive_url, output=output_dir, quiet=False, use_cookies=False)
    print(f"Data downloaded to: {output_dir}/")


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, mask):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        masked = bce * mask
        denom = mask.sum()
        return masked.sum() / denom if denom > 0 else torch.tensor(0.0, requires_grad=True)


def step_train(args, device):
    print("\n=== STEP 2: Training ===")

    df = parse_labels(args.labels)
    pos_weights = compute_pos_weights(df, ATTR_NAMES)
    print(f"Positive weights (imbalance correction): {pos_weights.tolist()}")

    dataset = MultilabelDataset(df, args.images_dir, transform=get_transforms(train=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    model = MultilabelResNet(num_classes=4, pretrained=True).to(device)

    backbone_params = [p for n, p in model.named_parameters() if 'model.fc' not in n]
    head_params = list(model.model.fc.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params,     'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    iteration_losses = []
    global_iter = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for images, labels, mask in loader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            global_iter += 1
            loss_val = loss.item()
            epoch_loss += loss_val
            iteration_losses.append((global_iter, loss_val))

        print(f"Epoch [{epoch}/{args.epochs}]  avg_loss={epoch_loss / len(loader):.4f}")

    return model, iteration_losses


# ---------------------------------------------------------------------------
# Step 3: Save model
# ---------------------------------------------------------------------------

def step_save_model(model, path):
    print(f"\n=== STEP 3: Saving model → {path} ===")
    torch.save({'model_state_dict': model.state_dict(), 'num_classes': 4}, path)
    print("Model saved.")


# ---------------------------------------------------------------------------
# Step 4: Save loss curve
# ---------------------------------------------------------------------------

def step_loss_curve(iteration_losses, path):
    print(f"\n=== STEP 4: Saving loss curve → {path} ===")
    iters, losses = zip(*iteration_losses)
    window = max(1, len(losses) // 50)
    smoothed = [sum(losses[max(0, i - window):i + 1]) / len(losses[max(0, i - window):i + 1])
                for i in range(len(losses))]

    plt.figure(figsize=(10, 5))
    plt.plot(iters, losses, linewidth=0.8, alpha=0.5, label='batch loss')
    plt.plot(iters, smoothed, linewidth=2, color='red', label='smoothed')
    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Loss curve saved.")


# ---------------------------------------------------------------------------
# Step 5: Inference on all images
# ---------------------------------------------------------------------------

def step_infer_all(model, args, device):
    print("\n=== STEP 5: Running inference on all images ===")
    model.eval()
    transform = get_transforms(train=False)

    image_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for fname in image_files:
        img_path = os.path.join(args.images_dir, fname)
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().tolist()

        present = [ATTR_NAMES[i] for i, p in enumerate(probs) if p >= args.threshold]
        print(f"{fname:30s} → {str(present if present else 'None'):30s}  "
              f"probs: {[round(p, 3) for p in probs]}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Step 1 — download and auto-detect paths
    if not args.skip_download:
        step_download(args.drive_url, args.data_dir)
        args.images_dir, args.labels = detect_paths(args.data_dir)
    else:
        # Even when skipping download, verify the paths exist
        if not os.path.isdir(args.images_dir):
            raise FileNotFoundError(f"Images dir not found: '{args.images_dir}'. "
                                    f"Try passing --data_dir and removing --skip_download.")
        if not os.path.isfile(args.labels):
            raise FileNotFoundError(f"Labels file not found: '{args.labels}'.")
        print(f"Using images dir : {args.images_dir}")
        print(f"Using labels file: {args.labels}")

    # Step 2
    model, iteration_losses = step_train(args, device)

    # Step 3
    step_save_model(model, args.output_model)

    # Step 4
    step_loss_curve(iteration_losses, args.loss_plot)

    # Step 5
    step_infer_all(model, args, device)

    print("\n=== Pipeline complete ===")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Multilabel Classification Pipeline — Aimonk')

    # Download
    parser.add_argument('--drive_url',     type=str,   default=None,
                        help='Google Drive folder URL')
    parser.add_argument('--data_dir',      type=str,   default='data',
                        help='Where to save/find downloaded data')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download if data already exists locally')

    # Data paths (only needed with --skip_download)
    parser.add_argument('--images_dir',    type=str,   default='images',
                        help='Path to images folder (only needed with --skip_download)')
    parser.add_argument('--labels',        type=str,   default='labels.txt',
                        help='Path to labels.txt (only needed with --skip_download)')

    # Training
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num_workers',   type=int,   default=4)

    # Outputs
    parser.add_argument('--output_model',  type=str,   default='model.pth')
    parser.add_argument('--loss_plot',     type=str,   default='loss_curve.png')

    # Inference
    parser.add_argument('--threshold',     type=float, default=0.5)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.skip_download and not args.drive_url:
        raise ValueError("Provide --drive_url or use --skip_download if data is already local.")

    run_pipeline(args)