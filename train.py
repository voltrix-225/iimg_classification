"""
train.py - Training script for multilabel classification.

Usage:
    python train.py --images_dir images --labels labels.txt \
                    --output_model model.pth --epochs 20 --batch_size 32

Deliverables produced:
  1. model.pth  - saved model weights
  2. loss_curve.png - training loss vs iteration plot
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MultilabelDataset, parse_labels, compute_pos_weights, get_transforms
from model import MultilabelResNet



# Masked BCE loss: ignores NA entries

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Computes BCEWithLogitsLoss only on valid (non-NA) entries.
    Supports per-attribute positive weighting for class imbalance.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight  # tensor of shape (num_classes,)

    def forward(self, logits, targets, mask):
        # logits, targets, mask: (B, C)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )  # (B, C)
        masked = bce * mask
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, requires_grad=True)
        return masked.sum() / denom

# Training loop

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    df = parse_labels(args.labels)
    pos_weights = compute_pos_weights(df, ['Attr1', 'Attr2', 'Attr3', 'Attr4'])
    print(f"Positive weights (for class imbalance): {pos_weights.tolist()}")

    dataset = MultilabelDataset(df, args.images_dir, transform=get_transforms(train=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    model = MultilabelResNet(num_classes=4, pretrained=True).to(device)

    # Fine-tuning strategy: lower LR for backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if 'model.fc' not in n]
    head_params = list(model.model.fc.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    # --- Training ---
    iteration_losses = []   # (iteration_number, loss)
    global_iter = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for images, labels, mask in loader:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            global_iter += 1
            loss_val = loss.item()
            epoch_loss += loss_val
            iteration_losses.append((global_iter, loss_val))

        avg = epoch_loss / len(loader)
        print(f"Epoch [{epoch}/{args.epochs}]  avg_loss={avg:.4f}")

    # --- Save model ---
    torch.save({'model_state_dict': model.state_dict(),
                'num_classes': 4}, args.output_model)
    print(f"Model saved to {args.output_model}")

    # --- Plot loss curve ---
    iters, losses = zip(*iteration_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(iters, losses, linewidth=0.8, alpha=0.7, label='batch loss')

    # Smoothed curve
    window = max(1, len(losses) // 50)
    smoothed = [sum(losses[max(0, i-window):i+1]) / len(losses[max(0, i-window):i+1])
                for i in range(len(losses))]
    plt.plot(iters, smoothed, linewidth=2, color='red', label='smoothed')

    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.loss_plot, dpi=150)
    plt.close()
    print(f"Loss curve saved to {args.loss_plot}")


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(description='Train multilabel classifier')
    parser.add_argument('--images_dir', type=str, default='images',
                        help='Path to folder containing images')
    parser.add_argument('--labels', type=str, default='labels.txt',
                        help='Path to labels.txt')
    parser.add_argument('--output_model', type=str, default='model.pth',
                        help='Path to save model weights')
    parser.add_argument('--loss_plot', type=str, default='loss_curve.png',
                        help='Path to save the loss curve plot')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
