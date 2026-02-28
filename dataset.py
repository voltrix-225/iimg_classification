"""
dataset.py - Dataset loading and preprocessing for multilabel classification.
Handles NA values by using masked loss during training.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def parse_labels(labels_path):
    """
    Parse labels.txt file. Returns a DataFrame with image names and attributes.
    NA values are preserved as NaN for masked loss computation.
    """
    df = pd.read_csv(labels_path, sep=r'\s+', header=0)
    df.columns = ['image_name', 'Attr1', 'Attr2', 'Attr3', 'Attr4']
    # Convert to numeric, NA becomes NaN
    for col in ['Attr1', 'Attr2', 'Attr3', 'Attr4']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_pos_weights(df, attr_cols):
    """
    Compute positive weights for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = num_negatives / num_positives for each attribute.
    NA values are excluded from the computation.
    """
    pos_weights = []
    for col in attr_cols:
        valid = df[col].dropna()
        n_pos = (valid == 1).sum()
        n_neg = (valid == 0).sum()
        if n_pos == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(n_neg / n_pos)
    return torch.tensor(pos_weights, dtype=torch.float32)


def get_transforms(train=True, img_size=224):
    """Return image transforms for train or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class MultilabelDataset(Dataset):
    """
    Dataset for multilabel classification.
    Returns (image, labels, mask) where mask=0 for NA entries, 1 otherwise.
    """

    ATTR_COLS = ['Attr1', 'Attr2', 'Attr3', 'Attr4']

    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        raw = row[self.ATTR_COLS].values  # shape (4,), may contain NaN
        mask = (~pd.isna(raw)).astype(np.float32)
        labels = np.where(pd.isna(raw), 0.0, raw.astype(np.float32))

        return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
