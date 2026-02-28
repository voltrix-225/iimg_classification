"""
model.py - ResNet50-based multilabel classification model (fine-tuned from ImageNet weights).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultilabelResNet(nn.Module):
    """
    ResNet50 fine-tuned for multilabel classification.
    Final FC layer replaced with a 4-output head.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace the final fully-connected layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)  # raw logits (no sigmoid); use BCEWithLogitsLoss
