"""
inference.py - Run inference on a single image and print present attributes.

Usage:
    python inference.py --image path/to/image.jpg --model model.pth
    python inference.py --image path/to/image.jpg --model model.pth --threshold 0.4
"""

import argparse
import torch
from PIL import Image

from dataset import get_transforms
from model import MultilabelResNet

ATTR_NAMES = ['Attr1', 'Attr2', 'Attr3', 'Attr4']


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 4)
    model = MultilabelResNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(image_path, model, device, threshold=0.5):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)                    # (1, 4)
        probs = torch.sigmoid(logits).squeeze(0)  # (4,)

    present = [ATTR_NAMES[i] for i, p in enumerate(probs) if p.item() >= threshold]
    return present, probs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description='Multilabel inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to model weights')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for attribute presence (default: 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    present, probs = predict(args.image, model, device, threshold=args.threshold)

    print(f"\nImage: {args.image}")
    print(f"Threshold: {args.threshold}")
    print("\nAttribute probabilities:")
    for name, prob in zip(ATTR_NAMES, probs):
        status = '✓ PRESENT' if prob >= args.threshold else '✗ absent'
        print(f"  {name}: {prob:.4f}  {status}")

    print(f"\nAttributes present: {present if present else 'None'}")


if __name__ == '__main__':
    main()
