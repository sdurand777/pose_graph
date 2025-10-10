#!/usr/bin/env python3
"""
Extract DINOv2 features from images for loop closure detection.

DINOv2 is a self-supervised vision transformer that produces high-quality
visual features without task-specific training.
"""

import os
import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def load_dinov2_model(model_size='small'):
    """
    Load DINOv2 model from torch hub.

    Args:
        model_size: 'small', 'base', 'large', or 'giant'
                   small: fastest, 384-dim features
                   base: balanced, 768-dim features
                   large: high quality, 1024-dim features
                   giant: best quality, 1536-dim features (requires ~24GB GPU)

    Returns:
        model: DINOv2 model
        transform: Image preprocessing transform
    """
    model_names = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }

    model_name = model_names.get(model_size, 'dinov2_vits14')
    print(f"Loading DINOv2 model: {model_name}")

    # Load model from torch hub
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    # DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return model, transform, device


def extract_features(image_path, model, transform, device):
    """
    Extract DINOv2 features from a single image.

    Args:
        image_path: Path to image
        model: DINOv2 model
        transform: Preprocessing transform
        device: torch device

    Returns:
        feature: Normalized feature vector (numpy array)
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = model(img_tensor)

        # Convert to numpy and normalize
        feature = features.cpu().numpy().squeeze()
        feature = feature / np.linalg.norm(feature)  # L2 normalization

        return feature

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 features from images')
    parser.add_argument('--image-dir', type=str, default='scenario/imgs',
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='dinov2_features.pkl',
                       help='Output pickle file for features')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'base', 'large', 'giant'],
                       help='DINOv2 model size (small=fastest, giant=best quality)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for feature extraction (not implemented yet)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit number of images to process (for testing)')

    args = parser.parse_args()

    # Check image directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory '{image_dir}' not found")
        return

    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"Error: No images found in '{image_dir}'")
        return

    if args.max_images is not None:
        image_files = image_files[:args.max_images]

    print(f"Found {len(image_files)} images")

    # Load DINOv2 model
    model, transform, device = load_dinov2_model(args.model_size)

    # Extract features for all images
    features_dict = {}
    feature_dim = None

    print("Extracting features...")
    for img_path in tqdm(image_files, desc="Processing images"):
        feature = extract_features(img_path, model, transform, device)

        if feature is not None:
            features_dict[img_path.name] = feature
            if feature_dim is None:
                feature_dim = len(feature)

    print(f"\nExtracted features from {len(features_dict)} images")
    print(f"Feature dimension: {feature_dim}")

    # Save features
    output_data = {
        'features': features_dict,
        'model_size': args.model_size,
        'feature_dim': feature_dim,
        'image_dir': str(image_dir),
        'num_images': len(features_dict)
    }

    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Features saved to '{args.output}'")
    print(f"\nModel info:")
    print(f"  Size: {args.model_size}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Device: {device}")


if __name__ == '__main__':
    main()
