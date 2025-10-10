#!/usr/bin/env python3
"""
Build a Bag of Words vocabulary from images using ORB features and K-Means clustering.
This creates a visual vocabulary that can be used for place recognition and loop closure detection.
"""
import cv2
import numpy as np
from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans
import argparse

def extract_orb_features(image_paths, max_images=None, features_per_image=500):
    """
    Extract ORB features from a set of images.

    Args:
        image_paths: List of image file paths
        max_images: Maximum number of images to process (None = all)
        features_per_image: Number of ORB features to extract per image

    Returns:
        all_descriptors: numpy array of all descriptors (N x 32)
    """
    orb = cv2.ORB_create(nfeatures=features_per_image)
    all_descriptors = []

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Extracting ORB features from {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"Processing image {idx}/{len(image_paths)}...")

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Detect and compute ORB features
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if descriptors is not None:
            all_descriptors.append(descriptors)

    # Concatenate all descriptors
    all_descriptors = np.vstack(all_descriptors)
    print(f"Extracted {len(all_descriptors)} descriptors in total")

    return all_descriptors

def build_vocabulary(descriptors, k=1000, batch_size=1000):
    """
    Build visual vocabulary using K-Means clustering.

    Args:
        descriptors: numpy array of descriptors (N x 32)
        k: Number of visual words (cluster centers)
        batch_size: Batch size for MiniBatchKMeans

    Returns:
        kmeans: Trained KMeans model
    """
    print(f"\nBuilding vocabulary with k={k} visual words...")
    print("This may take a few minutes...")

    # Use MiniBatchKMeans for efficiency with large datasets
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        verbose=1,
        random_state=42,
        max_iter=100
    )

    # Convert to float32 for sklearn
    descriptors_float = descriptors.astype(np.float32)

    # Fit the model
    kmeans.fit(descriptors_float)

    print(f"Vocabulary built with {k} visual words")

    return kmeans

def save_vocabulary(kmeans, output_path):
    """Save the vocabulary model to disk."""
    print(f"\nSaving vocabulary to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print("Vocabulary saved successfully")

def main():
    parser = argparse.ArgumentParser(description='Build BoW vocabulary from images')
    parser.add_argument('--images-dir', type=str, default='scenario/imgs',
                        help='Directory containing images')
    parser.add_argument('--output', type=str, default='bow_vocabulary.pkl',
                        help='Output vocabulary file')
    parser.add_argument('--k', type=int, default=1000,
                        help='Number of visual words (default: 1000)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to use (default: all)')
    parser.add_argument('--features-per-image', type=int, default=500,
                        help='Number of ORB features per image (default: 500)')

    args = parser.parse_args()

    # Get all images
    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("img*.jpg"))

    if len(image_paths) == 0:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images in {images_dir}")

    # Extract features
    descriptors = extract_orb_features(
        image_paths,
        max_images=args.max_images,
        features_per_image=args.features_per_image
    )

    # Build vocabulary
    kmeans = build_vocabulary(descriptors, k=args.k)

    # Save vocabulary
    save_vocabulary(kmeans, args.output)

    print("\n" + "="*60)
    print("BoW vocabulary construction complete!")
    print(f"Vocabulary file: {args.output}")
    print(f"Number of visual words: {args.k}")
    print(f"Total descriptors used: {len(descriptors)}")
    print("="*60)

if __name__ == "__main__":
    main()
