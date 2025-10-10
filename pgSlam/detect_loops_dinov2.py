#!/usr/bin/env python3
"""
Detect loop closures using DINOv2 features with cosine similarity.

This script loads pre-extracted DINOv2 features and finds similar image pairs
based on feature similarity.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def compute_similarity_matrix(features_dict):
    """
    Compute pairwise cosine similarity matrix for all images.

    Args:
        features_dict: Dictionary mapping image names to feature vectors

    Returns:
        similarity_matrix: N x N matrix of cosine similarities
        image_names: List of image names in order
    """
    image_names = sorted(features_dict.keys())
    n_images = len(image_names)

    # Stack features into matrix (N x D)
    features_matrix = np.stack([features_dict[name] for name in image_names])

    # Compute cosine similarity (features are already normalized)
    similarity_matrix = features_matrix @ features_matrix.T

    return similarity_matrix, image_names


def detect_loop_closures(similarity_matrix, image_names,
                         similarity_threshold=0.85,
                         min_temporal_distance=30):
    """
    Detect loop closures from similarity matrix.

    Args:
        similarity_matrix: N x N cosine similarity matrix
        image_names: List of image names
        similarity_threshold: Minimum similarity for loop closure
        min_temporal_distance: Minimum frame separation

    Returns:
        loop_pairs: List of (idx_i, idx_j, similarity) tuples
    """
    n_images = len(image_names)
    loop_pairs = []

    print("Detecting loop closures...")
    for i in tqdm(range(n_images)):
        for j in range(i + min_temporal_distance, n_images):
            similarity = similarity_matrix[i, j]

            if similarity >= similarity_threshold:
                loop_pairs.append((i, j, similarity))

    # Sort by similarity (descending)
    loop_pairs.sort(key=lambda x: x[2], reverse=True)

    return loop_pairs


def visualize_similarity_matrix(similarity_matrix, image_names,
                               min_temporal_distance=30,
                               output_file='dinov2_similarity_matrix.png'):
    """
    Visualize the similarity matrix as a heatmap.

    Args:
        similarity_matrix: N x N similarity matrix
        image_names: List of image names
        min_temporal_distance: Minimum temporal distance (for mask)
        output_file: Output filename
    """
    n_images = len(image_names)

    # Create mask for temporal constraint
    mask = np.ones_like(similarity_matrix, dtype=bool)
    for i in range(n_images):
        for j in range(i, min(i + min_temporal_distance, n_images)):
            mask[i, j] = False
            mask[j, i] = False

    # Apply mask
    masked_sim = similarity_matrix.copy()
    masked_sim[~mask] = 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Full similarity matrix
    im1 = axes[0].imshow(similarity_matrix, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Full Similarity Matrix', fontsize=14)
    axes[0].set_xlabel('Image Index')
    axes[0].set_ylabel('Image Index')
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')

    # Masked similarity matrix (valid loop closures only)
    im2 = axes[1].imshow(masked_sim, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Valid Loop Closures (min_dist={min_temporal_distance})', fontsize=14)
    axes[1].set_xlabel('Image Index')
    axes[1].set_ylabel('Image Index')
    plt.colorbar(im2, ax=axes[1], label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Similarity matrix saved to '{output_file}'")
    plt.show()


def display_image_pair(image_dir, image_names, idx_i, idx_j, similarity):
    """
    Display a pair of images that form a loop closure.

    Args:
        image_dir: Directory containing images
        image_names: List of image names
        idx_i: Index of first image
        idx_j: Index of second image
        similarity: Similarity score
    """
    img_path_i = Path(image_dir) / image_names[idx_i]
    img_path_j = Path(image_dir) / image_names[idx_j]

    img_i = cv2.imread(str(img_path_i))
    img_j = cv2.imread(str(img_path_j))

    if img_i is None or img_j is None:
        print(f"Error loading images: {img_path_i}, {img_path_j}")
        return

    # Convert BGR to RGB
    img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
    img_j = cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img_i)
    axes[0].set_title(f'Image {idx_i}: {image_names[idx_i]}', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(img_j)
    axes[1].set_title(f'Image {idx_j}: {image_names[idx_j]}', fontsize=12)
    axes[1].axis('off')

    fig.suptitle(f'Loop Closure Pair - Similarity: {similarity:.4f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def save_loop_closures(loop_pairs, image_names, output_file='loop_closures_dinov2.txt'):
    """
    Save loop closure pairs to text file.

    Args:
        loop_pairs: List of (idx_i, idx_j, similarity) tuples
        image_names: List of image names
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        f.write(f"# Loop closures detected with DINOv2\n")
        f.write(f"# Format: idx_i idx_j similarity image_name_i image_name_j\n")
        f.write(f"# Total: {len(loop_pairs)} pairs\n\n")

        for idx_i, idx_j, similarity in loop_pairs:
            f.write(f"{idx_i} {idx_j} {similarity:.6f} {image_names[idx_i]} {image_names[idx_j]}\n")

    print(f"Loop closures saved to '{output_file}'")


def plot_similarity_distribution(similarity_matrix, min_temporal_distance=30):
    """
    Plot histogram of similarity scores.

    Args:
        similarity_matrix: N x N similarity matrix
        min_temporal_distance: Minimum temporal distance for valid pairs
    """
    n_images = similarity_matrix.shape[0]

    # Extract valid similarities (upper triangle with temporal constraint)
    valid_similarities = []
    for i in range(n_images):
        for j in range(i + min_temporal_distance, n_images):
            valid_similarities.append(similarity_matrix[i, j])

    valid_similarities = np.array(valid_similarities)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(valid_similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Similarity Scores (min_dist={min_temporal_distance})',
              fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_sim = np.mean(valid_similarities)
    median_sim = np.median(valid_similarities)
    std_sim = np.std(valid_similarities)
    max_sim = np.max(valid_similarities)

    stats_text = f'Mean: {mean_sim:.4f}\nMedian: {median_sim:.4f}\nStd: {std_sim:.4f}\nMax: {max_sim:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('dinov2_similarity_distribution.png', dpi=150, bbox_inches='tight')
    print("Similarity distribution saved to 'dinov2_similarity_distribution.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detect loop closures using DINOv2 features')
    parser.add_argument('--features', type=str, default='dinov2_features.pkl',
                       help='Input pickle file with DINOv2 features')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                       help='Minimum cosine similarity for loop closure (0-1)')
    parser.add_argument('--min-temporal-distance', type=int, default=30,
                       help='Minimum frame separation for loop closure')
    parser.add_argument('--output', type=str, default='loop_closures_dinov2.txt',
                       help='Output file for detected loop closures')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize similarity matrix')
    parser.add_argument('--show-pairs', action='store_true',
                       help='Display detected loop closure pairs interactively')
    parser.add_argument('--max-pairs', type=int, default=10,
                       help='Maximum number of pairs to display')
    parser.add_argument('--plot-distribution', action='store_true',
                       help='Plot similarity score distribution')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit detection to first N images (e.g., 400)')

    args = parser.parse_args()

    # Load features
    print(f"Loading features from '{args.features}'...")
    try:
        with open(args.features, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Features file '{args.features}' not found")
        print("Run 'python build_dinov2_features.py' first to extract features")
        return

    features_dict = data['features']
    model_size = data.get('model_size', 'unknown')
    feature_dim = data.get('feature_dim', 'unknown')
    image_dir = data.get('image_dir', 'scenario/imgs')

    print(f"Loaded features for {len(features_dict)} images")

    # Limit to first N images if specified
    if args.max_images is not None and args.max_images < len(features_dict):
        sorted_names = sorted(features_dict.keys())
        selected_names = sorted_names[:args.max_images]
        features_dict = {name: features_dict[name] for name in selected_names}
        print(f"Limited to first {args.max_images} images")

    print(f"Using {len(features_dict)} images for loop detection")
    print(f"Model size: {model_size}, Feature dimension: {feature_dim}")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix, image_names = compute_similarity_matrix(features_dict)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Plot similarity distribution
    if args.plot_distribution:
        plot_similarity_distribution(similarity_matrix, args.min_temporal_distance)

    # Visualize similarity matrix
    if args.visualize:
        visualize_similarity_matrix(similarity_matrix, image_names,
                                   args.min_temporal_distance)

    # Detect loop closures
    loop_pairs = detect_loop_closures(
        similarity_matrix, image_names,
        similarity_threshold=args.similarity_threshold,
        min_temporal_distance=args.min_temporal_distance
    )

    print(f"\nDetected {len(loop_pairs)} loop closures")
    if len(loop_pairs) > 0:
        print(f"Similarity range: [{loop_pairs[-1][2]:.4f}, {loop_pairs[0][2]:.4f}]")
        print(f"Top 5 pairs:")
        for i, (idx_i, idx_j, sim) in enumerate(loop_pairs[:5]):
            print(f"  {i+1}. Image {idx_i} ↔ Image {idx_j}: {sim:.4f}")

    # Save loop closures
    save_loop_closures(loop_pairs, image_names, args.output)

    # Display pairs interactively
    if args.show_pairs and len(loop_pairs) > 0:
        print(f"\nDisplaying top {min(args.max_pairs, len(loop_pairs))} pairs...")
        for i in range(min(args.max_pairs, len(loop_pairs))):
            idx_i, idx_j, similarity = loop_pairs[i]
            print(f"\nPair {i+1}/{min(args.max_pairs, len(loop_pairs))}")
            display_image_pair(image_dir, image_names, idx_i, idx_j, similarity)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_names)}")
    if args.max_images is not None:
        print(f"Limited to: {args.max_images} images")
    print(f"Model size: {model_size}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Min temporal distance: {args.min_temporal_distance}")
    print(f"Detected loop closures: {len(loop_pairs)}")
    print(f"Output saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
