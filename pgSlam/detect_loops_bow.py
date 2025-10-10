#!/usr/bin/env python3
"""
Detect loop closures using Bag of Words (BoW) approach.
This script computes BoW histograms for each image and finds similar images
based on histogram similarity (cosine similarity or L1 distance).
"""
import cv2
import numpy as np
from pathlib import Path
import pickle
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

def load_vocabulary(vocab_path):
    """Load the pre-trained BoW vocabulary."""
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        kmeans = pickle.load(f)
    print(f"Vocabulary loaded: {kmeans.n_clusters} visual words")
    return kmeans

def compute_bow_histogram(descriptors, kmeans, normalize=True):
    """
    Compute BoW histogram for a set of descriptors.

    Args:
        descriptors: numpy array of ORB descriptors (N x 32)
        kmeans: Trained KMeans model
        normalize: Whether to normalize the histogram

    Returns:
        histogram: numpy array of visual word frequencies
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters)

    # Convert to float32
    descriptors_float = descriptors.astype(np.float32)

    # Predict cluster assignments
    labels = kmeans.predict(descriptors_float)

    # Build histogram
    histogram = np.bincount(labels, minlength=kmeans.n_clusters).astype(np.float32)

    # Normalize
    if normalize:
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm

    return histogram

def compute_all_histograms(image_paths, kmeans, features_per_image=500):
    """
    Compute BoW histograms for all images.

    Args:
        image_paths: List of image paths
        kmeans: Trained KMeans model
        features_per_image: Number of ORB features per image

    Returns:
        histograms: numpy array of shape (num_images, k)
    """
    orb = cv2.ORB_create(nfeatures=features_per_image)
    histograms = []

    print(f"\nComputing BoW histograms for {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        if idx % 50 == 0:
            print(f"Processing image {idx}/{len(image_paths)}...")

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            histograms.append(np.zeros(kmeans.n_clusters))
            continue

        # Detect and compute ORB features
        keypoints, descriptors = orb.detectAndCompute(img, None)

        # Compute histogram
        histogram = compute_bow_histogram(descriptors, kmeans)
        histograms.append(histogram)

    histograms = np.array(histograms)
    print(f"Computed {len(histograms)} histograms")

    return histograms

def cosine_similarity(hist1, hist2):
    """Compute cosine similarity between two histograms."""
    dot_product = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def detect_loop_closures(histograms, min_temporal_distance=30, similarity_threshold=0.7):
    """
    Detect loop closures based on BoW histogram similarity.

    Args:
        histograms: numpy array of histograms (num_images x k)
        min_temporal_distance: Minimum frame distance to consider as loop closure
        similarity_threshold: Minimum similarity score to consider as loop closure

    Returns:
        loop_closures: List of tuples (i, j, similarity_score) where i < j
    """
    print(f"\nDetecting loop closures...")
    print(f"Parameters: min_temporal_distance={min_temporal_distance}, threshold={similarity_threshold}")

    loop_closures = []
    num_images = len(histograms)

    for i in range(num_images):
        if i % 100 == 0:
            print(f"Processing image {i}/{num_images}...")

        # Only check images that are far enough in time
        for j in range(i + min_temporal_distance, num_images):
            # Compute similarity
            similarity = cosine_similarity(histograms[i], histograms[j])

            # Check if above threshold
            if similarity >= similarity_threshold:
                loop_closures.append((i, j, similarity))

    print(f"Found {len(loop_closures)} loop closures")

    return loop_closures

def save_loop_closures(loop_closures, output_path):
    """Save loop closures to file."""
    print(f"\nSaving loop closures to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("# Loop closures detected using BoW\n")
        f.write("# Format: image_i image_j similarity_score\n")
        for i, j, score in loop_closures:
            f.write(f"{i} {j} {score:.4f}\n")
    print(f"Saved {len(loop_closures)} loop closures")

def visualize_similarity_matrix(histograms, output_path=None, max_images=500):
    """
    Visualize the similarity matrix (useful for debugging).

    Args:
        histograms: numpy array of histograms
        output_path: Path to save the plot (optional)
        max_images: Maximum number of images to visualize
    """
    print("\nGenerating similarity matrix visualization...")

    # Limit size for visualization
    n = min(len(histograms), max_images)
    histograms_subset = histograms[:n]

    # Compute similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity(histograms_subset[i], histograms_subset[j])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.title(f'BoW Similarity Matrix (first {n} images)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Similarity matrix saved to {output_path}")
    else:
        plt.show()

def visualize_loop_pairs_interactive(loop_closures, image_paths, max_pairs=20, features_per_image=500):
    """
    Interactively display loop closure pairs with feature matches.
    Shows images side-by-side with matching features connected by lines.

    Args:
        loop_closures: List of tuples (i, j, similarity_score)
        image_paths: List of image paths
        max_pairs: Maximum number of pairs to visualize
        features_per_image: Number of ORB features per image
    """
    print(f"\nDisplaying loop closure pairs interactively...")
    print("Close window or press 'q' to see next pair, close all windows to quit")

    # Sort by similarity score (highest first)
    loop_closures_sorted = sorted(loop_closures, key=lambda x: x[2], reverse=True)

    # Visualize top pairs
    num_pairs = min(len(loop_closures_sorted), max_pairs)

    # Create ORB detector for feature matching
    orb = cv2.ORB_create(nfeatures=features_per_image)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for idx, (i, j, score) in enumerate(loop_closures_sorted[:num_pairs]):
        print(f"\n--- Pair {idx+1}/{num_pairs} ---")
        print(f"Image {i} <-> Image {j} | Similarity: {score:.4f}")

        # Read both images
        img1_color = cv2.imread(str(image_paths[i]))
        img2_color = cv2.imread(str(image_paths[j]))

        if img1_color is None or img2_color is None:
            print(f"Warning: Could not read images {i} or {j}")
            continue

        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for feature detection
        img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

        # Detect and compute ORB features
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        # Match features
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Keep only best matches (top 30%)
            num_good_matches = int(len(matches) * 0.3)
            good_matches = matches[:num_good_matches]

            print(f"Found {len(matches)} total matches, keeping {len(good_matches)} best matches")

            # Draw matches using OpenCV
            img_matches_bgr = cv2.drawMatches(
                img1_color, kp1,
                img2_color, kp2,
                good_matches,
                None,
                matchColor=(0, 255, 0),      # Green lines for matches
                singlePointColor=(255, 0, 0), # Red dots for keypoints
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Convert to RGB for matplotlib
            img_matches = cv2.cvtColor(img_matches_bgr, cv2.COLOR_BGR2RGB)

            # Display using matplotlib
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(img_matches)
            ax.axis('off')

            title = f"Loop Closure #{idx+1}/{num_pairs} | Image {i} <-> Image {j}\n"
            title += f"BoW Similarity: {score:.4f} | Feature Matches: {len(good_matches)}"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            plt.tight_layout()
            plt.show()

        else:
            print(f"Warning: Could not compute features for images {i} or {j}")

    print(f"\nDisplayed {num_pairs} loop closure pairs")

def main():
    parser = argparse.ArgumentParser(description='Detect loop closures using BoW')
    parser.add_argument('--images-dir', type=str, default='scenario/imgs',
                        help='Directory containing images')
    parser.add_argument('--vocabulary', type=str, default='bow_vocabulary.pkl',
                        help='BoW vocabulary file')
    parser.add_argument('--output', type=str, default='loop_closures_bow.txt',
                        help='Output file for loop closures')
    parser.add_argument('--min-temporal-distance', type=int, default=30,
                        help='Minimum temporal distance between frames (default: 30)')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                        help='Similarity threshold for loop detection (default: 0.7)')
    parser.add_argument('--features-per-image', type=int, default=500,
                        help='Number of ORB features per image (default: 500)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate similarity matrix visualization')
    parser.add_argument('--viz-output', type=str, default='similarity_matrix.png',
                        help='Output path for similarity matrix visualization')
    parser.add_argument('--show-pairs', action='store_true',
                        help='Display loop closure pairs interactively with feature matches')
    parser.add_argument('--max-pairs', type=int, default=20,
                        help='Maximum number of pairs to display (default: 20)')

    args = parser.parse_args()

    # Load vocabulary
    kmeans = load_vocabulary(args.vocabulary)

    # Get all images
    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("img*.jpg"))

    if len(image_paths) == 0:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images in {images_dir}")

    # Compute histograms for all images
    histograms = compute_all_histograms(image_paths, kmeans, args.features_per_image)

    # Detect loop closures
    loop_closures = detect_loop_closures(
        histograms,
        min_temporal_distance=args.min_temporal_distance,
        similarity_threshold=args.similarity_threshold
    )

    # Save results
    save_loop_closures(loop_closures, args.output)

    # Optional visualizations
    if args.visualize:
        visualize_similarity_matrix(histograms, args.viz_output)

    if args.show_pairs and len(loop_closures) > 0:
        visualize_loop_pairs_interactive(loop_closures, image_paths, args.max_pairs, args.features_per_image)

    # Print statistics
    print("\n" + "="*60)
    print("Loop closure detection complete!")
    print(f"Total images: {len(image_paths)}")
    print(f"Loop closures found: {len(loop_closures)}")
    print(f"Output file: {args.output}")
    if loop_closures:
        avg_similarity = np.mean([score for _, _, score in loop_closures])
        print(f"Average similarity score: {avg_similarity:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
