#!/usr/bin/env python3
"""
Visualize DINOv2 loop closures on trajectory.

This script reads detected loop closures and displays them overlayed
on the robot trajectory.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_scenario_poses(scenario_file):
    """
    Load poses from scenario file.

    Args:
        scenario_file: Path to scenario txt file

    Returns:
        poses: List of (x, y, z) tuples
    """
    poses = []
    with open(scenario_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                poses.append((x, y, z))

    return poses


def load_keyframe_mapping(mapping_file):
    """
    Load keyframe ID to image index mapping.

    Args:
        mapping_file: Path to id_kf_2_idx_img.txt

    Returns:
        kf_to_img: Dictionary mapping keyframe ID to image index
    """
    kf_to_img = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                kf_id = int(parts[0])
                img_idx = int(parts[1])
                kf_to_img[kf_id] = img_idx

    return kf_to_img


def load_loop_closures_dinov2(loop_file):
    """
    Load DINOv2 loop closures from file.

    Args:
        loop_file: Path to loop_closures_dinov2.txt

    Returns:
        loop_pairs: List of (img_idx_i, img_idx_j, similarity) tuples
    """
    loop_pairs = []
    with open(loop_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                idx_i = int(parts[0])
                idx_j = int(parts[1])
                similarity = float(parts[2])
                loop_pairs.append((idx_i, idx_j, similarity))

    return loop_pairs


def map_image_to_pose_idx(img_idx, kf_to_img):
    """
    Map image index to pose index using keyframe mapping.

    Args:
        img_idx: Image index
        kf_to_img: Dictionary mapping keyframe ID to image index

    Returns:
        pose_idx: Pose index (keyframe ID) or None if not found
    """
    for kf_id, mapped_img_idx in kf_to_img.items():
        if mapped_img_idx == img_idx:
            return kf_id
    return None


def visualize_loops_on_trajectory_2d(poses, loop_pairs, kf_to_img=None,
                                     max_loops=None, output_file=None):
    """
    Visualize loop closures overlayed on 2D trajectory (top view).

    Args:
        poses: List of (x, y, z) tuples
        loop_pairs: List of (img_idx_i, img_idx_j, similarity) tuples
        kf_to_img: Optional keyframe mapping (if None, assume img_idx == pose_idx)
        max_loops: Maximum number of loops to display
        output_file: Optional output filename
    """
    if not poses:
        print("Error: No poses to visualize")
        return

    # Extract coordinates
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot trajectory
    ax.plot(xs, ys, 'b-', linewidth=1.5, label='Trajectory', alpha=0.7)
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
    ax.plot(xs[-1], ys[-1], 'ro', markersize=10, label='End')

    # Plot loop closures
    n_loops_drawn = 0
    if max_loops is not None:
        loop_pairs = loop_pairs[:max_loops]

    for img_idx_i, img_idx_j, similarity in loop_pairs:
        # Map image indices to pose indices
        if kf_to_img is not None:
            pose_idx_i = map_image_to_pose_idx(img_idx_i, kf_to_img)
            pose_idx_j = map_image_to_pose_idx(img_idx_j, kf_to_img)
        else:
            # Assume direct mapping
            pose_idx_i = img_idx_i
            pose_idx_j = img_idx_j

        # Check if poses exist
        if (pose_idx_i is not None and pose_idx_j is not None and
            0 <= pose_idx_i < len(poses) and 0 <= pose_idx_j < len(poses)):

            x_i, y_i = poses[pose_idx_i][0], poses[pose_idx_i][1]
            x_j, y_j = poses[pose_idx_j][0], poses[pose_idx_j][1]

            # Draw loop closure line
            label = 'Loop closures' if n_loops_drawn == 0 else None
            ax.plot([x_i, x_j], [y_i, y_j], 'c--', linewidth=0.8,
                   alpha=0.6, label=label)

            n_loops_drawn += 1

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Trajectory with DINOv2 Loop Closures ({n_loops_drawn} loops)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to '{output_file}'")

    plt.show()


def visualize_loops_on_trajectory_3d(poses, loop_pairs, kf_to_img=None,
                                     max_loops=None, output_file=None):
    """
    Visualize loop closures overlayed on 3D trajectory.

    Args:
        poses: List of (x, y, z) tuples
        loop_pairs: List of (img_idx_i, img_idx_j, similarity) tuples
        kf_to_img: Optional keyframe mapping (if None, assume img_idx == pose_idx)
        max_loops: Maximum number of loops to display
        output_file: Optional output filename
    """
    if not poses:
        print("Error: No poses to visualize")
        return

    # Extract coordinates
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    zs = [p[2] for p in poses]

    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(xs, ys, zs, 'b-', linewidth=1.5, label='Trajectory', alpha=0.7)
    ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=10, label='Start')
    ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=10, label='End')

    # Plot loop closures
    n_loops_drawn = 0
    if max_loops is not None:
        loop_pairs = loop_pairs[:max_loops]

    for img_idx_i, img_idx_j, similarity in loop_pairs:
        # Map image indices to pose indices
        if kf_to_img is not None:
            pose_idx_i = map_image_to_pose_idx(img_idx_i, kf_to_img)
            pose_idx_j = map_image_to_pose_idx(img_idx_j, kf_to_img)
        else:
            # Assume direct mapping
            pose_idx_i = img_idx_i
            pose_idx_j = img_idx_j

        # Check if poses exist
        if (pose_idx_i is not None and pose_idx_j is not None and
            0 <= pose_idx_i < len(poses) and 0 <= pose_idx_j < len(poses)):

            x_i, y_i, z_i = poses[pose_idx_i]
            x_j, y_j, z_j = poses[pose_idx_j]

            # Draw loop closure line
            label = 'Loop closures' if n_loops_drawn == 0 else None
            ax.plot([x_i, x_j], [y_i, y_j], [z_i, z_j], 'c--',
                   linewidth=0.8, alpha=0.6, label=label)

            n_loops_drawn += 1

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Trajectory with DINOv2 Loop Closures ({n_loops_drawn} loops)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to '{output_file}'")

    plt.show()


def plot_loop_closure_statistics(loop_pairs):
    """
    Plot statistics about detected loop closures.

    Args:
        loop_pairs: List of (img_idx_i, img_idx_j, similarity) tuples
    """
    if not loop_pairs:
        print("No loop closures to analyze")
        return

    # Extract data
    similarities = [sim for _, _, sim in loop_pairs]
    temporal_gaps = [j - i for i, j, _ in loop_pairs]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Similarity distribution
    axes[0, 0].hist(similarities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Similarity Score', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Similarity Scores', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Temporal gap distribution
    axes[0, 1].hist(temporal_gaps, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Temporal Gap (frames)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Temporal Gaps', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Similarity vs temporal gap scatter
    axes[1, 0].scatter(temporal_gaps, similarities, alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Temporal Gap (frames)', fontsize=10)
    axes[1, 0].set_ylabel('Similarity Score', fontsize=10)
    axes[1, 0].set_title('Similarity vs Temporal Gap', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
    Loop Closure Statistics
    {'='*40}

    Total loop closures: {len(loop_pairs)}

    Similarity:
      Mean:   {np.mean(similarities):.4f}
      Median: {np.median(similarities):.4f}
      Min:    {np.min(similarities):.4f}
      Max:    {np.max(similarities):.4f}
      Std:    {np.std(similarities):.4f}

    Temporal Gap:
      Mean:   {np.mean(temporal_gaps):.1f} frames
      Median: {np.median(temporal_gaps):.1f} frames
      Min:    {np.min(temporal_gaps)} frames
      Max:    {np.max(temporal_gaps)} frames
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('dinov2_loop_statistics.png', dpi=150, bbox_inches='tight')
    print("Loop closure statistics saved to 'dinov2_loop_statistics.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize DINOv2 loop closures on trajectory')
    parser.add_argument('--loop-file', type=str, default='loop_closures_dinov2.txt',
                       help='Input file with detected loop closures')
    parser.add_argument('--scenario-file', type=str, default='scenario/scenario_noisy.txt',
                       help='Scenario file with poses')
    parser.add_argument('--mapping-file', type=str, default='scenario/id_kf_2_idx_img.txt',
                       help='Keyframe to image mapping file')
    parser.add_argument('--max-loops', type=int, default=None,
                       help='Maximum number of loops to display')
    parser.add_argument('--no-mapping', action='store_true',
                       help='Assume direct mapping (img_idx == pose_idx)')
    parser.add_argument('--3d', action='store_true', dest='show_3d',
                       help='Show 3D visualization')
    parser.add_argument('--statistics', action='store_true',
                       help='Show loop closure statistics')

    args = parser.parse_args()

    # Load loop closures
    print(f"Loading loop closures from '{args.loop_file}'...")
    try:
        loop_pairs = load_loop_closures_dinov2(args.loop_file)
    except FileNotFoundError:
        print(f"Error: Loop file '{args.loop_file}' not found")
        print("Run 'python detect_loops_dinov2.py' first")
        return

    print(f"Loaded {len(loop_pairs)} loop closures")

    if len(loop_pairs) == 0:
        print("No loop closures to visualize")
        return

    # Load poses
    print(f"Loading poses from '{args.scenario_file}'...")
    try:
        poses = load_scenario_poses(args.scenario_file)
    except FileNotFoundError:
        print(f"Error: Scenario file '{args.scenario_file}' not found")
        return

    print(f"Loaded {len(poses)} poses")

    # Load keyframe mapping (if needed)
    kf_to_img = None
    if not args.no_mapping:
        if Path(args.mapping_file).exists():
            print(f"Loading keyframe mapping from '{args.mapping_file}'...")
            kf_to_img = load_keyframe_mapping(args.mapping_file)
            print(f"Loaded mapping for {len(kf_to_img)} keyframes")
        else:
            print(f"Warning: Mapping file '{args.mapping_file}' not found")
            print("Using direct mapping (img_idx == pose_idx)")

    # Show statistics
    if args.statistics:
        plot_loop_closure_statistics(loop_pairs)

    # Visualize
    if args.show_3d:
        visualize_loops_on_trajectory_3d(poses, loop_pairs, kf_to_img,
                                        args.max_loops,
                                        'dinov2_loops_3d.png')
    else:
        visualize_loops_on_trajectory_2d(poses, loop_pairs, kf_to_img,
                                        args.max_loops,
                                        'dinov2_loops_2d.png')


if __name__ == '__main__':
    main()
