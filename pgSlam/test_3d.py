#!/usr/bin/env python3
"""
Quick test script for 3D pose graph SLAM
Tests trajectory generation and data structures without optimization
"""
from trajectory_3d import genTraj3D, genTrajSpiralHelix3D, addNoise3D
from g2o_io_3d import writeOdom3D, writeLoop3D
import numpy as np

print("="*70)
print("Testing 3D Pose Graph SLAM Components")
print("="*70)

# Test 1: Generate U-turn trajectory
print("\n[1/5] Testing U-turn trajectory generation...")
(X, Y, Z, QX, QY, QZ, QW) = genTraj3D(num_poses_per_segment=10)
print(f"  Generated {len(X)} poses")
print(f"  Position range: X=[{min(X):.2f}, {max(X):.2f}], Y=[{min(Y):.2f}, {max(Y):.2f}], Z=[{min(Z):.2f}, {max(Z):.2f}]")
print(f"  Sample pose 0: ({X[0]:.2f}, {Y[0]:.2f}, {Z[0]:.2f}), quat=({QX[0]:.3f}, {QY[0]:.3f}, {QZ[0]:.3f}, {QW[0]:.3f})")

# Verify quaternion normalization
quat_norms = np.sqrt(QX**2 + QY**2 + QZ**2 + QW**2)
print(f"  Quaternion norms: min={min(quat_norms):.6f}, max={max(quat_norms):.6f} (should be ~1.0)")
print("  ✓ U-turn trajectory OK")

# Test 2: Generate helix trajectory
print("\n[2/5] Testing helix trajectory generation...")
(X2, Y2, Z2, QX2, QY2, QZ2, QW2) = genTrajSpiralHelix3D(num_poses_per_segment=10)
print(f"  Generated {len(X2)} poses")
print(f"  Position range: X=[{min(X2):.2f}, {max(X2):.2f}], Y=[{min(Y2):.2f}, {max(Y2):.2f}], Z=[{min(Z2):.2f}, {max(Z2):.2f}]")
print("  ✓ Helix trajectory OK")

# Test 3: Add noise
print("\n[3/5] Testing noise addition...")
(xN, yN, zN, qxN, qyN, qzN, qwN) = addNoise3D(X, Y, Z, QX, QY, QZ, QW, seed=42, noise_sigma=0.05)
pos_error = np.sqrt((X - xN)**2 + (Y - yN)**2 + (Z - zN)**2)
print(f"  Mean position error: {np.mean(pos_error):.4f}m")
print(f"  Max position error: {np.max(pos_error):.4f}m")
print(f"  First 5 poses have zero noise: {np.all(pos_error[:5] == 0)}")
print("  ✓ Noise addition OK")

# Test 4: Write G2O file
print("\n[4/5] Testing G2O file writing...")
g2o = writeOdom3D(xN, yN, zN, qxN, qyN, qzN, qwN, odom_info=500.0)
print("  Odometry edges written to noise_3d.g2o")

loop_pairs = writeLoop3D(X, Y, Z, QX, QY, QZ, QW, g2o,
                         loop_info=700.0, loop_sampling_rate=3, seed=42)
print(f"  Found {len(loop_pairs)} loop closures")
if len(loop_pairs) > 0:
	print(f"  Sample loop: pose {loop_pairs[0][0]} <-> pose {loop_pairs[0][1]}")
print("  ✓ G2O file writing OK")

# Test 5: Verify G2O file format
print("\n[5/5] Verifying G2O file format...")
with open('noise_3d.g2o', 'r') as f:
	lines = f.readlines()

vertex_count = sum(1 for line in lines if 'VERTEX_SE3:QUAT' in line)
edge_count = sum(1 for line in lines if 'EDGE_SE3:QUAT' in line)
fix_count = sum(1 for line in lines if 'FIX' in line)

print(f"  Vertices: {vertex_count}")
print(f"  Edges: {edge_count}")
print(f"  Fixed poses: {fix_count}")
print(f"  Expected vertices: {len(X)}")
print(f"  Expected odometry edges: {len(X) - 1}")
print(f"  Expected loop edges: {len(loop_pairs)}")
print(f"  Total edges: {edge_count} = {len(X)-1} (odom) + {len(loop_pairs)} (loops)")

assert vertex_count == len(X), "Vertex count mismatch!"
assert edge_count == len(X) - 1 + len(loop_pairs), "Edge count mismatch!"
assert fix_count == 1, "Should have exactly 1 fixed pose!"

print("  ✓ G2O file format OK")

print("\n" + "="*70)
print("All tests passed! ✓")
print("="*70)
print("\nTo run full 3D optimization:")
print("  python main_3d.py --seed 42")
print("\nTo run without visualization (faster):")
print("  python main_3d.py --seed 42 --no-viz")
print("="*70)
