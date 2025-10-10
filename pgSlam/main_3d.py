"""
Main script for 3D pose graph SLAM demonstration
"""
import argparse
from trajectory_3d import genTraj3D, genTrajSpiralHelix3D, addNoise3D
from visualization_3d import draw3D, drawTwo3D, drawThree3D, plotErrors3D
from g2o_io_3d import writeOdom3D, writeLoop3D, readG2o3D
from pose_graph_3d import optimize3D
from metrics_3d import compute_pose_errors_3d, print_metrics_comparison_3d


def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='3D Pose Graph SLAM Optimization Demo')
	parser.add_argument('--trajectory-type', type=str, default='uturn', choices=['uturn', 'helix'],
	                    help='Type of trajectory to generate (default: uturn)')
	parser.add_argument('--odom-info', type=float, default=500.0,
	                    help='Information matrix value for odometry edges (default: 500.0)')
	parser.add_argument('--loop-info', type=float, default=700.0,
	                    help='Information matrix value for loop closure edges (default: 700.0)')
	parser.add_argument('--num-poses', type=int, default=20,
	                    help='Number of poses per trajectory segment (default: 20)')
	parser.add_argument('--loop-sampling', type=int, default=2,
	                    help='Sample every Nth pose for loop closures (default: 2). Higher = fewer loops. 0 = no loops')
	parser.add_argument('--seed', type=int, default=None,
	                    help='Random seed for noise generation (default: None for random noise)')
	parser.add_argument('--noise-sigma', type=float, default=0.03,
	                    help='Standard deviation of Gaussian noise for odometry (default: 0.03)')
	parser.add_argument('--loop-from-noisy', action='store_true',
	                    help='Use noisy poses instead of reference for loop closure edges (default: use reference)')
	parser.add_argument('--loop-noise-sigma', type=float, default=0.0,
	                    help='Standard deviation of Gaussian noise for loop closure observations (default: 0.0)')
	parser.add_argument('--no-viz', action='store_true',
	                    help='Disable visualization')
	args = parser.parse_args()

	print("="*70)
	print("3D POSE GRAPH SLAM OPTIMIZATION")
	print("="*70)
	print(f"  Trajectory type: {args.trajectory_type}")
	print(f"  Odometry info: {args.odom_info}")
	print(f"  Loop closure info: {args.loop_info}")
	print(f"  Poses per segment: {args.num_poses}")
	print(f"  Loop sampling rate: {args.loop_sampling}")
	print(f"  Odometry noise sigma: {args.noise_sigma}")
	print(f"  Loop from noisy poses: {args.loop_from_noisy}")
	print(f"  Loop noise sigma: {args.loop_noise_sigma}")
	print(f"  Random seed: {args.seed if args.seed is not None else 'None (random)'}")
	print("="*70)

	# Generate ground truth trajectory based on type
	if args.trajectory_type == 'uturn':
		(X, Y, Z, QX, QY, QZ, QW) = genTraj3D(num_poses_per_segment=args.num_poses)
		print(f"  Total poses: ~{args.num_poses * 6}")
	elif args.trajectory_type == 'helix':
		(X, Y, Z, QX, QY, QZ, QW) = genTrajSpiralHelix3D(num_poses_per_segment=args.num_poses)
		print(f"  Total poses: {len(X)}")

	if not args.no_viz:
		draw3D(X, Y, Z, QX, QY, QZ, QW, title="Ground Truth Trajectory (3D)")

	# Add noise to trajectory
	(xN, yN, zN, qxN, qyN, qzN, qwN) = addNoise3D(X, Y, Z, QX, QY, QZ, QW,
	                                              seed=args.seed, noise_sigma=args.noise_sigma)
	g2o = writeOdom3D(xN, yN, zN, qxN, qyN, qzN, qwN, odom_info=args.odom_info)

	# Choose which poses to use for loop closures
	if args.loop_from_noisy:
		loop_X, loop_Y, loop_Z = xN, yN, zN
		loop_QX, loop_QY, loop_QZ, loop_QW = qxN, qyN, qzN, qwN
	else:
		loop_X, loop_Y, loop_Z = X, Y, Z
		loop_QX, loop_QY, loop_QZ, loop_QW = QX, QY, QZ, QW

	loop_pairs = writeLoop3D(loop_X, loop_Y, loop_Z, loop_QX, loop_QY, loop_QZ, loop_QW, g2o,
	                         loop_info=args.loop_info,
	                         loop_sampling_rate=args.loop_sampling,
	                         loop_noise_sigma=args.loop_noise_sigma,
	                         seed=args.seed)

	print(f"\n  Number of loop closures: {len(loop_pairs)}")
	if len(loop_pairs) > 0:
		print(f"  Loop closure connections:")
		for i, (start, end) in enumerate(loop_pairs):
			if i < 5:  # Show first 5 loop closures
				print(f"    Pose {start} <-> Pose {end}")
		if len(loop_pairs) > 5:
			print(f"    ... and {len(loop_pairs) - 5} more loop closures")

	if not args.no_viz:
		draw3D(xN, yN, zN, qxN, qyN, qzN, qwN, loop_pairs=loop_pairs,
		       title="Noisy Trajectory with Loop Closures (3D)")

	# Optimize pose graph
	print("\n  Running g2o optimizer...")
	optimize3D()
	(xOpt, yOpt, zOpt, qxOpt, qyOpt, qzOpt, qwOpt) = readG2o3D("opt_3d.g2o")

	if not args.no_viz:
		drawTwo3D(X, Y, Z, QX, QY, QZ, QW,
		          xOpt, yOpt, zOpt, qxOpt, qyOpt, qzOpt, qwOpt,
		          loop_pairs=loop_pairs)

	# Show all three trajectories
	if not args.no_viz:
		drawThree3D(X, Y, Z, QX, QY, QZ, QW,
		            xOpt, yOpt, zOpt, qxOpt, qyOpt, qzOpt, qwOpt,
		            xN, yN, zN, qxN, qyN, qzN, qwN,
		            loop_pairs=loop_pairs)

	print("\n  Optimization complete!")

	# Compute and display metrics
	noisy_metrics = compute_pose_errors_3d(X, Y, Z, QX, QY, QZ, QW,
	                                       xN, yN, zN, qxN, qyN, qzN, qwN)
	optimized_metrics = compute_pose_errors_3d(X, Y, Z, QX, QY, QZ, QW,
	                                           xOpt, yOpt, zOpt, qxOpt, qyOpt, qzOpt, qwOpt)
	print_metrics_comparison_3d(noisy_metrics, optimized_metrics)

	# Plot error comparison
	if not args.no_viz:
		plotErrors3D(noisy_metrics, optimized_metrics)


if __name__ == '__main__':
	main()
