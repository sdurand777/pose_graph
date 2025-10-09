"""
Main script for pose graph SLAM demonstration
"""
import argparse
from trajectory import genTraj, genTrajSquareSpiral, addNoise
from visualization import draw, drawTwo, drawThree, plotErrors
from g2o_io import writeOdom, writeLoop, readG2o
from pose_graph import optimize
from metrics import compute_pose_errors, print_metrics_comparison


def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Pose Graph SLAM Optimization Demo')
	parser.add_argument('--trajectory-type', type=str, default='uturn', choices=['uturn', 'square-spiral'],
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

	print(f"Running pose graph optimization with:")
	print(f"  Trajectory type: {args.trajectory_type}")
	print(f"  Odometry info: {args.odom_info}")
	print(f"  Loop closure info: {args.loop_info}")
	print(f"  Poses per segment: {args.num_poses}")
	print(f"  Loop sampling rate: {args.loop_sampling}")
	print(f"  Odometry noise sigma: {args.noise_sigma}")
	print(f"  Loop from noisy poses: {args.loop_from_noisy}")
	print(f"  Loop noise sigma: {args.loop_noise_sigma}")
	print(f"  Random seed: {args.seed if args.seed is not None else 'None (random)'}")

	# Generate ground truth trajectory based on type
	if args.trajectory_type == 'uturn':
		(X, Y, THETA) = genTraj(num_poses_per_segment=args.num_poses)
		print(f"  Total poses: ~{args.num_poses * 6}")
	elif args.trajectory_type == 'square-spiral':
		(X, Y, THETA) = genTrajSquareSpiral(num_poses_per_segment=args.num_poses)
		print(f"  Total poses: {len(X)}")
	if not args.no_viz:
		draw(X, Y, THETA)

	# Add noise to trajectory
	(xN, yN, tN) = addNoise(X, Y, THETA, seed=args.seed, noise_sigma=args.noise_sigma)
	g2o = writeOdom(xN, yN, tN, odom_info=args.odom_info)

	# Choose which poses to use for loop closures
	if args.loop_from_noisy:
		loop_X, loop_Y, loop_THETA = xN, yN, tN
	else:
		loop_X, loop_Y, loop_THETA = X, Y, THETA

	loop_pairs = writeLoop(loop_X, loop_Y, loop_THETA, g2o,
	                       loop_info=args.loop_info,
	                       loop_sampling_rate=args.loop_sampling,
	                       loop_noise_sigma=args.loop_noise_sigma,
	                       seed=args.seed)

	print(f"  Number of loop closures: {len(loop_pairs)}")
	if len(loop_pairs) > 0:
		print(f"  Loop closure connections:")
		for i, (start, end) in enumerate(loop_pairs):
			if i < 5:  # Show first 5 loop closures
				print(f"    Pose {start} <-> Pose {end}")
		if len(loop_pairs) > 5:
			print(f"    ... and {len(loop_pairs) - 5} more loop closures")

	if not args.no_viz:
		draw(xN, yN, tN, loop_pairs=loop_pairs)

	# Optimize pose graph
	optimize()
	(xOpt, yOpt, tOpt) = readG2o("opt.g2o")
	if not args.no_viz:
		drawTwo(X, Y, THETA, xOpt, yOpt, tOpt, loop_pairs=loop_pairs)

	# Show all three trajectories
	if not args.no_viz:
		drawThree(X, Y, THETA, xOpt, yOpt, tOpt, xN, yN, tN, loop_pairs=loop_pairs)

	print("Optimization complete!")

	# Compute and display metrics
	noisy_metrics = compute_pose_errors(X, Y, THETA, xN, yN, tN)
	optimized_metrics = compute_pose_errors(X, Y, THETA, xOpt, yOpt, tOpt)
	print_metrics_comparison(noisy_metrics, optimized_metrics)

	# Plot error comparison
	if not args.no_viz:
		plotErrors(noisy_metrics, optimized_metrics)


if __name__ == '__main__':
	main()
