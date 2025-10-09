"""
Metrics computation module for trajectory evaluation
"""
import numpy as np
import math


def compute_pose_errors(X_ref, Y_ref, THETA_ref, X_test, Y_test, THETA_test):
	"""Compute per-pose errors between test trajectory and reference

	Args:
		X_ref, Y_ref, THETA_ref: Reference (ground truth) trajectory
		X_test, Y_test, THETA_test: Test trajectory to evaluate

	Returns:
		Dictionary containing:
		- 'position_errors': Array of Euclidean distances for each pose
		- 'orientation_errors': Array of angular differences (radians) for each pose
		- 'rmse_position': Root Mean Square Error for position
		- 'rmse_orientation': Root Mean Square Error for orientation
		- 'mean_position_error': Mean position error
		- 'mean_orientation_error': Mean orientation error
		- 'max_position_error': Maximum position error
		- 'max_orientation_error': Maximum orientation error
	"""
	n_poses = len(X_ref)

	# Compute position errors (Euclidean distance)
	position_errors = np.sqrt((np.array(X_ref) - np.array(X_test))**2 +
	                          (np.array(Y_ref) - np.array(Y_test))**2)

	# Compute orientation errors (normalized to [-pi, pi])
	orientation_errors = np.zeros(n_poses)
	for i in range(n_poses):
		diff = THETA_test[i] - THETA_ref[i]
		# Normalize to [-pi, pi]
		orientation_errors[i] = np.arctan2(np.sin(diff), np.cos(diff))

	# Compute statistics
	rmse_position = np.sqrt(np.mean(position_errors**2))
	rmse_orientation = np.sqrt(np.mean(orientation_errors**2))
	mean_position_error = np.mean(position_errors)
	mean_orientation_error = np.mean(np.abs(orientation_errors))
	max_position_error = np.max(position_errors)
	max_orientation_error = np.max(np.abs(orientation_errors))

	return {
		'position_errors': position_errors,
		'orientation_errors': orientation_errors,
		'rmse_position': rmse_position,
		'rmse_orientation': rmse_orientation,
		'mean_position_error': mean_position_error,
		'mean_orientation_error': mean_orientation_error,
		'max_position_error': max_position_error,
		'max_orientation_error': max_orientation_error
	}


def print_metrics_comparison(noisy_metrics, optimized_metrics):
	"""Print a formatted comparison of metrics between noisy and optimized trajectories

	Args:
		noisy_metrics: Metrics dictionary for noisy trajectory
		optimized_metrics: Metrics dictionary for optimized trajectory
	"""
	print("\n" + "="*70)
	print("TRAJECTORY METRICS COMPARISON")
	print("="*70)

	print("\n--- Position Errors (meters) ---")
	print(f"{'Metric':<25} {'Noisy':>15} {'Optimized':>15} {'Improvement':>12}")
	print("-"*70)

	# RMSE Position
	rmse_improvement = ((noisy_metrics['rmse_position'] - optimized_metrics['rmse_position']) /
	                    noisy_metrics['rmse_position'] * 100)
	print(f"{'RMSE':<25} {noisy_metrics['rmse_position']:>15.6f} "
	      f"{optimized_metrics['rmse_position']:>15.6f} {rmse_improvement:>11.2f}%")

	# Mean Position Error
	mean_improvement = ((noisy_metrics['mean_position_error'] - optimized_metrics['mean_position_error']) /
	                    noisy_metrics['mean_position_error'] * 100)
	print(f"{'Mean Error':<25} {noisy_metrics['mean_position_error']:>15.6f} "
	      f"{optimized_metrics['mean_position_error']:>15.6f} {mean_improvement:>11.2f}%")

	# Max Position Error
	max_improvement = ((noisy_metrics['max_position_error'] - optimized_metrics['max_position_error']) /
	                   noisy_metrics['max_position_error'] * 100)
	print(f"{'Max Error':<25} {noisy_metrics['max_position_error']:>15.6f} "
	      f"{optimized_metrics['max_position_error']:>15.6f} {max_improvement:>11.2f}%")

	print("\n--- Orientation Errors (radians) ---")
	print(f"{'Metric':<25} {'Noisy':>15} {'Optimized':>15} {'Improvement':>12}")
	print("-"*70)

	# RMSE Orientation
	rmse_orient_improvement = ((noisy_metrics['rmse_orientation'] - optimized_metrics['rmse_orientation']) /
	                           noisy_metrics['rmse_orientation'] * 100)
	print(f"{'RMSE':<25} {noisy_metrics['rmse_orientation']:>15.6f} "
	      f"{optimized_metrics['rmse_orientation']:>15.6f} {rmse_orient_improvement:>11.2f}%")

	# Mean Orientation Error
	mean_orient_improvement = ((noisy_metrics['mean_orientation_error'] - optimized_metrics['mean_orientation_error']) /
	                           noisy_metrics['mean_orientation_error'] * 100)
	print(f"{'Mean Error':<25} {noisy_metrics['mean_orientation_error']:>15.6f} "
	      f"{optimized_metrics['mean_orientation_error']:>15.6f} {mean_orient_improvement:>11.2f}%")

	# Max Orientation Error
	max_orient_improvement = ((noisy_metrics['max_orientation_error'] - optimized_metrics['max_orientation_error']) /
	                          noisy_metrics['max_orientation_error'] * 100)
	print(f"{'Max Error':<25} {noisy_metrics['max_orientation_error']:>15.6f} "
	      f"{optimized_metrics['max_orientation_error']:>15.6f} {max_orient_improvement:>11.2f}%")

	print("="*70 + "\n")
