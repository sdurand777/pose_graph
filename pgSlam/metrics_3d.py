"""
Metrics computation module for 3D trajectory evaluation
"""
import numpy as np
from scipy.spatial.transform import Rotation


def compute_pose_errors_3d(X_ref, Y_ref, Z_ref, QX_ref, QY_ref, QZ_ref, QW_ref,
                           X_test, Y_test, Z_test, QX_test, QY_test, QZ_test, QW_test):
	"""Compute per-pose errors between test trajectory and reference (3D)

	Args:
		X_ref, Y_ref, Z_ref, QX_ref, QY_ref, QZ_ref, QW_ref: Reference (ground truth) trajectory
		X_test, Y_test, Z_test, QX_test, QY_test, QZ_test, QW_test: Test trajectory to evaluate

	Returns:
		Dictionary containing:
		- 'position_errors': Array of 3D Euclidean distances for each pose
		- 'orientation_errors': Array of angular differences (radians) for each pose
		- 'rmse_position': Root Mean Square Error for position
		- 'rmse_orientation': Root Mean Square Error for orientation
		- 'mean_position_error': Mean position error
		- 'mean_orientation_error': Mean orientation error
		- 'max_position_error': Maximum position error
		- 'max_orientation_error': Maximum orientation error
	"""
	n_poses = len(X_ref)

	# Compute 3D position errors (Euclidean distance)
	position_errors = np.sqrt((np.array(X_ref) - np.array(X_test))**2 +
	                          (np.array(Y_ref) - np.array(Y_test))**2 +
	                          (np.array(Z_ref) - np.array(Z_test))**2)

	# Compute orientation errors (angle between quaternions)
	orientation_errors = np.zeros(n_poses)
	for i in range(n_poses):
		q_ref = [QX_ref[i], QY_ref[i], QZ_ref[i], QW_ref[i]]
		q_test = [QX_test[i], QY_test[i], QZ_test[i], QW_test[i]]

		r_ref = Rotation.from_quat(q_ref)
		r_test = Rotation.from_quat(q_test)

		# Compute relative rotation
		r_diff = r_ref.inv() * r_test

		# Get angle of rotation (geodesic distance on SO(3))
		orientation_errors[i] = r_diff.magnitude()

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


def print_metrics_comparison_3d(noisy_metrics, optimized_metrics):
	"""Print a formatted comparison of metrics between noisy and optimized 3D trajectories

	Args:
		noisy_metrics: Metrics dictionary for noisy trajectory
		optimized_metrics: Metrics dictionary for optimized trajectory
	"""
	print("\n" + "="*70)
	print("3D TRAJECTORY METRICS COMPARISON")
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

	print("\n--- Orientation Errors (degrees) ---")
	print(f"{'Metric':<25} {'Noisy':>15} {'Optimized':>15} {'Improvement':>12}")
	print("-"*70)

	# Convert to degrees for readability
	noisy_rmse_deg = noisy_metrics['rmse_orientation'] * 180 / np.pi
	opt_rmse_deg = optimized_metrics['rmse_orientation'] * 180 / np.pi
	noisy_mean_deg = noisy_metrics['mean_orientation_error'] * 180 / np.pi
	opt_mean_deg = optimized_metrics['mean_orientation_error'] * 180 / np.pi
	noisy_max_deg = noisy_metrics['max_orientation_error'] * 180 / np.pi
	opt_max_deg = optimized_metrics['max_orientation_error'] * 180 / np.pi

	print(f"{'RMSE (degrees)':<25} {noisy_rmse_deg:>15.4f} "
	      f"{opt_rmse_deg:>15.4f} {rmse_orient_improvement:>11.2f}%")
	print(f"{'Mean Error (degrees)':<25} {noisy_mean_deg:>15.4f} "
	      f"{opt_mean_deg:>15.4f} {mean_orient_improvement:>11.2f}%")
	print(f"{'Max Error (degrees)':<25} {noisy_max_deg:>15.4f} "
	      f"{opt_max_deg:>15.4f} {max_orient_improvement:>11.2f}%")

	print("="*70 + "\n")
