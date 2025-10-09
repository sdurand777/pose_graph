"""
Visualization module for pose graph SLAM
"""
import matplotlib.pyplot as plt
import numpy as np
import math


def draw(X, Y, THETA, loop_pairs=None):
	"""Draw a single trajectory with orientation arrows

	Args:
		X: X coordinates
		Y: Y coordinates
		THETA: Orientations
		loop_pairs: Optional list of (i, j) tuples representing loop closures
	"""
	ax = plt.subplot(111)
	ax.plot(X, Y, 'ro')
	plt.plot(X, Y, 'k-')

	for i in range(len(THETA)):
		x2 = 0.25 * math.cos(THETA[i]) + X[i]
		y2 = 0.25 * math.sin(THETA[i]) + Y[i]
		plt.plot([X[i], x2], [Y[i], y2], 'm->')

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			plt.plot([X[i], X[j]], [Y[i], Y[j]], 'c--', linewidth=1.5, alpha=0.6)

	if loop_pairs:
		plt.legend(['Poses', 'Trajectory', 'Orientation', 'Loop Closures'])

	plt.axis('equal')
	plt.grid(True, alpha=0.3)
	plt.show()


def drawTwo(X1, Y1, THETA1, X2, Y2, THETA2, loop_pairs=None):
	"""Draw two trajectories: ground truth and optimized

	Args:
		X1, Y1, THETA1: Ground truth trajectory
		X2, Y2, THETA2: Optimized trajectory
		loop_pairs: Optional list of (i, j) tuples representing loop closures
	"""
	ax = plt.subplot(111)
	ax.plot(X1, Y1, 'ro', label='Ground Truth')
	plt.plot(X1, Y1, 'k-')

	for i in range(len(THETA1)):
		x2 = 0.25 * math.cos(THETA1[i]) + X1[i]
		y2 = 0.25 * math.sin(THETA1[i]) + Y1[i]
		plt.plot([X1[i], x2], [Y1[i], y2], 'r->')

	ax.plot(X2, Y2, 'bo', label='Optimized')
	plt.plot(X2, Y2, 'k-')

	for i in range(len(THETA2)):
		x2 = 0.25 * math.cos(THETA2[i]) + X2[i]
		y2 = 0.25 * math.sin(THETA2[i]) + Y2[i]
		plt.plot([X2[i], x2], [Y2[i], y2], 'b->')

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			plt.plot([X1[i], X1[j]], [Y1[i], Y1[j]], 'c--', linewidth=1.5, alpha=0.6, label='Loop Closures' if (i, j) == loop_pairs[0] else '')

	plt.legend()
	plt.axis('equal')
	plt.grid(True, alpha=0.3)
	plt.show()


def drawThree(X1, Y1, THETA1, X2, Y2, THETA2, X3, Y3, THETA3, loop_pairs=None):
	"""Draw three trajectories: ground truth, optimized, and noisy

	Args:
		X1, Y1, THETA1: Ground truth trajectory
		X2, Y2, THETA2: Optimized trajectory
		X3, Y3, THETA3: Noisy trajectory
		loop_pairs: Optional list of (i, j) tuples representing loop closures
	"""
	ax = plt.subplot(111)
	ax.plot(X1, Y1, 'ro', label='Ground Truth')
	plt.plot(X1, Y1, 'k-')

	for i in range(len(THETA1)):
		x2 = 0.25 * math.cos(THETA1[i]) + X1[i]
		y2 = 0.25 * math.sin(THETA1[i]) + Y1[i]
		plt.plot([X1[i], x2], [Y1[i], y2], 'r->')

	ax.plot(X2, Y2, 'bo', label='Optimized')
	plt.plot(X2, Y2, 'k-')

	for i in range(len(THETA2)):
		x2 = 0.25 * math.cos(THETA2[i]) + X2[i]
		y2 = 0.25 * math.sin(THETA2[i]) + Y2[i]
		plt.plot([X2[i], x2], [Y2[i], y2], 'b->')

	ax.plot(X3, Y3, 'go', label='Noisy')
	plt.plot(X3, Y3, 'k-')

	for i in range(len(THETA3)):
		x2 = 0.25 * math.cos(THETA3[i]) + X3[i]
		y2 = 0.25 * math.sin(THETA3[i]) + Y3[i]
		plt.plot([X3[i], x2], [Y3[i], y2], 'g->')

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			plt.plot([X1[i], X1[j]], [Y1[i], Y1[j]], 'c--', linewidth=1.5, alpha=0.6, label='Loop Closures' if (i, j) == loop_pairs[0] else '')

	plt.legend()
	plt.axis('equal')
	plt.grid(True, alpha=0.3)
	plt.show()


def plotErrors(noisy_metrics, optimized_metrics):
	"""Plot per-pose errors for noisy and optimized trajectories

	Args:
		noisy_metrics: Metrics dictionary for noisy trajectory
		optimized_metrics: Metrics dictionary for optimized trajectory
	"""
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

	n_poses = len(noisy_metrics['position_errors'])
	pose_indices = np.arange(n_poses)

	# Plot position errors
	ax1.plot(pose_indices, noisy_metrics['position_errors'], 'g-', label='Noisy', linewidth=2)
	ax1.plot(pose_indices, optimized_metrics['position_errors'], 'b-', label='Optimized', linewidth=2)
	ax1.axhline(y=noisy_metrics['rmse_position'], color='g', linestyle='--', alpha=0.7,
	            label=f'Noisy RMSE: {noisy_metrics["rmse_position"]:.4f}m')
	ax1.axhline(y=optimized_metrics['rmse_position'], color='b', linestyle='--', alpha=0.7,
	            label=f'Optimized RMSE: {optimized_metrics["rmse_position"]:.4f}m')
	ax1.set_xlabel('Pose Index', fontsize=12)
	ax1.set_ylabel('Position Error (m)', fontsize=12)
	ax1.set_title('Position Errors vs Ground Truth', fontsize=14, fontweight='bold')
	ax1.legend(loc='upper right')
	ax1.grid(True, alpha=0.3)

	# Plot orientation errors (convert to degrees for better readability)
	noisy_orient_deg = np.abs(noisy_metrics['orientation_errors']) * 180 / np.pi
	opt_orient_deg = np.abs(optimized_metrics['orientation_errors']) * 180 / np.pi
	ax2.plot(pose_indices, noisy_orient_deg, 'g-', label='Noisy', linewidth=2)
	ax2.plot(pose_indices, opt_orient_deg, 'b-', label='Optimized', linewidth=2)
	ax2.axhline(y=noisy_metrics['rmse_orientation'] * 180 / np.pi, color='g', linestyle='--', alpha=0.7,
	            label=f'Noisy RMSE: {noisy_metrics["rmse_orientation"]*180/np.pi:.4f}°')
	ax2.axhline(y=optimized_metrics['rmse_orientation'] * 180 / np.pi, color='b', linestyle='--', alpha=0.7,
	            label=f'Optimized RMSE: {optimized_metrics["rmse_orientation"]*180/np.pi:.4f}°')
	ax2.set_xlabel('Pose Index', fontsize=12)
	ax2.set_ylabel('Orientation Error (degrees)', fontsize=12)
	ax2.set_title('Orientation Errors vs Ground Truth', fontsize=14, fontweight='bold')
	ax2.legend(loc='upper right')
	ax2.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def drawLoops(X, Y, THETA, loop_pairs):
	"""Draw only the reference trajectory with loop closures highlighted

	This visualization is useful to understand which poses are connected
	by loop closures and verify the loop closure detection algorithm.

	Args:
		X, Y, THETA: Reference (ground truth) trajectory
		loop_pairs: List of (i, j) tuples representing loop closures
	"""
	fig, ax = plt.subplots(figsize=(12, 10))

	# Draw trajectory
	ax.plot(X, Y, 'k-', linewidth=2, label='Reference Trajectory', alpha=0.7)
	ax.plot(X, Y, 'ro', markersize=4, alpha=0.6)

	# Draw orientation arrows (fewer for clarity)
	arrow_step = max(1, len(X) // 30)  # Show ~30 arrows
	for i in range(0, len(THETA), arrow_step):
		x2 = 0.3 * math.cos(THETA[i]) + X[i]
		y2 = 0.3 * math.sin(THETA[i]) + Y[i]
		ax.arrow(X[i], Y[i], x2 - X[i], y2 - Y[i],
		         head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.5)

	# Draw loop closures with different colors based on distance
	if loop_pairs and len(loop_pairs) > 0:
		distances = []
		for (i, j) in loop_pairs:
			dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
			distances.append(dist)

		# Normalize distances for color mapping
		max_dist = max(distances)
		min_dist = min(distances)

		for idx, (i, j) in enumerate(loop_pairs):
			# Color based on distance (green = close, yellow = far)
			if max_dist > min_dist:
				normalized = (distances[idx] - min_dist) / (max_dist - min_dist)
			else:
				normalized = 0
			color = plt.cm.RdYlGn_r(normalized)

			# Draw loop closure line
			ax.plot([X[i], X[j]], [Y[i], Y[j]], color=color,
			        linestyle='--', linewidth=1.5, alpha=0.7)

			# Highlight the connected poses
			ax.plot(X[i], Y[i], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
			ax.plot(X[j], Y[j], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)

	# Add grid and labels
	ax.set_xlabel('X (meters)', fontsize=12)
	ax.set_ylabel('Y (meters)', fontsize=12)
	ax.set_title(f'Loop Closures Visualization\n{len(loop_pairs)} loops detected',
	             fontsize=14, fontweight='bold')
	ax.axis('equal')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best')

	# Add colorbar for distance
	if loop_pairs and len(loop_pairs) > 0:
		sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
		                            norm=plt.Normalize(vmin=min_dist, vmax=max_dist))
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=ax)
		cbar.set_label('Loop Closure Distance (m)', fontsize=10)

	plt.tight_layout()
	plt.show()
