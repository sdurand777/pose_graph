"""
Visualization module for 3D pose graph SLAM
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import numpy as np


def draw_orientation_3d(ax, x, y, z, qx, qy, qz, qw, color='m', length=0.3, alpha=0.7):
	"""Draw 3D orientation arrow from quaternion

	Args:
		ax: 3D matplotlib axis
		x, y, z: Position
		qx, qy, qz, qw: Quaternion
		color: Arrow color
		length: Arrow length
		alpha: Transparency
	"""
	# Convert quaternion to rotation matrix
	r = Rotation.from_quat([qx, qy, qz, qw])
	# Forward direction (x-axis of local frame)
	direction = r.apply([length, 0, 0])

	ax.quiver(x, y, z, direction[0], direction[1], direction[2],
	          color=color, alpha=alpha, arrow_length_ratio=0.3, linewidth=1.5)


def draw3D(X, Y, Z, QX, QY, QZ, QW, loop_pairs=None, title="3D Trajectory"):
	"""Draw a single 3D trajectory with orientation arrows

	Args:
		X, Y, Z: Position coordinates
		QX, QY, QZ, QW: Quaternion orientation
		loop_pairs: Optional list of (i, j) tuples representing loop closures
		title: Plot title
	"""
	fig = plt.figure(figsize=(12, 9))
	ax = fig.add_subplot(111, projection='3d')

	# Plot trajectory
	ax.plot(X, Y, Z, 'ro', markersize=4, label='Poses')
	ax.plot(X, Y, Z, 'k-', linewidth=1.5, alpha=0.7, label='Trajectory')

	# Draw orientation arrows (subsample for clarity)
	arrow_step = max(1, len(X) // 20)
	for i in range(0, len(X), arrow_step):
		draw_orientation_3d(ax, X[i], Y[i], Z[i], QX[i], QY[i], QZ[i], QW[i])

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
			        'c--', linewidth=1.5, alpha=0.6)
		if not any('Loop Closures' in str(h.get_label()) for h in ax.get_legend_handles_labels()[0]):
			ax.plot([], [], 'c--', linewidth=1.5, alpha=0.6, label='Loop Closures')

	ax.set_xlabel('X (m)', fontsize=11)
	ax.set_ylabel('Y (m)', fontsize=11)
	ax.set_zlabel('Z (m)', fontsize=11)
	ax.set_title(title, fontsize=13, fontweight='bold')
	ax.legend(loc='upper right')
	ax.grid(True, alpha=0.3)

	# Equal aspect ratio for all axes
	set_axes_equal(ax)

	plt.tight_layout()
	plt.show()


def drawTwo3D(X1, Y1, Z1, QX1, QY1, QZ1, QW1,
              X2, Y2, Z2, QX2, QY2, QZ2, QW2, loop_pairs=None):
	"""Draw two 3D trajectories: ground truth and optimized

	Args:
		X1, Y1, Z1, QX1, QY1, QZ1, QW1: Ground truth trajectory
		X2, Y2, Z2, QX2, QY2, QZ2, QW2: Optimized trajectory
		loop_pairs: Optional list of (i, j) tuples representing loop closures
	"""
	fig = plt.figure(figsize=(14, 10))
	ax = fig.add_subplot(111, projection='3d')

	# Ground truth
	ax.plot(X1, Y1, Z1, 'ro', markersize=5, label='Ground Truth', alpha=0.8)
	ax.plot(X1, Y1, Z1, 'r-', linewidth=2, alpha=0.5)

	# Optimized
	ax.plot(X2, Y2, Z2, 'bo', markersize=5, label='Optimized', alpha=0.8)
	ax.plot(X2, Y2, Z2, 'b-', linewidth=2, alpha=0.5)

	# Draw orientation arrows for both (sparse)
	arrow_step = max(1, len(X1) // 15)
	for i in range(0, len(X1), arrow_step):
		draw_orientation_3d(ax, X1[i], Y1[i], Z1[i], QX1[i], QY1[i], QZ1[i], QW1[i],
		                    color='red', length=0.4, alpha=0.6)
		draw_orientation_3d(ax, X2[i], Y2[i], Z2[i], QX2[i], QY2[i], QZ2[i], QW2[i],
		                    color='blue', length=0.4, alpha=0.6)

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			ax.plot([X1[i], X1[j]], [Y1[i], Y1[j]], [Z1[i], Z1[j]],
			        'c--', linewidth=1.5, alpha=0.5, label='Loop Closures' if (i, j) == loop_pairs[0] else '')

	ax.set_xlabel('X (m)', fontsize=11)
	ax.set_ylabel('Y (m)', fontsize=11)
	ax.set_zlabel('Z (m)', fontsize=11)
	ax.set_title('Ground Truth vs Optimized Trajectory (3D)', fontsize=13, fontweight='bold')
	ax.legend(loc='upper right')
	ax.grid(True, alpha=0.3)

	set_axes_equal(ax)

	plt.tight_layout()
	plt.show()


def drawThree3D(X1, Y1, Z1, QX1, QY1, QZ1, QW1,
                X2, Y2, Z2, QX2, QY2, QZ2, QW2,
                X3, Y3, Z3, QX3, QY3, QZ3, QW3, loop_pairs=None):
	"""Draw three 3D trajectories: ground truth, optimized, and noisy

	Args:
		X1, Y1, Z1, QX1, QY1, QZ1, QW1: Ground truth trajectory
		X2, Y2, Z2, QX2, QY2, QZ2, QW2: Optimized trajectory
		X3, Y3, Z3, QX3, QY3, QZ3, QW3: Noisy trajectory
		loop_pairs: Optional list of (i, j) tuples representing loop closures
	"""
	fig = plt.figure(figsize=(16, 11))
	ax = fig.add_subplot(111, projection='3d')

	# Ground truth
	ax.plot(X1, Y1, Z1, 'ro', markersize=4, label='Ground Truth', alpha=0.8)
	ax.plot(X1, Y1, Z1, 'r-', linewidth=2, alpha=0.5)

	# Optimized
	ax.plot(X2, Y2, Z2, 'bo', markersize=4, label='Optimized', alpha=0.8)
	ax.plot(X2, Y2, Z2, 'b-', linewidth=2, alpha=0.5)

	# Noisy
	ax.plot(X3, Y3, Z3, 'go', markersize=4, label='Noisy', alpha=0.8)
	ax.plot(X3, Y3, Z3, 'g-', linewidth=2, alpha=0.5)

	# Draw orientation arrows (very sparse for clarity with 3 trajectories)
	arrow_step = max(1, len(X1) // 12)
	for i in range(0, len(X1), arrow_step):
		draw_orientation_3d(ax, X1[i], Y1[i], Z1[i], QX1[i], QY1[i], QZ1[i], QW1[i],
		                    color='red', length=0.35, alpha=0.5)
		draw_orientation_3d(ax, X2[i], Y2[i], Z2[i], QX2[i], QY2[i], QZ2[i], QW2[i],
		                    color='blue', length=0.35, alpha=0.5)
		draw_orientation_3d(ax, X3[i], Y3[i], Z3[i], QX3[i], QY3[i], QZ3[i], QW3[i],
		                    color='green', length=0.35, alpha=0.5)

	# Draw loop closures if provided
	if loop_pairs:
		for (i, j) in loop_pairs:
			ax.plot([X1[i], X1[j]], [Y1[i], Y1[j]], [Z1[i], Z1[j]],
			        'c--', linewidth=1.5, alpha=0.5, label='Loop Closures' if (i, j) == loop_pairs[0] else '')

	ax.set_xlabel('X (m)', fontsize=11)
	ax.set_ylabel('Y (m)', fontsize=11)
	ax.set_zlabel('Z (m)', fontsize=11)
	ax.set_title('Ground Truth vs Optimized vs Noisy (3D)', fontsize=13, fontweight='bold')
	ax.legend(loc='upper right')
	ax.grid(True, alpha=0.3)

	set_axes_equal(ax)

	plt.tight_layout()
	plt.show()


def plotErrors3D(noisy_metrics, optimized_metrics):
	"""Plot per-pose errors for noisy and optimized 3D trajectories

	Args:
		noisy_metrics: Metrics dictionary for noisy trajectory
		optimized_metrics: Metrics dictionary for optimized trajectory
	"""
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

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
	ax1.set_title('3D Position Errors vs Ground Truth', fontsize=14, fontweight='bold')
	ax1.legend(loc='upper right')
	ax1.grid(True, alpha=0.3)

	# Plot orientation errors (convert to degrees)
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
	ax2.set_title('3D Orientation Errors vs Ground Truth', fontsize=14, fontweight='bold')
	ax2.legend(loc='upper right')
	ax2.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def set_axes_equal(ax):
	"""Set 3D plot axes to equal scale

	Args:
		ax: matplotlib 3D axis
	"""
	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_middle = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_middle = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_middle = np.mean(z_limits)

	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = 0.5 * max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
