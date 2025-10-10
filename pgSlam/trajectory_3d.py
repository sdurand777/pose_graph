"""
Trajectory generation module for 3D pose graph SLAM
Uses SE(3) transformations with quaternions for orientation
"""
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_from_euler(roll, pitch, yaw):
	"""Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)"""
	r = Rotation.from_euler('xyz', [roll, pitch, yaw])
	quat = r.as_quat()  # Returns [x, y, z, w]
	return quat


def euler_from_quaternion(quat):
	"""Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)"""
	r = Rotation.from_quat(quat)
	euler = r.as_euler('xyz')
	return euler


def genTraj3D(num_poses_per_segment=20):
	"""Generate a synthetic 3D trajectory with multiple U-turns and vertical motion

	Creates a 3D path that includes:
	- Forward motion with ascending/descending altitude
	- U-turns in 3D space
	- Pitch and roll variations

	Args:
		num_poses_per_segment: Number of poses per trajectory segment (default: 20)
		                       Total poses will be approximately 6 * num_poses_per_segment

	Returns:
		Tuple of (X, Y, Z, QX, QY, QZ, QW) arrays representing the 3D trajectory
		where (QX, QY, QZ, QW) is the quaternion orientation
	"""
	# Forward I - ascending
	num = num_poses_per_segment
	xSt, ySt, zSt = -5.0, -8.0, 0.0
	leng = 9.0
	step = float(leng) / num
	X1 = np.zeros(num)
	Y1 = np.zeros(num)
	Z1 = np.zeros(num)
	QX1 = np.zeros(num)
	QY1 = np.zeros(num)
	QZ1 = np.zeros(num)
	QW1 = np.zeros(num)

	for i in range(num):
		X1[i] = xSt + i * step
		Y1[i] = ySt
		Z1[i] = zSt + i * (2.0 / num)  # Gradual ascent
		# Forward orientation with slight pitch up
		pitch = 0.1  # ~5.7 degrees
		quat = quaternion_from_euler(0, pitch, 0)
		QX1[i], QY1[i], QZ1[i], QW1[i] = quat

	# UTurn I - 3D semicircular arc
	rad = 2.5
	num = num_poses_per_segment
	xCen = X1[-1]
	yCen = Y1[-1] + rad
	zStart = Z1[-1]
	thetas = np.linspace(-np.pi / 2, np.pi / 2, num)
	X2 = np.zeros(num)
	Y2 = np.zeros(num)
	Z2 = np.zeros(num)
	QX2 = np.zeros(num)
	QY2 = np.zeros(num)
	QZ2 = np.zeros(num)
	QW2 = np.zeros(num)

	for i, theta in enumerate(thetas):
		X2[i] = xCen + rad * np.cos(theta)
		Y2[i] = yCen + rad * np.sin(theta)
		Z2[i] = zStart + 0.5 * np.sin(theta)  # Arc up and down
		# Orientation tangent to arc
		yaw = theta + np.pi / 2
		roll = 0.15 * np.sin(theta)  # Banking into turn
		quat = quaternion_from_euler(roll, 0, yaw)
		QX2[i], QY2[i], QZ2[i], QW2[i] = quat

	# Backward I - descending
	num = num_poses_per_segment
	leng = 10.0
	step = float(leng) / num
	xSt = X2[-1]
	ySt = Y2[-1]
	zSt = Z2[-1]
	X3 = np.zeros(num)
	Y3 = np.zeros(num)
	Z3 = np.zeros(num)
	QX3 = np.zeros(num)
	QY3 = np.zeros(num)
	QZ3 = np.zeros(num)
	QW3 = np.zeros(num)

	for i in range(num):
		X3[i] = xSt - i * step
		Y3[i] = ySt
		Z3[i] = zSt - i * (1.5 / num)  # Gradual descent
		# Backward orientation with slight pitch down
		pitch = -0.08
		quat = quaternion_from_euler(0, pitch, np.pi)
		QX3[i], QY3[i], QZ3[i], QW3[i] = quat

	# UTurn II
	rad = 2.6
	num = num_poses_per_segment
	xCen = X3[-1]
	yCen = Y3[-1] - rad
	zStart = Z3[-1]
	thetas = np.linspace(np.pi / 2, 3 * np.pi / 2, num)
	X4 = np.zeros(num)
	Y4 = np.zeros(num)
	Z4 = np.zeros(num)
	QX4 = np.zeros(num)
	QY4 = np.zeros(num)
	QZ4 = np.zeros(num)
	QW4 = np.zeros(num)

	for i, theta in enumerate(thetas):
		X4[i] = xCen + rad * np.cos(theta)
		Y4[i] = yCen + rad * np.sin(theta)
		Z4[i] = zStart + 0.4 * np.sin(theta - np.pi / 2)
		yaw = theta + np.pi / 2
		roll = 0.15 * np.sin(theta)
		quat = quaternion_from_euler(roll, 0, yaw)
		QX4[i], QY4[i], QZ4[i], QW4[i] = quat

	# Forward II - level flight
	num = num_poses_per_segment
	leng = 11.0
	step = float(leng) / num
	xSt = X4[-1]
	ySt = Y4[-1]
	zSt = Z4[-1]
	X5 = np.zeros(num)
	Y5 = np.zeros(num)
	Z5 = np.zeros(num)
	QX5 = np.zeros(num)
	QY5 = np.zeros(num)
	QZ5 = np.zeros(num)
	QW5 = np.zeros(num)

	for i in range(num):
		X5[i] = xSt + i * step
		Y5[i] = ySt
		Z5[i] = zSt + 0.3 * np.sin(i * np.pi / num)  # Slight wave
		pitch = 0.05 * np.cos(i * np.pi / num)
		quat = quaternion_from_euler(0, pitch, 0)
		QX5[i], QY5[i], QZ5[i], QW5[i] = quat

	# UTurn III
	rad = 2.7
	num = num_poses_per_segment
	xCen = X5[-1]
	yCen = Y5[-1] + rad
	zStart = Z5[-1]
	thetas = np.linspace(-np.pi / 2, np.pi / 2, num)
	X6 = np.zeros(num)
	Y6 = np.zeros(num)
	Z6 = np.zeros(num)
	QX6 = np.zeros(num)
	QY6 = np.zeros(num)
	QZ6 = np.zeros(num)
	QW6 = np.zeros(num)

	for i, theta in enumerate(thetas):
		X6[i] = xCen + rad * np.cos(theta)
		Y6[i] = yCen + rad * np.sin(theta)
		Z6[i] = zStart + 0.6 * np.sin(theta)
		yaw = theta + np.pi / 2
		roll = 0.15 * np.sin(theta)
		quat = quaternion_from_euler(roll, 0, yaw)
		QX6[i], QY6[i], QZ6[i], QW6[i] = quat

	# Assemble
	X = np.concatenate([X1, X2, X3, X4, X5, X6])
	Y = np.concatenate([Y1, Y2, Y3, Y4, Y5, Y6])
	Z = np.concatenate([Z1, Z2, Z3, Z4, Z5, Z6])
	QX = np.concatenate([QX1, QX2, QX3, QX4, QX5, QX6])
	QY = np.concatenate([QY1, QY2, QY3, QY4, QY5, QY6])
	QZ = np.concatenate([QZ1, QZ2, QZ3, QZ4, QZ5, QZ6])
	QW = np.concatenate([QW1, QW2, QW3, QW4, QW5, QW6])

	return (X, Y, Z, QX, QY, QZ, QW)


def genTrajSpiralHelix3D(num_poses_per_segment=20):
	"""Generate a 3D helical spiral trajectory

	Creates a trajectory that spirals upward in a helical pattern,
	creating natural 3D loop closures.

	Args:
		num_poses_per_segment: Number of poses per revolution (default: 20)

	Returns:
		Tuple of (X, Y, Z, QX, QY, QZ, QW) arrays representing the 3D trajectory
	"""
	X_all = []
	Y_all = []
	Z_all = []
	QX_all = []
	QY_all = []
	QZ_all = []
	QW_all = []

	num_revolutions = 4
	total_poses = num_poses_per_segment * num_revolutions

	for i in range(total_poses):
		# Parameter along helix
		t = i / float(num_poses_per_segment) * 2 * np.pi

		# Expanding radius spiral
		radius = 2.0 + 0.5 * (i / float(total_poses)) * 10

		# Position
		x = radius * np.cos(t)
		y = radius * np.sin(t)
		z = i * (5.0 / total_poses)  # Gradual ascent

		X_all.append(x)
		Y_all.append(y)
		Z_all.append(z)

		# Orientation tangent to helix
		yaw = t + np.pi / 2
		pitch = np.arctan(5.0 / (2 * np.pi * radius))  # Pitch based on helix angle
		roll = 0.1 * np.sin(t)  # Slight banking

		quat = quaternion_from_euler(roll, pitch, yaw)
		QX_all.append(quat[0])
		QY_all.append(quat[1])
		QZ_all.append(quat[2])
		QW_all.append(quat[3])

	return (np.array(X_all), np.array(Y_all), np.array(Z_all),
	        np.array(QX_all), np.array(QY_all), np.array(QZ_all), np.array(QW_all))


def addNoise3D(X, Y, Z, QX, QY, QZ, QW, seed=None, noise_sigma=0.03):
	"""Add Gaussian noise to 3D trajectory odometry

	Args:
		X, Y, Z: Position coordinates
		QX, QY, QZ, QW: Quaternion orientation
		seed: Random seed for reproducibility (default: None for random noise)
		noise_sigma: Standard deviation of Gaussian noise (default: 0.03)

	Returns:
		Tuple of (xN, yN, zN, qxN, qyN, qzN, qwN) noisy trajectory
	"""
	if seed is not None:
		np.random.seed(seed)

	n = len(X)
	xN = np.zeros(n)
	yN = np.zeros(n)
	zN = np.zeros(n)
	qxN = np.zeros(n)
	qyN = np.zeros(n)
	qzN = np.zeros(n)
	qwN = np.zeros(n)

	# First pose is fixed
	xN[0] = X[0]
	yN[0] = Y[0]
	zN[0] = Z[0]
	qxN[0] = QX[0]
	qyN[0] = QY[0]
	qzN[0] = QZ[0]
	qwN[0] = QW[0]

	for i in range(1, n):
		# Get relative transformation T2_1
		# T1_w
		q1 = [QX[i-1], QY[i-1], QZ[i-1], QW[i-1]]
		r1 = Rotation.from_quat(q1)
		T1_w = np.eye(4)
		T1_w[:3, :3] = r1.as_matrix()
		T1_w[:3, 3] = [X[i-1], Y[i-1], Z[i-1]]

		# T2_w
		q2 = [QX[i], QY[i], QZ[i], QW[i]]
		r2 = Rotation.from_quat(q2)
		T2_w = np.eye(4)
		T2_w[:3, :3] = r2.as_matrix()
		T2_w[:3, 3] = [X[i], Y[i], Z[i]]

		# Relative transform
		T2_1 = np.linalg.inv(T1_w) @ T2_w
		del_pos = T2_1[:3, 3]
		del_rot = Rotation.from_matrix(T2_1[:3, :3])

		# Add noise in local frame
		if i < 5:
			pos_noise = np.zeros(3)
			rot_noise = np.zeros(3)
		else:
			pos_noise = np.random.normal(0, noise_sigma, 3)
			rot_noise = np.random.normal(0, noise_sigma, 3)

		# Noisy relative transformation
		del_posN = del_pos + pos_noise
		del_rotN = del_rot * Rotation.from_euler('xyz', rot_noise)

		T2_1N = np.eye(4)
		T2_1N[:3, :3] = del_rotN.as_matrix()
		T2_1N[:3, 3] = del_posN

		# Propagate in world frame: T2_w' = T1_w' * T2_1'
		q1N = [qxN[i-1], qyN[i-1], qzN[i-1], qwN[i-1]]
		r1N = Rotation.from_quat(q1N)
		T1_wN = np.eye(4)
		T1_wN[:3, :3] = r1N.as_matrix()
		T1_wN[:3, 3] = [xN[i-1], yN[i-1], zN[i-1]]

		T2_wN = T1_wN @ T2_1N

		# Extract noisy pose
		xN[i] = T2_wN[0, 3]
		yN[i] = T2_wN[1, 3]
		zN[i] = T2_wN[2, 3]

		r2N = Rotation.from_matrix(T2_wN[:3, :3])
		q2N = r2N.as_quat()
		qxN[i], qyN[i], qzN[i], qwN[i] = q2N

	return (xN, yN, zN, qxN, qyN, qzN, qwN)
