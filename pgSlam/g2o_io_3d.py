"""
G2O file I/O module for reading and writing 3D pose graph files (SE3)
"""
import numpy as np
from scipy.spatial.transform import Rotation


def writeOdom3D(X, Y, Z, QX, QY, QZ, QW, odom_info=500.0):
	"""Write 3D odometry edges to a g2o file

	Args:
		X, Y, Z: Position coordinates
		QX, QY, QZ, QW: Quaternion orientation
		odom_info: Information matrix value for odometry (default: 500.0)

	Returns:
		File handle for continued writing
	"""
	g2o = open('noise_3d.g2o', 'w')

	# Write vertices
	for i in range(len(X)):
		# VERTEX_SE3:QUAT id x y z qx qy qz qw
		line = f"VERTEX_SE3:QUAT {i} {X[i]} {Y[i]} {Z[i]} {QX[i]} {QY[i]} {QZ[i]} {QW[i]}\n"
		g2o.write(line)

	# Information matrix for SE3 (6x6 upper triangular)
	# Order: x, y, z, roll, pitch, yaw
	# We use diagonal information matrix for simplicity
	info_mat = f"{odom_info} 0 0 0 0 0 {odom_info} 0 0 0 0 {odom_info} 0 0 0 {odom_info} 0 0 {odom_info} 0 {odom_info}"

	# Write odometry edges
	for i in range(1, len(X)):
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
		del_quat = del_rot.as_quat()  # [x, y, z, w]

		# EDGE_SE3:QUAT id1 id2 dx dy dz dqx dqy dqz dqw info(21 values)
		line = f"EDGE_SE3:QUAT {i-1} {i} {del_pos[0]} {del_pos[1]} {del_pos[2]} "
		line += f"{del_quat[0]} {del_quat[1]} {del_quat[2]} {del_quat[3]} {info_mat}\n"
		g2o.write(line)

	g2o.write("FIX 0\n")
	return g2o


def writeLoop3D(X, Y, Z, QX, QY, QZ, QW, g2o, loop_info=700.0,
                loop_sampling_rate=2, loop_noise_sigma=0.0, seed=None):
	"""Write 3D loop closure edges to a g2o file

	Args:
		X, Y, Z: Position coordinates (reference or noisy)
		QX, QY, QZ, QW: Quaternion orientation (reference or noisy)
		g2o: File handle
		loop_info: Information matrix value for loop closures (default: 700.0)
		loop_sampling_rate: Sample every Nth pose for loop closures (default: 2)
		                    Set to 0 to disable loop closures
		loop_noise_sigma: Standard deviation of Gaussian noise for loop observations (default: 0.0)
		seed: Random seed for loop noise (default: None)

	Returns:
		List of loop closure pairs (for visualization)
	"""
	# Set seed for loop noise if provided
	if seed is not None and loop_noise_sigma > 0:
		np.random.seed(seed + 1000)

	pairs = []

	# If loop_sampling_rate is 0, return empty list (no loops)
	if loop_sampling_rate == 0:
		g2o.close()
		return pairs

	total_poses = len(X)

	# Use proximity-based loop closure detection in 3D
	min_time_separation = 30  # Minimum pose index difference
	proximity_threshold = 3.0  # Maximum 3D distance for loop closure

	for i in range(0, total_poses - min_time_separation, loop_sampling_rate):
		for j in range(i + min_time_separation, total_poses, loop_sampling_rate):
			# Calculate 3D Euclidean distance
			dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
			if dist < proximity_threshold:
				pairs.append((i, j))
				break  # Only one loop per pose

	# Information matrix for SE3 (6x6 upper triangular)
	info_mat = f"{loop_info} 0 0 0 0 0 {loop_info} 0 0 0 0 {loop_info} 0 0 0 {loop_info} 0 0 {loop_info} 0 {loop_info}"

	for p in pairs:
		# Get relative transformation T2_1
		q1 = [QX[p[0]], QY[p[0]], QZ[p[0]], QW[p[0]]]
		r1 = Rotation.from_quat(q1)
		T1_w = np.eye(4)
		T1_w[:3, :3] = r1.as_matrix()
		T1_w[:3, 3] = [X[p[0]], Y[p[0]], Z[p[0]]]

		q2 = [QX[p[1]], QY[p[1]], QZ[p[1]], QW[p[1]]]
		r2 = Rotation.from_quat(q2)
		T2_w = np.eye(4)
		T2_w[:3, :3] = r2.as_matrix()
		T2_w[:3, 3] = [X[p[1]], Y[p[1]], Z[p[1]]]

		T2_1 = np.linalg.inv(T1_w) @ T2_w
		del_pos = T2_1[:3, 3]
		del_rot = Rotation.from_matrix(T2_1[:3, :3])

		# Add noise to loop closure measurement if requested
		if loop_noise_sigma > 0:
			pos_noise = np.random.normal(0, loop_noise_sigma, 3)
			rot_noise = np.random.normal(0, loop_noise_sigma, 3)
			del_pos = del_pos + pos_noise
			del_rot = del_rot * Rotation.from_euler('xyz', rot_noise)

		del_quat = del_rot.as_quat()

		# Write edge
		line = f"EDGE_SE3:QUAT {p[0]} {p[1]} {del_pos[0]} {del_pos[1]} {del_pos[2]} "
		line += f"{del_quat[0]} {del_quat[1]} {del_quat[2]} {del_quat[3]} {info_mat}\n"
		g2o.write(line)

	g2o.close()
	return pairs


def readG2o3D(fileName):
	"""Read vertex positions from a 3D g2o file

	Args:
		fileName: Path to g2o file

	Returns:
		Tuple of (X, Y, Z, QX, QY, QZ, QW) arrays
	"""
	with open(fileName, 'r') as f:
		lines = f.readlines()

	X = []
	Y = []
	Z = []
	QX = []
	QY = []
	QZ = []
	QW = []

	for line in lines:
		if "VERTEX_SE3:QUAT" in line:
			parts = line.split()
			if len(parts) >= 9:
				# Format: VERTEX_SE3:QUAT id x y z qx qy qz qw
				x = float(parts[2])
				y = float(parts[3])
				z = float(parts[4])
				qx = float(parts[5])
				qy = float(parts[6])
				qz = float(parts[7])
				qw = float(parts[8])

				X.append(x)
				Y.append(y)
				Z.append(z)
				QX.append(qx)
				QY.append(qy)
				QZ.append(qz)
				QW.append(qw)

	return (X, Y, Z, QX, QY, QZ, QW)
