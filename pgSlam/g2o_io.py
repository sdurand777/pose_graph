"""
G2O file I/O module for reading and writing pose graph files
"""
import numpy as np
import math


def writeOdom(X, Y, THETA, odom_info=500.0):
	"""Write odometry edges to a g2o file

	Args:
		X: X coordinates
		Y: Y coordinates
		THETA: Orientations
		odom_info: Information matrix value for odometry (default: 500.0)
	"""
	g2o = open('noise.g2o', 'w')

	for i, (x, y, theta) in enumerate(zip(X, Y, THETA)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	info_mat = f"{odom_info} 0.0 0.0 {odom_info} 0.0 {odom_info}"
	for i in range(1, len(X)):
		p1 = (X[i - 1], Y[i - 1], THETA[i - 1])
		p2 = (X[i], Y[i], THETA[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]],
		                 [math.sin(p1[2]), math.cos(p1[2]), p1[1]],
		                 [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]],
		                 [math.sin(p2[2]), math.cos(p2[2]), p2[1]],
		                 [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))

		line = "EDGE_SE2 " + str(i - 1) + " " + str(i) + " " + del_x + " " + del_y + " " + del_theta + " " + info_mat
		g2o.write(line)
		g2o.write("\n")

	g2o.write("FIX 0")
	g2o.write("\n")
	return g2o


def writeLoop(X, Y, THETA, g2o, loop_info=700.0, loop_sampling_rate=2, loop_noise_sigma=0.0, seed=None):
	"""Write loop closure edges to a g2o file

	Args:
		X: X coordinates (reference or noisy, depending on use case)
		Y: Y coordinates (reference or noisy, depending on use case)
		THETA: Orientations (reference or noisy, depending on use case)
		g2o: File handle
		loop_info: Information matrix value for loop closures (default: 700.0)
		loop_sampling_rate: Sample every Nth pose for loop closures (default: 2)
		                    Higher values = fewer loops, lower values = more loops
		                    Set to 0 to disable loop closures
		loop_noise_sigma: Standard deviation of Gaussian noise for loop observations (default: 0.0)
		seed: Random seed for loop noise (default: None)

	Returns:
		List of loop closure pairs (for visualization)
	"""
	# Set seed for loop noise if provided
	if seed is not None and loop_noise_sigma > 0:
		np.random.seed(seed + 1000)  # Offset to avoid collision with trajectory noise
	pairs = []

	# If loop_sampling_rate is 0, return empty list (no loops)
	if loop_sampling_rate == 0:
		g2o.close()
		return pairs

	total_poses = len(X)

	# Use proximity-based loop closure detection
	# Find poses that are close in space but far in time
	min_time_separation = 30  # Minimum pose index difference (reduced for more loops)
	proximity_threshold = 2.5  # Maximum distance for loop closure (increased for square spiral)

	for i in range(0, total_poses - min_time_separation, loop_sampling_rate):
		for j in range(i + min_time_separation, total_poses, loop_sampling_rate):
			# Calculate Euclidean distance
			dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
			if dist < proximity_threshold:
				pairs.append((i, j))
				break  # Only one loop per pose to avoid too many constraints

	info_mat = f"{loop_info} 0.0 0.0 {loop_info} 0.0 {loop_info}"

	for p in pairs:
		p1 = (X[p[0]], Y[p[0]], THETA[p[0]])
		p2 = (X[p[1]], Y[p[1]], THETA[p[1]])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]],
		                 [math.sin(p1[2]), math.cos(p1[2]), p1[1]],
		                 [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]],
		                 [math.sin(p2[2]), math.cos(p2[2]), p2[1]],
		                 [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = T2_1[0][2]
		del_y = T2_1[1][2]
		del_theta = math.atan2(T2_1[1, 0], T2_1[0, 0])

		# Add noise to loop closure measurement if requested
		if loop_noise_sigma > 0:
			del_x += np.random.normal(0, loop_noise_sigma)
			del_y += np.random.normal(0, loop_noise_sigma)
			del_theta += np.random.normal(0, loop_noise_sigma)

		line = "EDGE_SE2 " + str(p[0]) + " " + str(p[1]) + " " + str(del_x) + " " + str(del_y) + " " + str(del_theta) + " " + info_mat
		g2o.write(line)
		g2o.write("\n")

	g2o.close()
	return pairs


def readG2o(fileName):
	"""Read vertex positions from a g2o file"""
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for line in A:
		if "VERTEX_SE2" in line:
			parts = line.split()
			if len(parts) >= 5:
				ver, ind, x, y, theta = parts[0], parts[1], parts[2], parts[3], parts[4]
				X.append(float(x))
				Y.append(float(y))
				THETA.append(float(theta))

	return (X, Y, THETA)
