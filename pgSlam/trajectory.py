"""
Trajectory generation module for pose graph SLAM
"""
import numpy as np
import math


def getTheta(X, Y):
	"""Calculate orientation angles from X, Y positions"""
	THETA = [None] * len(X)
	for i in range(1, len(X) - 1):
		if X[i + 1] == X[i - 1]:
			if Y[i + 1] > Y[i - 1]:
				THETA[i] = math.pi / 2
			else:
				THETA[i] = 3 * math.pi / 2
			continue

		THETA[i] = math.atan((Y[i + 1] - Y[i - 1]) / (X[i + 1] - X[i - 1]))

		if X[i + 1] - X[i - 1] < 0:
			THETA[i] += math.pi

	if X[1] == X[0]:
		if Y[1] > Y[0]:
			THETA[0] = math.pi / 2
		else:
			THETA[0] = 3 * math.pi / 2
	else:
		THETA[0] = math.atan((Y[1] - Y[0]) / (X[1] - X[0]))

	if X[-1] == X[len(Y) - 2]:
		if Y[1] > Y[0]:
			THETA[-1] = math.pi / 2
		else:
			THETA[-1] = 3 * math.pi / 2
	else:
		THETA[-1] = math.atan((Y[-1] - Y[len(Y) - 2]) / (X[-1] - X[len(Y) - 2]))

	return THETA


def genTraj(num_poses_per_segment=20):
	"""Generate a synthetic trajectory with multiple U-turns

	Args:
		num_poses_per_segment: Number of poses per trajectory segment (default: 20)
		                       Total poses will be approximately 6 * num_poses_per_segment

	Returns:
		Tuple of (X, Y, THETA) arrays representing the trajectory
	"""
	# Forward I
	num = num_poses_per_segment
	xSt = -5
	ySt = -8
	leng = 9.0
	step = float(leng) / num
	X1 = np.zeros(num)
	Y1 = np.zeros(num)
	X1[0] = xSt
	Y1[0] = ySt
	for i in range(1, num):
		X1[i] = X1[i - 1] + step
		Y1[i] = ySt

	# UTurn I
	rad = 2.5
	num = num_poses_per_segment
	xCen = X1[-1]
	yCen = Y1[-1] + rad
	thetas = np.linspace(-math.pi / 2, math.pi / 2, num)
	X2 = np.zeros(num)
	Y2 = np.zeros(num)
	for i, theta in enumerate(thetas):
		X2[i] = (xCen + rad * math.cos(theta))
		Y2[i] = (yCen + rad * math.sin(theta))

	# Backward I
	num = num_poses_per_segment
	leng = 10.0
	step = float(leng) / num
	xSt = X2[-1]
	ySt = Y2[-1]
	X3 = np.zeros(num)
	Y3 = np.zeros(num)
	X3[0] = xSt
	Y3[0] = ySt
	for i in range(1, num):
		X3[i] = X3[i - 1] - step
		Y3[i] = ySt

	# UTurn II
	rad = 2.6
	num = num_poses_per_segment
	xCen = X3[-1]
	yCen = Y3[-1] - rad
	thetas = np.linspace(math.pi / 2, 3 * math.pi / 2, num)
	X4 = np.zeros(num)
	Y4 = np.zeros(num)
	for i, theta in enumerate(thetas):
		X4[i] = (xCen + rad * math.cos(theta))
		Y4[i] = (yCen + rad * math.sin(theta))

	# Forward II
	num = num_poses_per_segment
	leng = 11.0
	step = float(leng) / num
	xSt = X4[-1]
	ySt = Y4[-1]
	X5 = np.zeros(num)
	Y5 = np.zeros(num)
	X5[0] = xSt
	Y5[0] = ySt
	for i in range(1, num):
		X5[i] = X5[i - 1] + step
		Y5[i] = ySt

	# UTurn III
	rad = 2.7
	num = num_poses_per_segment
	xCen = X5[-1]
	yCen = Y5[-1] + rad
	thetas = np.linspace(-math.pi / 2, math.pi / 2, num)
	X6 = np.zeros(num)
	Y6 = np.zeros(num)
	for i, theta in enumerate(thetas):
		X6[i] = (xCen + rad * math.cos(theta))
		Y6[i] = (yCen + rad * math.sin(theta))

	# Assemble
	X = np.concatenate([X1, X2, X3, X4, X5, X6])
	Y = np.concatenate([Y1, Y2, Y3, Y4, Y5, Y6])
	THETA = np.array(getTheta(X, Y))

	return (X, Y, THETA)


def genTrajSquareSpiral(num_poses_per_segment=20):
	"""Generate a square spiral trajectory (escargot)

	Creates a trajectory that spirals outward in a square pattern, with each
	side getting progressively longer. This creates natural loop closures as
	the robot passes near previous locations.

	Args:
		num_poses_per_segment: Number of poses per side segment (default: 20)

	Returns:
		Tuple of (X, Y, THETA) arrays representing the trajectory
	"""
	X_all = []
	Y_all = []

	x, y = 0.0, 0.0  # Start at origin
	direction = 0  # 0=right, 1=up, 2=left, 3=down

	# Generate spiral: each pair of sides increases in length
	# More compact spiral with smaller increments
	num_spirals = 4  # Number of complete square loops
	for spiral in range(num_spirals):
		for pair in range(2):  # Two sides per length increment
			# Smaller initial size and increment for tighter spiral
			side_length = 2.0 + spiral * 1.0  # Smaller increment
			step = side_length / num_poses_per_segment

			for i in range(num_poses_per_segment):
				X_all.append(x)
				Y_all.append(y)

				# Move in current direction
				if direction == 0:  # Right
					x += step
				elif direction == 1:  # Up
					y += step
				elif direction == 2:  # Left
					x -= step
				elif direction == 3:  # Down
					y -= step

			# Turn 90 degrees for next side
			direction = (direction + 1) % 4

	# Convert to numpy arrays
	X = np.array(X_all)
	Y = np.array(Y_all)
	THETA = np.array(getTheta(X, Y))

	return (X, Y, THETA)


def addNoise(X, Y, THETA, seed=None, noise_sigma=0.03):
	"""Add Gaussian noise to trajectory odometry

	Args:
		X: X coordinates
		Y: Y coordinates
		THETA: Orientations
		seed: Random seed for reproducibility (default: None for random noise)
		noise_sigma: Standard deviation of Gaussian noise (default: 0.03)

	Returns:
		Tuple of (xN, yN, tN) noisy trajectory
	"""
	if seed is not None:
		np.random.seed(seed)

	xN = np.zeros(len(X))
	yN = np.zeros(len(Y))
	tN = np.zeros(len(THETA))
	xN[0] = X[0]
	yN[0] = Y[0]
	tN[0] = THETA[0]

	for i in range(1, len(X)):
		# Get T2_1
		p1 = (X[i - 1], Y[i - 1], THETA[i - 1])
		p2 = (X[i], Y[i], THETA[i])
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

		# Add noise
		if i < 5:
			xNoise = 0
			yNoise = 0
			tNoise = 0
		else:
			xNoise = np.random.normal(0, noise_sigma)
			yNoise = np.random.normal(0, noise_sigma)
			tNoise = np.random.normal(0, noise_sigma)
		del_xN = del_x + xNoise
		del_yN = del_y + yNoise
		del_thetaN = del_theta + tNoise

		# Convert to T2_1'
		T2_1N = np.array([[math.cos(del_thetaN), -math.sin(del_thetaN), del_xN],
		                  [math.sin(del_thetaN), math.cos(del_thetaN), del_yN],
		                  [0, 0, 1]])

		# Get T2_w' = T1_w' . T2_1'
		p1 = (xN[i - 1], yN[i - 1], tN[i - 1])
		T1_wN = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]],
		                  [math.sin(p1[2]), math.cos(p1[2]), p1[1]],
		                  [0, 0, 1]])
		T2_wN = np.dot(T1_wN, T2_1N)

		# Get x2', y2', theta2'
		x2N = T2_wN[0][2]
		y2N = T2_wN[1][2]
		theta2N = math.atan2(T2_wN[1, 0], T2_wN[0, 0])

		xN[i] = x2N
		yN[i] = y2N
		tN[i] = theta2N

	return (xN, yN, tN)
