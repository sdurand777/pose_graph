"""
Pose graph optimization module for 3D SLAM
"""
import os


def optimize3D():
	"""Run g2o optimizer on the 3D pose graph"""
	cmd = "g2o -o opt_3d.g2o noise_3d.g2o"
	os.system(cmd)
