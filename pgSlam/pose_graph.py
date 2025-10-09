"""
Pose graph optimization module
"""
import os


def optimize():
	"""Run g2o optimizer on the pose graph"""
	cmd = "g2o -o opt.g2o noise.g2o"
	os.system(cmd)
