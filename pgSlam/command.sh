#!/bin/bash

# ==============================================================================
# POSE GRAPH SLAM - COMMAND REFERENCE
# ==============================================================================

# ------------------------------------------------------------------------------
# 2D SLAM (main.py)
# ------------------------------------------------------------------------------

# Basic 2D optimization
python main.py

# Square spiral trajectory (escargot)
python main.py --trajectory-type square-spiral --seed 42

# U-turn trajectory (default)
python main.py --trajectory-type uturn --seed 42

# More poses with denser loop closures
python main.py --num-poses 30 --loop-sampling 1

# Fewer poses with no loop closures
python main.py --num-poses 10 --loop-sampling 0

# Higher odometry noise amplitude
python main.py --noise-sigma 0.1 --seed 42

# Lower odometry noise amplitude (more accurate odometry)
python main.py --noise-sigma 0.01 --seed 42

# Loop closures from noisy poses (realistic drift scenario)
python main.py --loop-from-noisy --seed 42

# Loop closures with observation noise (imperfect place recognition)
python main.py --loop-noise-sigma 0.05 --seed 42

# Combined: noisy loops with observation noise
python main.py --loop-from-noisy --loop-noise-sigma 0.05 --seed 42

# Reproducible experiment with fixed seed
python main.py --seed 42

# Custom information matrices
python main.py --odom-info 300.0 --loop-info 900.0

# Complete configuration
python main.py --num-poses 25 --loop-sampling 2 --noise-sigma 0.05 --loop-noise-sigma 0.02 --seed 123 --odom-info 500.0 --loop-info 700.0

# Without visualization (faster)
python main.py --no-viz --seed 42

# ------------------------------------------------------------------------------
# 3D SLAM (main_3d.py)
# ------------------------------------------------------------------------------

# Basic 3D optimization (U-turn trajectory with altitude variations)
python main_3d.py

# Helical spiral trajectory
python main_3d.py --trajectory-type helix

# U-turn trajectory (default)
python main_3d.py --trajectory-type uturn

# More poses with denser loop closures
python main_3d.py --num-poses 30 --loop-sampling 1

# Fewer poses with sparser loop closures
python main_3d.py --num-poses 15 --loop-sampling 5

# No loop closures (odometry-only optimization)
python main_3d.py --loop-sampling 0

# Higher odometry noise amplitude
python main_3d.py --noise-sigma 0.1 --seed 42

# Lower odometry noise amplitude (more accurate odometry)
python main_3d.py --noise-sigma 0.01 --seed 42

# Loop closures from noisy poses (realistic drift scenario)
python main_3d.py --loop-from-noisy --seed 42

# Loop closures with observation noise (imperfect place recognition)
python main_3d.py --loop-noise-sigma 0.05 --seed 42

# Combined: noisy loops with observation noise
python main_3d.py --loop-from-noisy --loop-noise-sigma 0.05 --seed 42

# Reproducible experiment with fixed seed
python main_3d.py --seed 42

# Custom information matrices
python main_3d.py --odom-info 300.0 --loop-info 900.0

# Complete 3D configuration
python main_3d.py --trajectory-type helix --num-poses 25 --loop-sampling 2 --noise-sigma 0.05 --loop-noise-sigma 0.02 --seed 123 --odom-info 500.0 --loop-info 700.0

# Without visualization (faster)
python main_3d.py --no-viz --seed 42

# Helix with high noise and loop closures
python main_3d.py --trajectory-type helix --noise-sigma 0.08 --loop-sampling 1 --seed 42

# ------------------------------------------------------------------------------
# REAL SCENARIO OPTIMIZATION
# ------------------------------------------------------------------------------

# Full optimization with all visualizations and metrics
python optimize_scenario_complete.py

# With loop sampling (use 1 loop out of 3)
python optimize_scenario_complete.py --loop-sampling 3

# Without loop closures (odometry-only optimization)
python optimize_scenario_complete.py --loop-sampling 0

# Custom information matrices
python optimize_scenario_complete.py --odom-info 300.0 --loop-info 900.0

# More g2o iterations
python optimize_scenario_complete.py --iterations 100

# Skip visualizations (faster)
python optimize_scenario_complete.py --no-viz

# ------------------------------------------------------------------------------
# SCENARIO DATA PREPARATION
# ------------------------------------------------------------------------------

# Generate noisy scenario (first 400 poses with loops)
python generate_scenario_noisy.py

# Generate reference scenario (poses [304-392] = copies of [0-88])
python generate_scenario_ref.py

# Visualize all poses
python visualize_all_poses.py --num-poses 400

# Visualize with loop closures between segments
python visualize_loops_segments.py --num-poses 400

# Compare noisy vs reference scenarios
python visualize_scenarios.py

# ------------------------------------------------------------------------------
# BOW LOOP CLOSURE DETECTION
# ------------------------------------------------------------------------------

# Build vocabulary with default settings (k=1000 visual words)
python build_bow_vocabulary.py

# Recommended: k=500 for ~1000-2000 images
python build_bow_vocabulary.py --k 500 --features-per-image 300

# Fast test with fewer images
python build_bow_vocabulary.py --k 300 --max-images 300 --features-per-image 200

# High precision vocabulary
python build_bow_vocabulary.py --k 1000 --features-per-image 500

# Detect loops with BoW
python detect_loops_bow.py --similarity-threshold 0.75 --min-temporal-distance 50 --show-pairs --max-pairs 20

# With similarity matrix visualization
python detect_loops_bow.py --similarity-threshold 0.75 --visualize --show-pairs

# ------------------------------------------------------------------------------
# DINOV2 LOOP CLOSURE DETECTION
# ------------------------------------------------------------------------------

# Extract DINOv2 features (small model, fastest)
python build_dinov2_features.py

# Recommended: base model (balanced speed/quality)
python build_dinov2_features.py --model-size base

# High quality: large model (best accuracy)
python build_dinov2_features.py --model-size large

# Test on subset of images
python build_dinov2_features.py --max-images 200 --model-size base

# Detect loops with DINOv2 (recommended settings)
python detect_loops_dinov2.py --similarity-threshold 0.85 --min-temporal-distance 50

# With all visualizations
python detect_loops_dinov2.py --similarity-threshold 0.85 --min-temporal-distance 50 --visualize --plot-distribution --show-pairs --max-pairs 20

# More conservative (fewer false positives)
python detect_loops_dinov2.py --similarity-threshold 0.90 --min-temporal-distance 100

# Visualize DINOv2 loop closures (2D)
python visualize_loops_dinov2.py --statistics

# 3D visualization
python visualize_loops_dinov2.py --3d --statistics

# ------------------------------------------------------------------------------
# PARAMETER REFERENCE
# ------------------------------------------------------------------------------

# Common parameters for main.py and main_3d.py:
# --trajectory-type <type>    : Trajectory type (2D: uturn, square-spiral | 3D: uturn, helix)
# --num-poses <value>         : Number of poses per trajectory segment (default: 20)
# --odom-info <value>         : Information matrix value for odometry edges (default: 500.0)
# --loop-info <value>         : Information matrix value for loop closure edges (default: 700.0)
# --loop-sampling <value>     : Sample every Nth pose for loop closures (default: 2, 0=disable)
# --loop-from-noisy           : Use noisy poses for loop closure edges (default: use ground truth)
# --loop-noise-sigma <value>  : Standard deviation of Gaussian noise for loop observations (default: 0.0)
# --noise-sigma <value>       : Standard deviation of Gaussian noise for odometry (default: 0.03)
# --seed <value>              : Random seed for noise generation (default: None)
# --no-viz                    : Disable visualization (faster execution)
