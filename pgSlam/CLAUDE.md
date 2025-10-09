# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a modular pose graph SLAM (Simultaneous Localization and Mapping) demonstration that generates synthetic trajectory data, adds noise to simulate odometry drift, applies loop closure constraints, and optimizes the pose graph using the g2o optimizer.

## Running the Code

### Main entry point
```bash
python main.py
```

### Key parameters

**Trajectory & Optimization:**
- `--trajectory-type <type>`: Type of trajectory to generate (choices: `uturn`, `square-spiral`, default: `uturn`)
- `--num-poses <value>`: Number of poses per trajectory segment (default: 20)
- `--odom-info <value>`: Information matrix value for odometry edges (default: 500.0)
- `--loop-info <value>`: Information matrix value for loop closure edges (default: 700.0)

**Loop Closures:**
- `--loop-sampling <value>`: Sample every Nth pose for loop closures (default: 2). Higher = fewer loops. Set to 0 to disable loop closures
- `--loop-from-noisy`: Use noisy poses instead of ground truth for loop closure edges (default: use ground truth)
- `--loop-noise-sigma <value>`: Standard deviation of Gaussian noise for loop observations (default: 0.0)

**Noise & Reproducibility:**
- `--noise-sigma <value>`: Standard deviation of Gaussian noise for odometry (default: 0.03)
- `--seed <value>`: Random seed for noise generation (default: None). Use for reproducible experiments

**Visualization:**
- `--no-viz`: Disable visualization (faster execution)

Examples with custom parameters:
```bash
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
```

### Alternative versions
The original monolithic version can still be run:
```bash
python pgSlam.py
```

## Architecture

### Core Pipeline (main.py)
The system follows a five-stage pipeline:

1. **Trajectory Generation** → 2. **Noise Addition** → 3. **Optimization** → 4. **Metrics Evaluation** → 5. **Visualization**

Each stage is handled by a separate module to maintain separation of concerns. After optimization, the system automatically computes error metrics and displays improvement percentages.

### Module Responsibilities

**trajectory.py**: Synthetic data generation
- `genTraj()`: Creates a U-turn trajectory with 6 segments (3 forward passes, 3 U-turns)
- `genTrajSquareSpiral()`: Creates a square spiral (escargot) trajectory with expanding sides
- `addNoise()`: Simulates odometry drift by adding Gaussian noise to pose-to-pose transformations
- `getTheta()`: Computes orientation angles from (x,y) positions

**g2o_io.py**: G2O file format handling
- `writeOdom()`: Creates VERTEX_SE2 and sequential EDGE_SE2 entries for odometry
- `writeLoop()`: Adds EDGE_SE2 entries for loop closures between parallel trajectory segments
- `readG2o()`: Parses optimized VERTEX_SE2 positions from g2o output

**pose_graph.py**: Optimization wrapper
- `optimize()`: Executes the g2o command-line optimizer (requires g2o to be installed)

**visualization.py**: Matplotlib-based plotting
- `draw()`: Single trajectory with orientation arrows and optional loop closures (cyan dashed lines)
- `drawTwo()`: Ground truth vs. optimized comparison with optional loop closures
- `drawThree()`: Shows ground truth, optimized, and noisy trajectories with optional loop closures
- `plotErrors()`: Two-panel plot showing position and orientation errors per pose for noisy and optimized trajectories

All visualization functions now accept a `loop_pairs` parameter to display loop closure constraints as cyan dashed lines overlaid on the trajectory plots.

**metrics.py**: Trajectory error evaluation
- `compute_pose_errors()`: Computes per-pose position and orientation errors, plus statistics (RMSE, mean, max)
- `print_metrics_comparison()`: Formatted console output comparing noisy vs. optimized trajectory metrics

### Key Transformations

The codebase uses SE(2) transformations (x, y, theta) represented as 3x3 homogeneous matrices:
```
T = [cos(θ)  -sin(θ)  x]
    [sin(θ)   cos(θ)  y]
    [0        0       1]
```

Relative transformations between poses are computed as: `T2_1 = inv(T1_w) * T2_w`

### Information Matrices

The g2o file format uses 3x3 information matrices for SE(2) edges, stored as upper triangular values:
`info_xx info_xy info_xθ info_yy info_yθ info_θθ`

Higher information values indicate higher confidence. Loop closures typically have higher information than odometry.

### Generated Files

- `noise.g2o`: Input pose graph with noisy odometry and loop closures
- `opt.g2o`: Output from g2o optimizer with corrected vertex positions

### Performance Metrics

The system automatically computes and displays comprehensive error metrics:

**Position Metrics:**
- RMSE (Root Mean Square Error) in meters
- Mean position error
- Maximum position error

**Orientation Metrics:**
- RMSE in radians
- Mean orientation error
- Maximum orientation error

**Improvement Percentage:** Shows how much the optimization reduced each metric

**Visualization:** Per-pose error plots show:
- Position errors (meters) for each pose index
- Orientation errors (degrees) for each pose index
- RMSE reference lines for both trajectories
- Green lines = noisy trajectory errors
- Blue lines = optimized trajectory errors

## Dependencies

- **numpy**: Numerical operations and transformations
- **matplotlib**: Trajectory visualization
- **g2o**: External command-line graph optimizer (must be installed separately)

Check g2o installation: `which g2o`

## Development Notes

### Coordinate System
All poses use a 2D world frame with (x, y, θ) representation where θ is in radians.

### Trajectory Types

**U-Turn (`--trajectory-type uturn`, default):**
- 6 segments: 3 forward passes with U-turns in between
- Natural loop closures as parallel paths revisit similar locations
- Good for testing optimization with known geometry

**Square Spiral (`--trajectory-type square-spiral`):**
- Spiral outward in a square pattern with increasing side lengths
- Robot repeatedly passes near previous locations
- Creates many natural loop closure opportunities
- Excellent for testing robustness with complex trajectories

### Loop Closure Strategy

Loop closures are **automatically detected** using proximity-based matching:
- Finds poses that are close in space (< 2.5m) but far in time (> 30 poses apart)
- The `--loop-sampling` parameter controls sampling rate:
  - `--loop-sampling 1`: Check every pose (maximum loops)
  - `--loop-sampling 2`: Check every 2nd pose (default)
  - `--loop-sampling 5`: Check every 5th pose (sparse loops)
  - `--loop-sampling 0`: No loop closures

This adaptive approach works for all trajectory types. Loop closures are visualized as cyan dashed lines connecting corresponding poses.

**Note:** Parameters are tuned for compact trajectories. For the square spiral, the tighter spiral pattern (2m, 3m, 4m, 5m sides) ensures multiple loop closures as each turn passes near previous positions.

**Loop Closure Source (configurable):**

By default, loop closure edges are computed from the **ground truth trajectory**. This simulates perfect place recognition. You can control this behavior:

1. **`--loop-from-noisy` (flag)**: Use noisy poses instead of ground truth
   - Simulates loop closures detected from accumulated odometry
   - More realistic when place recognition is based on the robot's estimated position
   - Introduces inconsistencies in loop constraints

2. **`--loop-noise-sigma <value>`**: Add observation noise to loop measurements
   - Default: 0.0 (perfect observations)
   - Simulates imperfect sensor observations (e.g., vision-based place recognition errors)
   - Applied regardless of whether using ground truth or noisy poses as base

**Realistic scenarios:**
- Perfect loop closures: Default (ground truth, no noise)
- Visual SLAM: `--loop-noise-sigma 0.02` (small observation error)
- GPS-based loops: `--loop-noise-sigma 0.5` (larger sensor uncertainty)
- Drift-affected loops: `--loop-from-noisy` (odometry-based detection)

### Noise Model
Gaussian noise (configurable via `--noise-sigma`, default σ = 0.03) is applied to odometry measurements in the robot's local frame, then propagated through transformation composition. The first 5 poses are noise-free to establish a stable initial anchor.

**Noise amplitude guidelines:**
- `--noise-sigma 0.01`: Low noise (high-quality odometry like visual odometry)
- `--noise-sigma 0.03`: Default (typical wheel encoder noise)
- `--noise-sigma 0.05-0.10`: High noise (challenging scenarios, low-quality sensors)
- `--noise-sigma 0.15+`: Very high noise (extreme test cases)

### Reproducibility
Use the `--seed` parameter to fix the random seed for noise generation. This ensures identical noisy trajectories across multiple runs, which is essential for:
- Comparing different optimization parameters
- Debugging
- Creating reproducible research results

Example: `python main.py --seed 42` will always generate the same noisy trajectory.
