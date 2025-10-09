# Pose Graph SLAM - Modular Version

This is a refactored version of the pose graph SLAM demonstration, split into modular components.

## Structure

```
pgSlam/
├── trajectory.py      - Trajectory generation and noise addition
├── visualization.py   - Plotting and visualization functions
├── g2o_io.py         - G2O file reading and writing
├── pose_graph.py     - Pose graph optimization wrapper
├── main.py           - Main script (entry point)
├── pgSlam.py         - Original monolithic script
└── README.md         - This file
```

## Modules

### trajectory.py
- `genTraj()`: Generates a synthetic trajectory with U-turns
- `addNoise()`: Adds Gaussian noise to odometry measurements
- `getTheta()`: Calculates orientation angles from positions

### visualization.py
- `draw()`: Draws a single trajectory
- `drawTwo()`: Compares ground truth and optimized trajectories
- `drawThree()`: Shows ground truth, optimized, and noisy trajectories

### g2o_io.py
- `writeOdom()`: Writes odometry edges to g2o file
- `writeLoop()`: Writes loop closure constraints to g2o file
- `readG2o()`: Reads optimized poses from g2o file

### pose_graph.py
- `optimize()`: Runs g2o optimizer on the pose graph

## Usage

### Basic usage

Run with default parameters:
```bash
python main.py
```

### Advanced usage with custom information matrices

Adjust odometry edge information matrix:
```bash
python main.py --odom-info 1000.0
```

Adjust loop closure edge information matrix:
```bash
python main.py --loop-info 500.0
```

Adjust both:
```bash
python main.py --odom-info 300.0 --loop-info 900.0
```

Run without visualization (faster):
```bash
python main.py --no-viz
```

Get help:
```bash
python main.py --help
```

### Original monolithic version

You can still run the original version:
```bash
python pgSlam.py
```

## Dependencies

- numpy
- matplotlib
- g2o (command-line optimizer)
