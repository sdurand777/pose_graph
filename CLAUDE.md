# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two main components:

1. **pgSlam/**: A comprehensive modular pose graph SLAM (Simultaneous Localization and Mapping) system
   - 2D and 3D trajectory optimization using g2o
   - Multiple loop closure detection methods (BoW, DINOv2)
   - Real scenario data processing
   - Synthetic trajectory generation with configurable noise
   - Automated metrics evaluation and visualization

2. **scripts_mapanything/**: MapAnything integration scripts for vision-based loop closure pose estimation

---

## Pose Graph SLAM (pgSlam/)

### Quick Start

```bash
# 2D pose graph optimization
cd pgSlam
python main.py

# 3D pose graph optimization
python main_3d.py

# Process real scenario data with full pipeline
python optimize_scenario_complete.py
```

### Common Commands

**2D Trajectory Optimization:**
```bash
# Basic with default U-turn trajectory
python main.py --seed 42

# Square spiral trajectory with custom parameters
python main.py --trajectory-type square-spiral --num-poses 30 --loop-sampling 1 --seed 42

# High noise scenario with noisy loop closures
python main.py --noise-sigma 0.1 --loop-from-noisy --loop-noise-sigma 0.05 --seed 42

# Custom information matrices
python main.py --odom-info 300.0 --loop-info 900.0

# No visualization (faster)
python main.py --no-viz
```

**3D Trajectory Optimization:**
```bash
# Basic 3D U-turn
python main_3d.py --seed 42

# Helical spiral trajectory
python main_3d.py --trajectory-type helix --num-poses 30 --seed 42

# Complete configuration
python main_3d.py --trajectory-type helix --num-poses 25 --loop-sampling 2 \
  --noise-sigma 0.05 --loop-noise-sigma 0.02 --seed 123 \
  --odom-info 500.0 --loop-info 700.0
```

**Vision-Based Loop Closure (BoW):**
```bash
# Build vocabulary (one-time)
python build_bow_vocabulary.py --k 500 --features-per-image 300

# Detect loops with visualization
python detect_loops_bow.py --similarity-threshold 0.75 --min-temporal-distance 50 --show-pairs --max-pairs 20

# Visualize detected loops on trajectory
python visualize_loops_segments.py --num-poses 400
```

**Vision-Based Loop Closure (DINOv2):**
```bash
# Extract features (one-time, requires GPU for speed)
python build_dinov2_features.py --model-size base

# Detect loops with all visualizations
python detect_loops_dinov2.py \
  --similarity-threshold 0.85 \
  --min-temporal-distance 50 \
  --visualize \
  --plot-distribution \
  --show-pairs \
  --max-pairs 20

# Visualize on trajectory
python visualize_loops_dinov2.py --statistics
```

**Real Scenario Processing:**
```bash
# Complete pipeline: generate scenarios, optimize, evaluate
python optimize_scenario_complete.py --loop-sampling 2

# Individual steps
python generate_scenario_noisy.py       # Generate noisy scenario
python generate_scenario_ref.py         # Generate reference (ground truth)
python optimize_scenario_complete.py    # Full optimization pipeline
python compute_scenario_metrics.py      # Compute metrics only
python visualize_scenario_optimization.py  # Visualize results only
```

---

## Architecture

### Core Pipeline (2D/3D SLAM)

**1. Trajectory Generation** → **2. Noise Addition** → **3. Optimization** → **4. Metrics** → **5. Visualization**

Each stage is modular and handled by separate files:

**Trajectory Generation (`trajectory.py`, `trajectory_3d.py`):**
- `genTraj()` / `genTraj3D()`: U-turn trajectory with parallel segments
- `genTrajSquareSpiral()` / `genTrajSpiralHelix3D()`: Spiral/helix trajectories
- `addNoise()` / `addNoise3D()`: Gaussian noise in local frame
- First 5 poses noise-free for stable anchor

**G2O I/O (`g2o_io.py`, `g2o_io_3d.py`, `g2o_io_scenario.py`):**
- `writeOdom()`: Creates VERTEX_SE2/SE3 and sequential EDGE entries
- `writeLoop()`: Adds loop closure EDGE entries
- `readG2o()`: Parses optimized vertex positions
- Format: 2D uses VERTEX_SE2/EDGE_SE2, 3D uses VERTEX_SE3:QUAT/EDGE_SE3:QUAT

**Optimization (`pose_graph.py`, `pose_graph_3d.py`):**
- `optimize()` / `optimize3D()`: Executes g2o command-line optimizer
- Requires g2o installed: check with `which g2o`

**Metrics (`metrics.py`, `metrics_3d.py`):**
- `compute_pose_errors()`: Position and orientation errors (RMSE, mean, max)
- `print_metrics_comparison()`: Formatted console output
- Position: Euclidean distance (2D or 3D)
- Orientation: Angular difference (2D) or geodesic distance on SO(3) (3D)

**Visualization (`visualization.py`, `visualization_3d.py`):**
- `draw()`: Single trajectory with optional loop closures (cyan dashed)
- `drawTwo()`: Ground truth vs optimized comparison
- `drawThree()`: Three-way comparison (GT, noisy, optimized)
- `plotErrors()`: Per-pose error plots (position and orientation)

### Loop Closure Detection

**Automatic Proximity-Based (default):**
- Finds poses close in space but far in time
- 2D: distance < 2.5m, temporal gap > 30 poses
- 3D: distance < 3.0m, temporal gap > 30 poses
- Controlled via `--loop-sampling` parameter (1=all, 2=every 2nd, 0=none)

**Bag of Words (BoW):**
- `build_bow_vocabulary.py`: K-Means clustering of ORB features → visual words
- `detect_loops_bow.py`: Histogram comparison using cosine similarity
- Typical threshold: 0.75-0.80
- Fast, requires vocabulary construction

**DINOv2 (Deep Learning):**
- `build_dinov2_features.py`: Extract global semantic features using Vision Transformer
- `detect_loops_dinov2.py`: Compute pairwise similarities
- Typical threshold: 0.85-0.90
- Better accuracy, no training required, GPU recommended

### Key Parameters

**Trajectory Configuration:**
- `--trajectory-type`: `uturn` (default) or `square-spiral` (2D), `uturn` or `helix` (3D)
- `--num-poses`: Poses per segment (default: 20)
- `--seed`: Random seed for reproducibility

**Noise Model:**
- `--noise-sigma`: Odometry noise std dev (default: 0.03)
  - 0.01: Low noise (high-quality odometry)
  - 0.03: Default (typical wheel encoder)
  - 0.05-0.10: High noise (challenging scenarios)
- Applied in robot's local frame, propagated through transformations

**Loop Closures:**
- `--loop-sampling`: Sample every Nth pose (0=disable, 1=all, 2=default)
- `--loop-from-noisy`: Use noisy poses instead of ground truth (realistic drift)
- `--loop-noise-sigma`: Observation noise for loop measurements (default: 0.0)

**Information Matrices:**
- `--odom-info`: Odometry confidence (default: 500.0)
- `--loop-info`: Loop closure confidence (default: 700.0)
- Higher values = higher confidence/stronger constraint

**Visualization:**
- `--no-viz`: Disable plots for faster execution

### Real Scenario Data Format

**Input Files (in `scenario/`):**
- `vertices_stan.txt`: Camera poses (tx, ty, tz, qx, qy, qz, qw) in c2w format
- `id_kf_2_idx_img.txt`: Mapping from keyframe IDs to image indices
- `loop_all.txt`: Loop closure pairs
- `imgs/`: Image sequence (e.g., img00000.jpg to img01197.jpg)

**Processing Pipeline:**
1. `add_loops_segments.py`: Annotate loops between trajectory segments
2. `generate_scenario_noisy.py`: Create noisy scenario (first 400 poses)
3. `generate_scenario_ref.py`: Create reference (segment 2 = copy of segment 1)
4. `optimize_scenario_complete.py`: Full pipeline with optimization and evaluation

**Key Technical Detail:**
- Input: camera-to-world (c2w) format
- Converted to world-to-camera (w2c) for visualization: `R_w2c = R_c2w.T`, `t_w2c = -R_w2c @ t_c2w`
- Loop edges computed from **reference poses** (perfect constraints)
- Odometry edges computed from **noisy poses** (sequential drift)

---

## MapAnything Scripts (scripts_mapanything/)

### Overview

Scripts for using the MapAnything model to estimate relative poses between loop closure pairs detected from images.

### Running Loop Pose Estimation

```bash
cd scripts_mapanything

# Basic estimation (no filtering)
./run_loop_estimation.sh basic

# Accurate mode (confidence threshold 0.6)
./run_loop_estimation.sh accurate

# Fast mode (confidence threshold 0.3)
./run_loop_estimation.sh fast

# Memory-efficient mode (low VRAM)
./run_loop_estimation.sh memory-save

# Apache license model (commercial use)
./run_loop_estimation.sh apache

# CPU mode (very slow, no GPU needed)
./run_loop_estimation.sh cpu

# Custom configuration with custom threshold
CONF_THRESHOLD=0.7 ./run_loop_estimation.sh custom

# Override default threshold for any mode
CONF_THRESHOLD=1.2 ./run_loop_estimation.sh apache

# With environment variables
IMAGE_FOLDER=/path/to/images OUTPUT_CSV=results.csv ./run_loop_estimation.sh accurate

# Batch processing and testing
BATCH_SIZE=4 MAX_PAIRS=10 ./run_loop_estimation.sh basic

# ===== NEW: Automatic filtering of validated loops =====
# Estimation + automatic filtering with same threshold
FILTER_LOOPS=1 ./run_loop_estimation.sh accurate

# Estimation + filtering with custom threshold
FILTER_LOOPS=1 CONF_THRESHOLD=1.2 ./run_loop_estimation.sh apache

# Different thresholds for estimation and filtering
FILTER_LOOPS=1 CONF_THRESHOLD=0.8 MIN_CONFIDENCE=1.2 ./run_loop_estimation.sh accurate

# Custom output folder for validated images
FILTER_LOOPS=1 OUTPUT_FOLDER=good_loops ./run_loop_estimation.sh accurate
```

### Configuration

**Environment Variables:**

*Estimation Parameters:*
- `IMAGE_FOLDER`: Directory containing query_*.jpg and loop_*.jpg pairs (required)
- `OUTPUT_CSV`: Output file (default: loop_poses.csv)
- `CONF_THRESHOLD`: **Confidence threshold (0.0-1.0), OVERRIDES mode default**
- `USE_CPU`: Force CPU mode (export USE_CPU=1)
- `BATCH_SIZE`: Number of pairs per batch (default: 1)
- `MAX_PAIRS`: Limit pairs for testing (e.g., 10)
- `MODEL`: Model to use (default: facebook/map-anything)

*NEW - Automatic Filtering Parameters:*
- `FILTER_LOOPS`: Enable automatic filtering after estimation (export FILTER_LOOPS=1)
- `OUTPUT_FOLDER`: Folder for validated images (default: validated_loops)
- `MIN_CONFIDENCE`: Filtering threshold, overrides CONF_THRESHOLD for filtering only
- `LIST_FILE`: Text file listing validated loops (default: validated_loops.txt)

**Modes:**
- `basic`: No threshold by default, standard inference
- `fast`: Threshold 0.3 (default), keeps most loops
- `accurate`: Threshold 0.6 (default), strict filtering
- `memory-save`: Threshold 0.5 (default), memory-efficient inference for large images
- `apache`: Threshold 0.5 (default), Apache 2.0 licensed model (commercial friendly)
- `cpu`: Threshold 0.5 (default), force CPU (10-50x slower than GPU)
- `custom`: Threshold 0.5 (default), user-defined configuration

**Note:** `CONF_THRESHOLD` always takes priority over mode defaults. For example:
- `CONF_THRESHOLD=1.2 ./run_loop_estimation.sh apache` uses 1.2, not 0.5
- `./run_loop_estimation.sh accurate` uses 0.6 (mode default)

### Python Script

Direct invocation:
```bash
python scripts/estimate_loop_poses.py \
  --image_folder /path/to/images \
  --output loop_poses.csv \
  --confidence_threshold 0.6 \
  --batch_size 1
```

**Key Parameters:**
- `--image_folder`: Directory with image pairs
- `--output`: Output CSV file
- `--confidence_threshold`: Minimum confidence (0.0-1.0, higher = stricter)
- `--cpu`: Force CPU mode
- `--apache`: Use Apache license model
- `--memory_efficient`: Reduce VRAM usage
- `--batch_size`: Process multiple pairs simultaneously
- `--max_pairs`: Limit number of pairs (testing)

**Output Format (CSV):**
```
pair_id,query_image,loop_image,tx,ty,tz,r11,r12,r13,r21,r22,r23,r31,r32,r33,conf_query,conf_loop,conf_mean,is_good_loop,processing_time_s
2,query_2.jpg,loop_2.jpg,-0.60,0.34,0.23,0.98,0.15,0.03,-0.15,0.99,0.03,-0.03,-0.04,1.00,1.00,1.00,1.00,True,0.62
```

**Automatic Filtering Outputs:**

When `FILTER_LOOPS=1` is used, three outputs are generated:

1. **loop_poses.csv** - Complete estimation results (all pairs)
2. **validated_loops.txt** - Human-readable list with confidence scores, sorted by confidence:
```
# ======================================================================
# Loops Validées - Liste avec Confiance Moyenne
# ======================================================================
# Total loops validées: 144
#
# Format: pair_id | query_image -> loop_image | confiance_moyenne
# ======================================================================

Loop   2 | query_2.jpg              -> loop_2.jpg                | Conf: 3.6596
Loop  34 | query_34.jpg             -> loop_34.jpg               | Conf: 1.8218
Loop  28 | query_28.jpg             -> loop_28.jpg               | Conf: 1.6950
...

# ======================================================================
# Statistiques
# ======================================================================
# Confiance minimum:  1.0000
# Confiance maximum:  3.6596
# Confiance moyenne:  1.5761
# ======================================================================
```
3. **validated_loops/** - Directory containing only query/loop image pairs that passed the confidence threshold

### Automatic Loop Filtering

**How It Works:**

The automatic filtering feature (`FILTER_LOOPS=1`) processes estimation results to:
1. Read the CSV output with confidence scores
2. Filter loops based on `conf_mean` threshold
3. Copy validated query/loop image pairs to a separate folder
4. Generate a sorted, human-readable list of validated loops

**Threshold Priority (for filtering):**
1. `MIN_CONFIDENCE` (if set) - allows different threshold for filtering than estimation
2. `CONF_THRESHOLD` (if set) - same threshold for both estimation and filtering
3. Mode default threshold (e.g., 0.6 for accurate mode)
4. 0.0 (keeps all loops)

**Use Cases:**

```bash
# Quick validation: estimate and filter in one step
FILTER_LOOPS=1 ./run_loop_estimation.sh accurate

# Strict filtering: only keep high-confidence loops
FILTER_LOOPS=1 CONF_THRESHOLD=1.5 ./run_loop_estimation.sh apache

# Two-stage: permissive estimation, strict filtering
FILTER_LOOPS=1 CONF_THRESHOLD=0.5 MIN_CONFIDENCE=1.2 ./run_loop_estimation.sh accurate
# Estimates with 0.5 threshold, but only copies images with confidence >= 1.2

# Organize results: separate folders for different thresholds
FILTER_LOOPS=1 MIN_CONFIDENCE=1.0 OUTPUT_FOLDER=loops_conf_1.0 ./run_loop_estimation.sh accurate
FILTER_LOOPS=1 MIN_CONFIDENCE=1.5 OUTPUT_FOLDER=loops_conf_1.5 ./run_loop_estimation.sh accurate
```

**Python Script for Manual Filtering:**

If you already have estimation results and want to filter them later:
```bash
python scripts/filter_validated_loops.py \
  --csv loop_poses.csv \
  --image_folder /path/to/images \
  --output_folder validated_loops \
  --min_confidence 1.2 \
  --list_file validated_loops.txt
```

### Demo Scripts

The `scripts/` directory contains various demo scripts:

**Loop Closure Scripts:**
- `estimate_loop_poses.py`: Main script for loop pose estimation with MapAnything
- `filter_validated_loops.py`: Filter and organize validated loops from CSV results

**Monocular Demos:**
- `demo_mono_simple.py`: Basic monocular depth estimation
- `demo_mono_calibration.py`: Monocular with camera calibration

**Stereo Demos:**
- `demo_stereo_simple.py`: Basic stereo depth estimation
- `demo_stereo_calibration.py`: Stereo with calibration

**XML/COLMAP Integration:**
- `demo_xml_calibration.py`: Process with XML calibration files
- `demo_xml_calibration_fixed.py`: Fixed version for specific datasets
- `demo_colmap.py`: COLMAP format integration

**Depth Video Generation:**
- `generate_depth_video.py`: Generate depth videos from image sequences
- `generate_depth_video_safe.py`: Safe mode with error handling
- `generate_depth_video_xml.py`: With XML calibration

**Other:**
- `demo_images_only_inference.py`: Image-only inference without pose
- `gradio_app.py`: Interactive Gradio web interface
- `train.py`: Model training script
- `profile_dataloading.py`: Performance profiling

---

## Development Guidelines

### G2O File Format

**2D (SE2):**
```
VERTEX_SE2 id x y theta
EDGE_SE2 id1 id2 dx dy dtheta info_xx info_xy info_xt info_yy info_yt info_tt
```

**3D (SE3 with quaternions):**
```
VERTEX_SE3:QUAT id x y z qx qy qz qw
EDGE_SE3:QUAT id1 id2 dx dy dz dqx dqy dqz dqw info_11 info_12 ... info_66 (21 values)
```

### Coordinate Systems

**2D:** (x, y, θ) where θ is in radians
**3D:** (x, y, z, qx, qy, qz, qw) where quaternion is [x, y, z, w] format (scipy convention)

**Transformations:**
- 2D: 3×3 homogeneous matrices for SE(2)
- 3D: 4×4 homogeneous matrices for SE(3)
- Relative transform: `T2_1 = inv(T1_w) @ T2_w`

### Information Matrices

Upper triangular values representing confidence:
- **2D:** 3×3 → 6 values (x, y, θ)
- **3D:** 6×6 → 21 values (x, y, z, roll, pitch, yaw)

Higher information = higher confidence = stronger constraint

Typical values:
- Odometry: 500.0 (moderate confidence)
- Loop closures: 700.0 (higher confidence)
- Noisy measurements: 300.0 (lower confidence)

### Quaternion Conventions

- **scipy.spatial.transform.Rotation:** [x, y, z, w] order
- **g2o VERTEX_SE3:QUAT:** qx, qy, qz, qw order
- Conversion functions in `trajectory_3d.py`:
  - `quaternion_from_euler(roll, pitch, yaw)`: Euler → quaternion
  - `euler_from_quaternion(qx, qy, qz, qw)`: quaternion → Euler

### Loop Closure Integration

**For synthetic trajectories:**
- Automatically detected by proximity + temporal constraint
- Configurable via `--loop-sampling`

**For BoW/DINOv2:**
1. Detect loops: get image pairs with similarity scores
2. Map image indices to pose indices using `id_kf_2_idx_img.txt`
3. Compute relative pose (visual odometry, ICP, or MapAnything)
4. Add EDGE_SE3:QUAT to g2o file
5. Weight information matrix by similarity: `info = 700.0 * (sim / threshold)^2`

### Performance Optimization

**For large datasets:**
- Use `--no-viz` to skip visualization
- Reduce `--num-poses` for testing
- Increase `--loop-sampling` to reduce loop count
- Use DINOv2 `small` model instead of `base`/`large`
- Process scenarios in chunks

**For accuracy:**
- Use reproducible seeds: `--seed 42`
- Increase loop closures: `--loop-sampling 1`
- Use stricter thresholds: `--similarity-threshold 0.90` (DINOv2)
- Tune information matrices: higher `--loop-info` pulls trajectory more strongly

---

## Dependencies

**Core (pose graph SLAM):**
- numpy: Numerical operations
- matplotlib: Visualization
- scipy: Transformations (3D only)
- g2o: Graph optimizer (external, must be installed)

**BoW loop closure:**
- opencv-python (cv2): ORB features and matching
- scikit-learn: K-Means clustering

**DINOv2 loop closure:**
- torch: PyTorch framework
- torchvision: Image preprocessing
- opencv-python: Image loading
- PIL (Pillow): Image handling
- tqdm: Progress bars

**MapAnything scripts:**
- torch, torchvision: Deep learning inference
- transformers: Hugging Face model loading
- opencv-python: Image processing
- pandas: CSV output handling

**Check installations:**
```bash
# G2O optimizer
which g2o

# Python packages
pip install numpy matplotlib scipy opencv-python scikit-learn torch torchvision pillow tqdm pandas
```

---

## Troubleshooting

**G2O not found:**
- Install: `sudo apt install ros-kinetic-libg2o` (ROS version) or build from source
- Verify: `which g2o` should return a path
- 3D requires SE3 support: check with `g2o --help`

**Too many/few loop closures:**
- Too many: Increase similarity threshold or `--loop-sampling`
- Too few: Decrease similarity threshold or `--loop-sampling`, verify trajectory has revisits

**Out of memory (GPU):**
- Use smaller DINOv2 model: `--model-size small`
- Enable memory-efficient mode: `--memory_efficient` (MapAnything)
- Reduce batch size: `--batch_size 1`
- Use CPU: `--cpu` (much slower)

**Poor optimization results:**
- Check loop closures are valid: visualize with `visualize_loops_dinov2.py`
- Tune information matrices: increase `--loop-info` relative to `--odom-info`
- Verify ground truth alignment in reference scenarios
- Increase loop closure density: decrease `--loop-sampling`

**Reproducibility issues:**
- Always use `--seed` parameter: `--seed 42`
- Same seed → same noise → same trajectory

**MapAnything confidence threshold issues:**
- CONF_THRESHOLD not respected: Make sure to use `CONF_THRESHOLD=X.X` before the command
- Mode default overriding custom threshold: Bug fixed - now `CONF_THRESHOLD` always takes priority
- Filtering with different threshold: Use `MIN_CONFIDENCE` for filtering-only threshold
- No loops after filtering: Lower `MIN_CONFIDENCE` or check `conf_mean` distribution in CSV
- All loops kept despite threshold: Verify CSV has `conf_mean` column and check threshold value

**MapAnything installation issues:**
- Module not found: Activate virtualenv with `source mapanything_env/bin/activate`
- PyTorch not installed: Run `./install_mapanything.sh venv` to install all dependencies
- CUDA not available: Script auto-detects GPU; use `USE_CPU=1` for CPU-only mode

---

## File Structure Summary

```
pose_graph/
├── README.md                      # High-level repository overview
├── CLAUDE.md                      # This file
├── pgSlam/                        # Pose graph SLAM implementation
│   ├── README.md                  # 2D SLAM documentation
│   ├── README_3D.md               # 3D SLAM documentation (French)
│   ├── main.py                    # 2D entry point
│   ├── main_3d.py                 # 3D entry point
│   ├── trajectory.py              # 2D trajectory generation
│   ├── trajectory_3d.py           # 3D trajectory generation
│   ├── g2o_io.py                  # 2D g2o file I/O
│   ├── g2o_io_3d.py               # 3D g2o file I/O
│   ├── g2o_io_scenario.py         # Real scenario g2o conversion
│   ├── pose_graph.py              # 2D optimization wrapper
│   ├── pose_graph_3d.py           # 3D optimization wrapper
│   ├── visualization.py           # 2D plotting
│   ├── visualization_3d.py        # 3D plotting
│   ├── metrics.py                 # 2D error metrics
│   ├── metrics_3d.py              # 3D error metrics
│   ├── build_bow_vocabulary.py    # BoW vocabulary construction
│   ├── detect_loops_bow.py        # BoW loop detection
│   ├── build_dinov2_features.py   # DINOv2 feature extraction
│   ├── detect_loops_dinov2.py     # DINOv2 loop detection
│   ├── visualize_loops_dinov2.py  # DINOv2 loop visualization
│   ├── optimize_scenario_complete.py  # Complete real scenario pipeline
│   ├── generate_scenario_noisy.py     # Generate noisy scenario
│   ├── generate_scenario_ref.py       # Generate reference scenario
│   ├── compute_scenario_metrics.py    # Scenario metrics only
│   ├── visualize_scenario_optimization.py  # Scenario visualization
│   └── scenario/                  # Real scenario data directory
│       ├── vertices_stan.txt      # Camera poses
│       ├── id_kf_2_idx_img.txt    # Keyframe to image mapping
│       ├── loop_all.txt           # Loop closure pairs
│       └── imgs/                  # Image sequence
└── scripts_mapanything/           # MapAnything integration
    ├── install_mapanything.sh     # Installation script for MapAnything and dependencies
    ├── activate_mapanything.sh    # Quick activation script for virtualenv
    ├── run_loop_estimation.sh     # Bash wrapper for loop pose estimation (with filtering)
    └── scripts/
        ├── estimate_loop_poses.py # Main loop pose estimation script
        ├── filter_validated_loops.py  # Filter and organize validated loops
        ├── demo_*.py              # Various demo scripts
        ├── generate_depth_video*.py  # Depth video generation
        └── gradio_app.py          # Interactive web interface
```
