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

---

## 3D Pose Graph SLAM

The repository includes a complete 3D implementation of the pose graph SLAM system using SE(3) transformations with quaternions.

### Running 3D SLAM

```bash
# Basic 3D optimization
python main_3d.py

# With custom parameters
python main_3d.py --trajectory-type helix --num-poses 30 --loop-sampling 2
```

### 3D Modules

**trajectory_3d.py**: 3D trajectory generation
- `genTraj3D()`: Creates a U-turn trajectory with altitude variations
- `genTrajSpiralHelix3D()`: Creates a helical spiral trajectory
- `addNoise3D()`: Adds Gaussian noise to SE(3) transformations

**g2o_io_3d.py**: G2O 3D file format handling
- Uses `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT` format
- Handles quaternion-based rotations (qx, qy, qz, qw)
- 6×6 information matrices for SE(3) edges

**visualization_3d.py**: 3D matplotlib visualization
- `draw3D()`: Single 3D trajectory with orientation arrows
- `drawTwo3D()`: Ground truth vs optimized comparison
- `drawThree3D()`: Three trajectories comparison
- `plotErrors3D()`: Error plots for position and orientation

**metrics_3d.py**: 3D error metrics
- Position errors: 3D Euclidean distance
- Orientation errors: Geodesic distance on SO(3) using quaternions
- `compute_pose_errors_3d()`: Computes RMSE, mean, max for both position and orientation

**pose_graph_3d.py**: 3D optimization wrapper
- Executes g2o with 3D pose graph files

---

## Real Scenario Data Processing

The repository includes tools for processing and optimizing real trajectory data from a SLAM system.

### Scenario File Format

Input files in `scenario/` folder:
- `vertices_stan.txt`: Camera poses in camera-to-world format (tx, ty, tz, qx, qy, qz, qw)
- `id_kf_2_idx_img.txt`: Mapping of keyframe IDs to image indices
- `loop_all.txt`: Loop closure pairs

### Preparing Scenario Data

#### 1. Generate Reference and Noisy Scenarios

```bash
# Step 1: Add loop closures between trajectory segments [0-88] and [304-392]
python add_loops_segments.py

# Step 2: Generate noisy scenario (first 400 poses with loops)
python generate_scenario_noisy.py

# Step 3: Generate reference scenario (poses [304-392] = copies of [0-88])
python generate_scenario_ref.py
```

This creates:
- `scenario/scenario_noisy.txt`: Noisy trajectory with loop closure annotations
- `scenario/scenario_ref.txt`: Ground truth where segment 2 equals segment 1 (perfect alignment)

#### 2. Visualize Scenarios

```bash
# Visualize all poses
python visualize_all_poses.py --num-poses 400

# Visualize with loop closures between segments
python visualize_loops_segments.py --num-poses 400

# Compare noisy vs reference scenarios
python visualize_scenarios.py
```

### Optimizing Real Scenarios

#### Complete Pipeline (Recommended)

```bash
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
```

**The complete pipeline includes:**

1. **Initial Visualization**: Scenario noisy with loop closures (cyan lines)
2. **G2O File Generation**:
   - `scenario_noisy_input.g2o`: Input with noisy odometry + loop constraints
   - `scenario_ref_input.g2o`: Reference for verification
3. **G2O Optimization**: Runs g2o on noisy scenario
4. **Results Visualization**:
   - Reference vs Optimized (side-by-side)
   - Three trajectories (separate views)
   - Three trajectories (overlayed): Green=Reference, Red=Noisy, Blue=Optimized
5. **Error Metrics**:
   - Position errors (RMSE, mean, max) in meters
   - Orientation errors (RMSE, mean, max) in degrees
   - Improvement percentages
   - Per-pose error plots

#### Individual Steps

```bash
# Just compute and visualize metrics (after optimization)
python compute_scenario_metrics.py

# Just visualize optimization results
python visualize_scenario_optimization.py
```

### Scenario Data Architecture

**Key Modules:**

**g2o_io_scenario.py**: Scenario to g2o conversion
- `load_scenario()`: Loads scenario txt files
- `write_g2o_from_scenario()`: Converts to VERTEX_SE3:QUAT and EDGE_SE3:QUAT format
- `compute_relative_transform()`: Computes SE(3) transformations between poses
- `read_g2o_optimized()`: Reads optimized results
- Supports loop closure sampling with `--loop-sampling` parameter
- **Critical feature**: Loop edges calculated from reference poses (perfect constraints)

**compute_scenario_metrics.py**: Error evaluation
- `compute_pose_errors_3d()`: Compares trajectories against reference
- Position: 3D Euclidean distance
- Orientation: Geodesic angle on SO(3)
- Generates detailed statistics and plots

**visualize_scenario_optimization.py**: Result visualization
- Four-panel visualization showing noisy, optimized, and reference trajectories
- Side-by-side comparisons
- Overlayed view for direct comparison

### Important Technical Details

**Coordinate Frame Conversion:**
- Input data is in camera-to-world (c2w) format
- Converted to world-to-camera (w2c) for visualization:
  ```python
  R_w2c = R_c2w.T
  t_w2c = -R_w2c @ t_c2w
  ```

**Loop Closure Strategy:**
- Segment 1: Poses [0-88]
- Segment 2: Poses [304-392]
- 1-to-1 mapping: pose i ↔ pose (304+i)
- In `scenario_ref.txt`: Both segments have identical poses (ground truth)
- In `scenario_noisy.txt`: Different poses simulate odometry drift
- **Critical**: Loop edges are computed from reference poses, ensuring perfect constraints (≈identity transformation)

**Loop Closure Edge Calculation:**
- Odometry edges: Computed from noisy poses (sequential)
- Loop closure edges: Computed from reference poses (perfect alignment)
- This simulates perfect loop detection despite noisy odometry

**Information Matrices:**
- Default odometry info: 500.0 (moderate confidence in odometry)
- Default loop info: 700.0 (higher confidence in loops)
- Higher loop info helps pull noisy trajectory toward reference

### Generated Files

After running `optimize_scenario_complete.py`:
- `scenario_noisy_input.g2o`: Input for g2o (noisy vertices + perfect loop edges)
- `scenario_ref_input.g2o`: Reference (for verification)
- `scenario_optimized.g2o`: Output from g2o
- `scenario_optimized.txt`: Optimized poses in readable format

### Typical Results

With 89 loop closures between segments [0-88] and [304-392]:
- Position RMSE reduction: 50-80% improvement
- Segment alignment: Optimized trajectory aligns segment 2 with segment 1
- Visual inspection: Overlayed plot shows blue trajectory approaching green reference

### Loop Sampling Impact

Test the effect of loop closure density:

```bash
# All loops (89 loops for 400 poses)
python optimize_scenario_complete.py --loop-sampling 1

# Sparse loops (30 loops)
python optimize_scenario_complete.py --loop-sampling 3

# Very sparse (18 loops)
python optimize_scenario_complete.py --loop-sampling 5

# No loops (pure odometry)
python optimize_scenario_complete.py --loop-sampling 0
```

Compare metrics to see minimum loop density needed for good optimization.

---

## Bag of Words (BoW) Loop Closure Detection

The repository includes a complete Bag of Words implementation for vision-based loop closure detection using ORB features.

### Overview

The BoW approach works in two stages:
1. **Vocabulary Construction**: Extract ORB features from images and cluster them into visual words using K-Means
2. **Loop Detection**: Represent each image as a histogram of visual words and find similar images using cosine similarity

### Image Preparation

Images should be placed in `scenario/imgs/` with sequential naming (e.g., `img00000.jpg` to `img01197.jpg`).

**Helper scripts:**
- `copy_images.py`: Copy images from source directory based on mapping file
- `rename_images.py`: Rename images to sequential format with zero-padding

### Building the BoW Vocabulary

```bash
# Build vocabulary with default settings (k=1000 visual words)
python build_bow_vocabulary.py

# Custom vocabulary size (recommended: k=500 for ~1000-2000 images)
python build_bow_vocabulary.py --k 500 --features-per-image 300

# Fast test with fewer images
python build_bow_vocabulary.py --k 300 --max-images 300 --features-per-image 200

# High precision vocabulary
python build_bow_vocabulary.py --k 1000 --features-per-image 500
```

**Key Parameters:**
- `--k`: Number of visual words (clusters)
  - `100-300`: Fast, less discriminant
  - `500-1000`: **Recommended** - good balance
  - `1000-5000`: High precision, slower
- `--features-per-image`: ORB features extracted per image
  - `200-300`: Fast, sufficient for simple scenes
  - `300-500`: **Recommended** - balanced
  - `500-1000`: More robust, slower
- `--max-images`: Limit number of images for vocabulary (useful for testing)
- `--output`: Output vocabulary file (default: `bow_vocabulary.pkl`)

**Output:**
- `bow_vocabulary.pkl`: Trained K-Means model for visual word assignment

### Detecting Loop Closures

```bash
# Basic detection
python detect_loops_bow.py --similarity-threshold 0.7 --min-temporal-distance 30

# With interactive visualization (shows feature matches)
python detect_loops_bow.py --similarity-threshold 0.75 --min-temporal-distance 50 --show-pairs

# Adjust detection parameters
python detect_loops_bow.py --similarity-threshold 0.80 --min-temporal-distance 100 --show-pairs --max-pairs 10

# With similarity matrix visualization
python detect_loops_bow.py --similarity-threshold 0.75 --visualize --show-pairs
```

**Key Parameters:**
- `--similarity-threshold`: Cosine similarity threshold (0-1)
  - `0.65-0.70`: More detections, more false positives
  - `0.75-0.80`: **Recommended** - good balance
  - `0.85+`: Very strict, fewer detections
- `--min-temporal-distance`: Minimum frame separation for loop closure
  - `30-50`: Standard for short sequences
  - `50-100`: **Recommended** for longer sequences
  - `100+`: Very conservative
- `--show-pairs`: Display detected pairs interactively with feature matches
- `--max-pairs`: Number of pairs to visualize (sorted by similarity)
- `--visualize`: Generate similarity matrix plot (useful for debugging)
- `--features-per-image`: Must match vocabulary building setting

**Output:**
- `loop_closures_bow.txt`: Detected loop closure pairs (format: `image_i image_j similarity_score`)
- Interactive display shows:
  - Images side-by-side
  - ORB feature matches (green lines)
  - Keypoints (red dots)
  - Similarity score and match count

### BoW Pipeline Workflow

```bash
# 1. Build vocabulary (one-time operation)
python build_bow_vocabulary.py --k 500 --features-per-image 300

# 2. Detect loops and visualize top 20 pairs
python detect_loops_bow.py --similarity-threshold 0.75 --min-temporal-distance 50 --show-pairs --max-pairs 20

# 3. Integrate with pose graph optimization (manual step)
# Use detected loop closures from loop_closures_bow.txt
# Add corresponding edges to g2o file based on image indices
```

### BoW Implementation Details

**Feature Extraction:**
- Uses ORB (Oriented FAST and Rotated BRIEF) features
- Binary descriptors (256 bits / 32 bytes each)
- Rotation and scale invariant
- Fast extraction and matching

**Visual Vocabulary:**
- K-Means clustering on descriptor space
- Each cluster center = one "visual word"
- MiniBatchKMeans used for efficiency with large datasets

**Image Representation:**
- Each image → histogram of visual word frequencies
- L2-normalized histograms for robust comparison
- Similarity measured using cosine similarity

**Loop Closure Detection:**
- Compare all image pairs with sufficient temporal separation
- Threshold on cosine similarity
- Feature matching verification available for visualization

### Typical Results

For a dataset of ~1200 images:
- Vocabulary building: 5-10 minutes (k=500)
- Loop detection: 2-5 minutes
- Expected loops: 50-500 depending on trajectory overlap
- Top matches typically have similarity > 0.80

**Troubleshooting:**

If too many loop closures detected (>1000):
- Increase `--similarity-threshold` (try 0.80-0.85)
- Increase `--min-temporal-distance` (try 100+)
- Increase vocabulary size `--k` for better discrimination

If too few loop closures detected (<10):
- Decrease `--similarity-threshold` (try 0.65-0.70)
- Decrease `--min-temporal-distance` (try 30)
- Check that images contain distinctive features

### Integration with SLAM Pipeline

The detected loop closures can be integrated with the pose graph optimization:

1. Map image indices to pose indices using `id_kf_2_idx_img.txt`
2. For each detected loop closure (i, j):
   - Compute relative pose constraint (e.g., using visual odometry or ICP)
   - Add EDGE_SE3:QUAT to g2o file with appropriate information matrix
3. Run g2o optimization with combined odometry + BoW loop edges

### Dependencies

Additional dependencies for BoW:
- **opencv-python** (`cv2`): Feature extraction and matching
- **scikit-learn**: K-Means clustering
- **pickle**: Model serialization

---

## DINOv2 Loop Closure Detection

The repository includes a DINOv2-based loop closure detection system using deep learning vision features for superior place recognition.

### Overview

DINOv2 is a self-supervised Vision Transformer that produces high-quality global image features without task-specific training. It offers several advantages over traditional BoW approaches:

**Advantages of DINOv2:**
- Global semantic features (vs. local ORB features)
- No vocabulary construction required
- Robust to illumination and viewpoint changes
- Higher similarity thresholds (0.85-0.90 typical)
- Better generalization to unseen environments
- State-of-the-art performance on place recognition

**When to use DINOv2 vs BoW:**
- **DINOv2**: Better accuracy, requires GPU for reasonable speed, no training phase
- **BoW**: Faster inference on CPU, requires vocabulary construction, good for resource-constrained systems

### Building DINOv2 Features

Extract DINOv2 features from all images in the dataset:

```bash
# Basic extraction (small model, fastest)
python build_dinov2_features.py

# Recommended: base model (balanced speed/quality)
python build_dinov2_features.py --model-size base

# High quality: large model (best accuracy)
python build_dinov2_features.py --model-size large

# Test on subset of images
python build_dinov2_features.py --max-images 200 --model-size base

# Custom output location
python build_dinov2_features.py --output my_features.pkl --model-size base
```

**Model Sizes:**
- `small`: 384-dim features, fastest, ~1GB VRAM
- `base`: 768-dim features, **recommended**, ~3GB VRAM
- `large`: 1024-dim features, high quality, ~6GB VRAM
- `giant`: 1536-dim features, best quality, requires ~24GB VRAM

**Parameters:**
- `--image-dir`: Directory containing images (default: `scenario/imgs`)
- `--output`: Output pickle file (default: `dinov2_features.pkl`)
- `--model-size`: Model size (default: `small`)
- `--max-images`: Limit number of images for testing

**Output:**
- `dinov2_features.pkl`: Dictionary containing:
  - `features`: Image name → normalized feature vector mapping
  - `model_size`: Model size used
  - `feature_dim`: Feature dimension
  - `num_images`: Number of images processed

**Performance:**
- Feature extraction: ~5-15 images/second (GPU), ~0.5-2 images/second (CPU)
- For ~1200 images: 2-10 minutes on GPU, 10-40 minutes on CPU

### Detecting Loop Closures with DINOv2

Compute pairwise image similarities and detect loop closures:

```bash
# Basic detection (recommended settings)
python detect_loops_dinov2.py --similarity-threshold 0.85 --min-temporal-distance 50

# With all visualizations
python detect_loops_dinov2.py \
  --similarity-threshold 0.85 \
  --min-temporal-distance 50 \
  --visualize \
  --plot-distribution \
  --show-pairs \
  --max-pairs 20

# More conservative (fewer false positives)
python detect_loops_dinov2.py --similarity-threshold 0.90 --min-temporal-distance 100

# More permissive (more detections)
python detect_loops_dinov2.py --similarity-threshold 0.80 --min-temporal-distance 30

# Custom input/output files
python detect_loops_dinov2.py \
  --features my_features.pkl \
  --output my_loops.txt \
  --similarity-threshold 0.85
```

**Key Parameters:**
- `--features`: Input pickle file with DINOv2 features (default: `dinov2_features.pkl`)
- `--similarity-threshold`: Minimum cosine similarity (0-1)
  - `0.80-0.85`: More detections, some false positives
  - `0.85-0.90`: **Recommended** - good precision/recall balance
  - `0.90-0.95`: Very strict, high precision
- `--min-temporal-distance`: Minimum frame separation (default: 30)
  - `30-50`: Standard for typical sequences
  - `50-100`: **Recommended** for longer sequences
  - `100+`: Conservative for very long trajectories
- `--output`: Output file for loop closures (default: `loop_closures_dinov2.txt`)
- `--visualize`: Generate similarity matrix heatmap
- `--plot-distribution`: Plot histogram of similarity scores
- `--show-pairs`: Display detected pairs interactively
- `--max-pairs`: Number of pairs to visualize (default: 10)

**Output Files:**
- `loop_closures_dinov2.txt`: Detected loop closures
  - Format: `idx_i idx_j similarity image_name_i image_name_j`
  - Sorted by similarity (descending)
- `dinov2_similarity_matrix.png`: Heatmap visualizations (if `--visualize`)
- `dinov2_similarity_distribution.png`: Similarity histogram (if `--plot-distribution`)

**Performance:**
- Similarity computation: Fast (~0.1-1 second for 1000×1000 matrix)
- Interactive visualization: Shows image pairs with similarity scores

### Visualizing DINOv2 Loop Closures

Display detected loop closures overlayed on the robot trajectory:

```bash
# 2D visualization (top view)
python visualize_loops_dinov2.py

# 3D visualization
python visualize_loops_dinov2.py --3d

# With loop closure statistics
python visualize_loops_dinov2.py --statistics

# Limit number of displayed loops (for clarity)
python visualize_loops_dinov2.py --max-loops 100

# Without keyframe mapping (direct image idx = pose idx)
python visualize_loops_dinov2.py --no-mapping

# Custom files
python visualize_loops_dinov2.py \
  --loop-file my_loops.txt \
  --scenario-file scenario/scenario_noisy.txt \
  --mapping-file scenario/id_kf_2_idx_img.txt
```

**Parameters:**
- `--loop-file`: Loop closures file (default: `loop_closures_dinov2.txt`)
- `--scenario-file`: Scenario poses file (default: `scenario/scenario_noisy.txt`)
- `--mapping-file`: Keyframe to image mapping (default: `scenario/id_kf_2_idx_img.txt`)
- `--max-loops`: Maximum loops to display (default: all)
- `--no-mapping`: Assume direct mapping (img_idx == pose_idx)
- `--3d`: Show 3D trajectory view
- `--statistics`: Plot loop closure statistics

**Output Files:**
- `dinov2_loops_2d.png`: 2D trajectory with loop closures
- `dinov2_loops_3d.png`: 3D trajectory with loop closures (if `--3d`)
- `dinov2_loop_statistics.png`: Statistics plots (if `--statistics`)

**Visualization Features:**
- Blue line: Robot trajectory
- Green circle: Start position
- Red circle: End position
- Cyan dashed lines: Loop closure constraints
- Statistics panel: Similarity distribution, temporal gaps, correlations

### Complete DINOv2 Workflow

```bash
# Step 1: Extract features (one-time operation)
python build_dinov2_features.py --model-size base
# Output: dinov2_features.pkl (~50-200MB depending on dataset size)

# Step 2: Detect loop closures with visualizations
python detect_loops_dinov2.py \
  --similarity-threshold 0.85 \
  --min-temporal-distance 50 \
  --visualize \
  --show-pairs \
  --max-pairs 20
# Output: loop_closures_dinov2.txt, similarity matrix, distribution plot

# Step 3: Visualize on trajectory
python visualize_loops_dinov2.py --statistics
# Output: Trajectory plot with loop closures, statistics

# Step 4: (Optional) Integrate with pose graph optimization
# Use loop_closures_dinov2.txt to add loop constraints to g2o file
```

### DINOv2 Implementation Details

**Architecture:**
- Vision Transformer (ViT) backbone with 14×14 patch size
- Self-supervised training on large-scale image datasets
- Global [CLS] token used as image representation
- L2-normalized features for cosine similarity

**Feature Extraction:**
- Input: RGB images resized to 224×224
- Preprocessing: Standard ImageNet normalization
- Output: L2-normalized feature vectors (384/768/1024/1536-dim)
- No gradient computation (inference only)

**Similarity Computation:**
- Cosine similarity between normalized features
- Equivalent to dot product for unit vectors
- Range: [-1, 1], typically [0.3, 0.95] for real images
- Higher values = more similar images

**Loop Closure Detection:**
- Compute full N×N similarity matrix
- Apply temporal constraint (min frame separation)
- Threshold on similarity score
- Sort by similarity for prioritization

### Typical Results

For a dataset of ~1200 images:
- Feature extraction: 2-10 minutes (GPU), 10-40 minutes (CPU)
- Loop detection: <1 minute
- Expected loops: 50-500 depending on trajectory overlap
- Typical similarity range: [0.40, 0.95]
- Top matches typically have similarity > 0.90

**Comparison with BoW:**
| Metric | BoW | DINOv2 |
|--------|-----|--------|
| Feature extraction | 2-5 min | 2-10 min (GPU) |
| Vocabulary building | 5-10 min | Not required |
| Loop detection | 2-5 min | <1 min |
| Typical threshold | 0.70-0.80 | 0.85-0.90 |
| False positive rate | Moderate | Low |
| Robustness | Good | Excellent |

### Troubleshooting

**Too many loop closures detected (>1000):**
- Increase `--similarity-threshold` (try 0.90-0.95)
- Increase `--min-temporal-distance` (try 100+)
- Use larger model size for better discrimination

**Too few loop closures detected (<10):**
- Decrease `--similarity-threshold` (try 0.75-0.80)
- Decrease `--min-temporal-distance` (try 30)
- Check that trajectory has actual revisits
- Verify images are loaded correctly

**Out of memory (GPU):**
- Use smaller model: `--model-size small`
- Reduce batch size (currently 1, minimal impact)
- Process images in chunks (modify script if needed)

**Slow performance:**
- Use GPU if available (CUDA)
- Use `small` model instead of `base`/`large`
- Reduce `--max-images` for testing

### Integration with SLAM Pipeline

Integrate DINOv2 loop closures with pose graph optimization:

1. **Map image indices to pose indices:**
   ```python
   # Use scenario/id_kf_2_idx_img.txt mapping
   # Maps keyframe ID (pose index) to image index
   ```

2. **For each detected loop closure (img_i, img_j, similarity):**
   - Map img_i, img_j → pose_i, pose_j using keyframe mapping
   - Compute relative pose constraint (e.g., visual odometry, ICP)
   - Add EDGE_SE3:QUAT to g2o file
   - Use information matrix based on similarity score:
     ```python
     info_weight = 700.0 * (similarity / 0.85) ** 2
     ```

3. **Run optimization:**
   ```bash
   python optimize_scenario_complete.py
   ```

**Information Matrix Weighting:**
- High similarity (>0.90): Higher information weight (~800-1000)
- Medium similarity (0.85-0.90): Standard weight (~700)
- Low similarity (0.80-0.85): Lower weight (~500-600)

### Dependencies

Additional dependencies for DINOv2:
- **torch**: PyTorch framework for neural networks
- **torchvision**: Image preprocessing and transforms
- **opencv-python** (`cv2`): Image loading and visualization
- **PIL** (Pillow): Image handling
- **tqdm**: Progress bars
- **matplotlib**: Visualization
- **numpy**: Numerical operations

Install dependencies:
```bash
pip install torch torchvision opencv-python pillow tqdm matplotlib numpy
```

**Hardware Requirements:**
- **GPU (recommended)**: NVIDIA GPU with CUDA support, 2-8GB VRAM depending on model size
- **CPU fallback**: Works but 5-10× slower
- **RAM**: 4-8GB minimum, 16GB recommended for large datasets
- **Disk**: ~100-500MB for cached models (downloaded automatically)
