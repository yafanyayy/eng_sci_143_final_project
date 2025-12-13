# Structure from Motion (SfM) Pipeline

A complete implementation of Structure from Motion that reconstructs 3D scenes from multiple 2D images by estimating camera poses and triangulating 3D points.

## Overview

This project implements a sequential Structure from Motion pipeline that:
1. Calibrates cameras using AprilTag detection
2. Detects and matches SIFT keypoints across image sequences
3. Estimates camera poses using essential matrix decomposition and PnP solving
4. Triangulates 3D scene points from matched correspondences
5. Reconstructs a dense 3D point cloud of the scene

The pipeline processes images incrementally, starting with a baseline pair of cameras and then adding subsequent cameras one by one using Perspective-n-Point (PnP) solving.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yafanyayy/eng_sci_143_final_project.git
cd eng_sci_143_final_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

- `opencv-python` - Computer vision operations (SIFT, essential matrix, PnP, triangulation)
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `plotly` - Interactive 3D visualizations
- `pupil-apriltags` - AprilTag detection for camera calibration
- `pillow-heif` - HEIC image format support
- `scipy` - Scientific computing utilities
- `torch` - PyTorch for bundle adjustment (optional)

## Project Structure

```
eng_sci_143_final_project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Master_Truly_Cleaned.ipynb         # Main Jupyter notebook with full pipeline
├── pipeline_description.md            # Detailed pipeline flow documentation
├── triangulation_description.md      # Triangulation function details
│
├── detect_and_match_keypoints.py      # SIFT keypoint detection and matching
├── estimate_essential_matrix.py       # Essential matrix estimation for first pair
├── estimate_new_camera_pose.py        # PnP-based pose estimation for new cameras
├── triangulation.py                   # 3D point triangulation and correspondence management
├── bundle_adjustment.py               # Bundle adjustment optimization (PyTorch)
├── visualize_scene_points.py          # 3D scene visualization utilities
│
├── camera_calib_raw2/                 # Camera calibration images (AprilTags)
├── scene_raw2/                        # Scene images for reconstruction
└── colmap_results/                    # COLMAP reconstruction results (for comparison)
```

## Usage

### Running the Full Pipeline

The main pipeline is implemented in `Master_Truly_Cleaned.ipynb`. Open the notebook and run all cells:

```bash
jupyter notebook Master_Truly_Cleaned.ipynb
```

The notebook includes:
- Camera calibration using AprilTags
- SIFT keypoint detection and matching
- Essential matrix estimation for the first camera pair
- Sequential camera pose estimation and triangulation
- 3D scene visualization

### Using Individual Modules

You can also import and use individual functions:

```python
from detect_and_match_keypoints import detect_and_match_keypoints
from estimate_essential_matrix import estimate_essential_and_pose_first_pair
from triangulation import triangulate_pair, initialize_corr
from estimate_new_camera_pose import estimate_new_camera_pose

# Detect and match keypoints
data = detect_and_match_keypoints(img_grayscales, good_match_percentage=0.10)

# Estimate essential matrix for first pair
E, R, t, mask = estimate_essential_and_pose_first_pair(
    data['pair_matches'], 
    camera_matrix, 
    pair_key=(0, 1)
)

# Initialize correspondence structure
corr = initialize_corr(num_images)

# Triangulate points for a camera pair
num_updated, num_new = triangulate_pair(
    (0, 1), 
    data['pair_matches'], 
    camera_matrices, 
    corr
)

# Estimate pose for a new camera
R_new, t_new, inlier_mask, kp_new, desc_new, correspondences = estimate_new_camera_pose(
    new_img_gray,
    K,
    scene_points_3d,
    scene_point_features,
    descriptors_list
)
```

## Key Components

### 1. Camera Calibration (`camera_calib.py`)
- Detects AprilTag markers in calibration images
- Computes camera intrinsic matrix (K) and distortion coefficients
- Uses `pupil-apriltags` for marker detection

### 2. Keypoint Detection and Matching (`detect_and_match_keypoints.py`)
- **`detect_and_match_keypoints()`**: Detects SIFT keypoints across all images and matches consecutive pairs
- **`find_matches()`**: Filters matches to keep only the top percentage (default 10%) by descriptor distance
- Returns structured dictionary with keypoints, descriptors, and pair matches

### 3. Essential Matrix Estimation (`estimate_essential_matrix.py`)
- **`estimate_essential_and_pose_first_pair()`**: Estimates essential matrix between first two cameras using RANSAC
- Recovers relative pose (rotation R and translation t) using `cv2.recoverPose()`
- Used only for the initial baseline pair (cameras 0 and 1)

### 4. Triangulation (`triangulation.py`)
- **`triangulate_pair()`**: Triangulates 3D points from matched 2D correspondences
- **`initialize_corr()`**: Initializes correspondence structure
- **`find_existing_point()`**: Checks if a point already exists in the scene
- **`update_existing_point()`**: Updates existing 3D point with new observation
- **`add_new_point()`**: Adds new 3D point to the scene
- Maintains correspondence structure to track 3D points across multiple cameras

### 5. New Camera Pose Estimation (`estimate_new_camera_pose.py`)
- **`estimate_new_camera_pose()`**: Uses PnP (Perspective-n-Point) solving to estimate pose of new camera
- Matches 3D point descriptors to new image descriptors
- Uses Lowe's ratio test for robust matching
- Solves PnP via `cv2.solvePnPRansac()` with RANSAC for outlier rejection

### 6. Bundle Adjustment (`bundle_adjustment.py`)
- Non-linear optimization to refine 3D points and camera poses
- Implemented using PyTorch for gradient-based optimization
- Optional refinement step (not included in main pipeline execution)

## Pipeline Flow

The pipeline operates in two main stages:

1. **Initialization (Cameras 0 & 1)**:
   - Detect and match SIFT keypoints
   - Estimate essential matrix and recover relative pose
   - Initialize correspondence structure
   - Triangulate initial 3D points

2. **Incremental Addition (Cameras 2, 3, 4, ...)**:
   - For each new camera:
     - Estimate pose using PnP with existing 3D points
     - Add camera to pose dictionary
     - Triangulate new points or update existing ones

For detailed pseudocode and function flow, see [`pipeline_description.md`](pipeline_description.md).

## Data Format

### Input Images
- Calibration images: HEIC format in `camera_calib_raw2/`
- Scene images: HEIC format in `scene_raw2/`
- Images are converted to grayscale for processing

### Output Structure
- **Correspondence structure (`corr`)**:
  - `scene_points_3d`: List of 3D point coordinates (N x 3)
  - `matches_2d_location`: Per-point list of (x,y) observations in each camera
  - `scene_point_features`: Per-point list of keypoint indices in each camera

- **Camera poses**: List of (R, t) tuples for each camera

## Visualization

The pipeline includes interactive 3D visualizations using Plotly:
- 3D scene points
- Camera positions and orientations
- Epipolar geometry visualization
- Match visualization between image pairs

See `visualize_scene_points.py` for visualization utilities.

## Parameters

Key parameters that can be adjusted:

- **`good_match_percentage`**: Fraction of best matches to keep (default: 0.10 = top 10%)
- **`good_match_ratio`**: Lowe's ratio test threshold for PnP matching (default: 0.75)
- **`min_inliers`**: Minimum number of inliers required for PnP (default: 20)
- **RANSAC parameters**: Threshold and confidence for essential matrix and PnP estimation

## Documentation

- [`pipeline_description.md`](pipeline_description.md): Detailed pipeline flow with pseudocode
- [`triangulation_description.md`](triangulation_description.md): Triangulation function details

## References

- OpenCV Documentation: https://docs.opencv.org/
- Structure from Motion: Multiple View Geometry in Computer Vision (Hartley & Zisserman)
- SIFT: Distinctive Image Features from Scale-Invariant Keypoints (Lowe, 2004)
- AprilTag: A robust and flexible visual fiducial system (Olson, 2011)

## License

This project is part of ES 143 (Engineering Sciences 143) coursework at Harvard University.

## GitHub Repository

[View on GitHub](https://github.com/yafanyayy/eng_sci_143_final_project)

