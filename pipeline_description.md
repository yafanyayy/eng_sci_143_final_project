# Structure from Motion Pipeline - Function Flow

## Overview

Our Structure from Motion (SfM) pipeline reconstructs a 3D scene from multiple 2D images by estimating camera poses and triangulating 3D points. The implementation follows a sequential approach where we first establish a baseline with two cameras, then incrementally add new cameras using Perspective-n-Point (PnP) solving.

## Pipeline Flow with Pseudocode

The pipeline operates in two main stages:

```
STEP 1: Initialize with cameras 0 & 1
  └─> detect_and_match_keypoints(img_grayscales)
      └─> Uses SIFT to detect keypoints and descriptors
      └─> find_matches() filters top 10% of matches by distance
      └─> Returns: keypoints_list, descriptors_list, pair_matches
  
  └─> estimate_essential_and_pose_first_pair(pair_matches, K, (0,1))
      └─> Computes essential matrix E using RANSAC
      └─> Recovers relative pose (R_01, t_01) via cv2.recoverPose()
      └─> Returns: E, R, t, inlier_mask
  
  └─> initialize_corr(num_images)
      └─> Creates correspondence structure to track 3D points
  
  └─> triangulate_pair((0,1), pair_matches, camera_matrices, corr)
      └─> For each match:
          └─> Checks if point exists using find_existing_point()
          └─> If new: triangulates 3D point via cv2.triangulatePoints()
          └─> Adds point using add_new_point() or updates with update_existing_point()
      └─> Returns: num_updated, num_new

STEP 2: Loop through cameras 2, 3, 4, ...
  For each camera i:
    └─> estimate_new_camera_pose(img_gray[i], K, scene_points_3d, ...)
        └─> Detects SIFT keypoints in new image
        └─> Matches 3D point descriptors to new image descriptors
        └─> Uses Lowe's ratio test (good_match_ratio=0.75)
        └─> Solves PnP via cv2.solvePnPRansac() to get (R_i, t_i)
        └─> Returns: R, t, inlier_mask, keypoints, descriptors, correspondences
    
    └─> add_camera_pose(R_i, t_i)
        └─> Adds new camera to camera_poses list
    
    └─> triangulate_pair((i-1, i), pair_matches, camera_matrices, corr)
        └─> Triangulates new points or updates existing ones
        └─> Maintains correspondence structure across all cameras
```

## Key Functions

1. **`detect_and_match_keypoints()`**: Detects SIFT keypoints across all images and matches consecutive pairs, keeping only the top 10% of matches for quality.

2. **`estimate_essential_and_pose_first_pair()`**: Estimates the essential matrix between the first two cameras using RANSAC and recovers their relative pose.

3. **`triangulate_pair()`**: Triangulates 3D points from matched 2D correspondences between two cameras, maintaining a correspondence structure that tracks which cameras observe each 3D point.

4. **`estimate_new_camera_pose()`**: Uses PnP (Perspective-n-Point) solving to estimate the pose of a new camera given existing 3D points and their 2D projections in the new image.

5. **`initialize_corr()`**, **`add_new_point()`**, **`update_existing_point()`**, **`find_existing_point()`**: Manage the correspondence structure that tracks 3D points, their 2D observations across cameras, and keypoint indices.

6. **`visualize_scene_points()`** and **`visualize_scene_points_clean()`**: Create interactive 3D visualizations of the reconstructed scene points and camera poses using Plotly.

## Implementation Details

The pipeline maintains a global correspondence structure (`corr`) that stores:
- `scene_points_3d`: List of 3D point coordinates
- `matches_2d_location`: Per-point list of (x,y) observations in each camera
- `scene_point_features`: Per-point list of keypoint indices in each camera

This structure allows us to:
- Avoid duplicate triangulation of the same 3D point
- Update existing points when new cameras observe them
- Track the full observation history of each 3D point

The matching threshold is set to keep only the top 10% of matches (sorted by descriptor distance), ensuring high-quality correspondences for robust pose estimation and triangulation.



