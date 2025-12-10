"""
Estimate new camera pose using PnP (Perspective-n-Point) solver

The PnP problem: Given n 3D points and their 2D projections in an image,
find the camera pose (rotation R and translation t) that projects the 3D
points to the 2D image points.
"""

import cv2
import numpy as np


def estimate_new_camera_pose(matched_keypoints, scene_points, K, distCoeffs=None):
    """
    Add a new camera and estimate the camera pose using PnP solver.
    
    Purpose: Estimate the pose (rotation and translation) of a new camera
    given matched keypoints between the previous camera image and the new
    camera image, along with the 3D coordinates of those matched points.
    
    Input:
        matched_keypoints: List of 2D points from the new camera (Nx2 array)
            These should be extracted from keypoints_list using trainIdx before calling
            Example: [keypoints_list[i+1][match.trainIdx].pt for match in matched_keypoints[i]]
        scene_points: 3D coordinates of matched points (Nx3 array)
            At least one derived 3D coordinate of matched points
            Must correspond 1-to-1 with matched_keypoints (same length)
        K: Camera calibration matrix (3x3)
        distCoeffs: Distortion coefficients (optional, defaults to None/zeros)
    
    Output:
        R: Rotation matrix (3x3) of the new camera
        t: Translation vector (3x1) of the new camera
    
    Unit test(s) and/or other plans to verify and demonstrate correctness:
        - Plot epipolar points and verify that corresponding points are on the epipolar lines
        - Verify that epipolar points and lines are consistent with camera geometry
    """
    # Convert inputs to numpy arrays
    new_camera_points = np.array(matched_keypoints, dtype=np.float32)
    scene_points = np.array(scene_points, dtype=np.float32)
    
    # Ensure correct shape
    if new_camera_points.ndim == 1:
        new_camera_points = new_camera_points.reshape(-1, 2)
    if new_camera_points.shape[1] != 2:
        raise ValueError(f"matched_keypoints must be Nx2 array, got shape: {new_camera_points.shape}")
    
    if scene_points.ndim == 1:
        scene_points = scene_points.reshape(-1, 3)
    if scene_points.shape[1] != 3:
        raise ValueError(f"scene_points must be Nx3 array, got shape: {scene_points.shape}")
    
    # Check that we have matching number of points
    if len(new_camera_points) != len(scene_points):
        raise ValueError(f"Mismatch: {len(new_camera_points)} 2D points vs {len(scene_points)} 3D points")
    
    if len(new_camera_points) < 4:
        raise ValueError("Need at least 4 point correspondences for PnP solver")
    
    # Set up distortion coefficients
    if distCoeffs is None:
        distCoeffs = np.zeros((4, 1), dtype=np.float32)
    else:
        distCoeffs = np.array(distCoeffs, dtype=np.float32)
    
    # Use PnP solver to estimate camera pose
    # cv2.solvePnP solves: project 3D points to 2D using camera pose
    # We use RANSAC for robustness against outliers
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=scene_points,      # 3D points in world coordinates
        imagePoints=new_camera_points,   # 2D points in image coordinates
        cameraMatrix=K,                 # Camera calibration matrix
        distCoeffs=distCoeffs,         # Distortion coefficients
        flags=cv2.SOLVEPNP_ITERATIVE,  # Algorithm flag
        reprojectionError=8.0,          # Maximum reprojection error (in pixels)
        confidence=0.99,                # Confidence level
        iterationsCount=200             # Maximum iterations
    )
    
    if not success:
        raise RuntimeError("PnP solver failed to find a solution")
    
    # Convert rotation vector to rotation matrix
    # cv2.solvePnP returns rotation as a rotation vector (axis-angle representation)
    # We need to convert it to a 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Return rotation matrix and translation vector
    t = tvec.reshape(3, 1) if tvec.ndim == 1 else tvec
    
    return R, t
