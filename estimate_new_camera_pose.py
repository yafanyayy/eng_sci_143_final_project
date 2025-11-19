"""
Estimate new camera pose using PnP solver
"""

import cv2
import numpy as np


def estimate_new_camera_pose(matched_keypoints, scene_points):
    """
    Add a new camera and estimate the camera pose using PnP solver.
    
    Purpose: Estimate the pose (rotation and translation) of a new camera
    given matched keypoints between the previous camera image and the new
    camera image, along with the 3D coordinates of those matched points.
    
    Input:
        matched_keypoints: Dictionary or tuple containing:
            - previous_camera_points: 2D image points from previous camera (Nx2 array)
            - new_camera_points: 2D image points from new camera (Nx2 array)
        scene_points: 3D coordinates of matched points (Nx3 array)
            At least one derived 3D coordinate of matched points
    
    Output:
        R: Rotation matrix (3x3) of the new camera
        t: Translation vector (3x1) of the new camera
    
    Unit test(s) and/or other plans to verify and demonstrate correctness:
        - Plot epipolar points and verify that corresponding points are on the epipolar lines
        - Verify that epipolar points and lines are consistent with camera geometry
    """
    # TODO: Implement PnP solver to estimate camera pose
    # This will use cv2.solvePnP or cv2.solvePnPRansac
    
    R = None
    t = None
    
    return R, t

