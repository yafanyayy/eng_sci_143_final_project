"""
Triangulation functions for Structure from Motion

Functions:
    - initialize_corr: Initialize the correspondence structure
    - find_existing_point: Check if a point already exists in the scene
    - update_existing_point: Update an existing 3D point with a new observation
    - add_new_point: Add a new 3D point to the scene
    - triangulate_pair: Triangulate 3D scene points from one matched image pair
"""

import cv2
import numpy as np


def initialize_corr(num_images):
    """
    Initialize the correspondence structure to track 3D points and their 2D observations.
    
    Args
    ----
      num_images : int
          Number of images in the sequence
    
    Returns
    -------
      corr : dict with keys
        - "scene_points_3d": list of 3D points (N x 3)
        - "matches_2d_location": list of 2D coordinates
        - "scene_point_features": list of descriptors
        - "num_images": number of images
    """
    corr = {
        "scene_points_3d": [],        # list of 3D points (N x 3)
        "matches_2d_location": [],    # list of 2D coordinates
        "scene_point_features": [],   # list of descriptors
        "num_images": num_images
    }
    return corr


def find_existing_point(corr, cam_idx, kp_idx):
    """
    Check if a point already exists in the scene based on camera and keypoint index.
    
    Args
    ----
      corr : dict
          Correspondence structure
      cam_idx : int
          Camera index
      kp_idx : int
          Keypoint index in that camera
    
    Returns
    -------
      point_idx : int or None
          Index of existing point in corr, or None if not found
    """
    # Check if a point already exists in the scene based on camera and keypoint index.
    for point_idx, scene_point_feature in enumerate(corr["scene_point_features"]):
        if scene_point_feature is None:
            continue
        if cam_idx < len(scene_point_feature) and scene_point_feature[cam_idx] == kp_idx:
            return point_idx
    return None


def update_existing_point(corr, point_idx, cam_idx, kp_idx, x, y):
    """
    Update an existing 3D point with a new observation from a new camera.
    
    Args
    ----
      corr : dict
          Correspondence structure
      point_idx : int
          Index of the existing point
      cam_idx : int
          Camera index of the new observation
      kp_idx : int
          Keypoint index in that camera
      x, y : float
          2D coordinates of the observation
    """
    # first, make sure that lists are long enough for new camera
    while len(corr['matches_2d_location'][point_idx]) < cam_idx + 1:
        corr['matches_2d_location'][point_idx].append(None)
    while len(corr['scene_point_features'][point_idx]) < cam_idx + 1:
        corr['scene_point_features'][point_idx].append(None)

    # Update the observation
    corr['matches_2d_location'][point_idx][cam_idx] = (float(x), float(y))
    corr['scene_point_features'][point_idx][cam_idx] = int(kp_idx)


def add_new_point(corr, point_3d, observations):
    """
    Add a new 3D point to the scene.
    
    Args
    ----
      corr : dict
          Correspondence structure
      point_3d : (3,) numpy array
          3D point coordinates
      observations : list of tuples
          List of (cam_idx, kp_idx, x, y) representing which cameras see this point
    
    Returns
    -------
      point_idx : int
          Index of the newly added point
    """
    # Add a new 3D point to the scene
    num_images = corr['num_images']
    point_idx = len(corr['scene_points_3d'])

    corr['scene_points_3d'].append(point_3d.copy())

    # initialize observation lists
    matches_2d = [None] * num_images
    scene_features = [None] * num_images

    # add in observations
    for cam_idx, kp_idx, x, y in observations:
        matches_2d[cam_idx] = (float(x), float(y))
        scene_features[cam_idx] = int(kp_idx)

    corr['matches_2d_location'].append(matches_2d)
    corr['scene_point_features'].append(scene_features)

    return point_idx


def triangulate_pair(pair_key, pair_matches, camera_matrices, corr):
    """
    Triangulate 3D scene points from one matched image pair.

    Args
    ----
      pair_key : tuple (i, j)
          Example: (0, 1)

      pair_matches : dict
          pair_matches[(i, j)] must contain:
              {
                "matches": list[cv2.DMatch],
                "points1": (N x 2) array of 2D points in image i,
                "points2": (N x 2) array of 2D points in image j
              }

      camera_matrices : list of 3x4 matrices
          camera_matrices[k] = P_k for the k-th image

      corr : dict
          Correspondence structure (will be updated)

    Returns
    -------
      num_updated : int
          Number of existing points that were updated
      num_new : int
          Number of new points that were added
    """
    i, j = pair_key
    entry = pair_matches[pair_key]

    matches = entry["matches"]
    pts1 = entry["points1"]  # Nx2
    pts2 = entry["points2"]  # Nx2

    if pts1.shape[0] == 0:
        return 0, 0

    P_i = camera_matrices[i]
    P_j = camera_matrices[j]

    num_updated = 0
    num_new = 0

    for k, m in enumerate(matches):
        # Fill in image i
        kp_idx_i = m.queryIdx
        x1, y1 = pts1[k]

        # Fill in image j
        kp_idx_j = m.trainIdx
        x2, y2 = pts2[k]

        # check if points already prev exist
        existing_i = find_existing_point(corr, i, kp_idx_i)
        existing_j = find_existing_point(corr, j, kp_idx_j)

        if existing_i is not None and existing_j is not None:
            # if they exist (in both cameras), skip
            continue
        elif existing_i is not None:
            # if only one exists (in camera i), update with observation from camera j
            update_existing_point(corr, existing_i, j, kp_idx_j, x2, y2)
            num_updated += 1
        else:
            # new point, triangulate
            pts1_col = np.array([[x1], [y1]], dtype=np.float32)
            pts2_col = np.array([[x2], [y2]], dtype=np.float32)

            point_4d = cv2.triangulatePoints(P_i, P_j, pts1_col, pts2_col)
            point_3d = (point_4d[:3] / point_4d[3]).flatten()

            # add them to the corr structure
            observations = [
                (i, kp_idx_i, x1, y1),
                (j, kp_idx_j, x2, y2)]
            add_new_point(corr, point_3d, observations)
            num_new += 1

    return num_updated, num_new

