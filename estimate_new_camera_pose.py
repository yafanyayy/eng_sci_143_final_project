"""
Estimate new camera pose using PnP (Perspective-n-Point) solver

The PnP problem: Given n 3D points and their 2D projections in an image,
find the camera pose (rotation R and translation t) that projects the 3D
points to the 2D image points.
"""

import cv2
import numpy as np


def estimate_new_camera_pose(
    new_img_gray,
    K,
    scene_points_3d,
    scene_point_features,
    descriptors_list,
    good_match_ratio=0.75,
    min_inliers=20
):
    """
    Estimate pose (R, t) of a new camera given an existing 3D map.
    
    Args
    ----
      new_img_gray : numpy array
          Grayscale image from the new camera
      K : (3 x 3) numpy array
          Camera intrinsic matrix
      scene_points_3d : (N x 3) numpy array
          Existing 3D scene points
      scene_point_features : list
          List of feature descriptors for each 3D point
          scene_point_features[p_idx] is a list of keypoint indices per camera
      descriptors_list : list
          List of descriptor arrays for each camera
          descriptors_list[cam_idx] is the descriptor array for that camera
      good_match_ratio : float
          Ratio threshold for good matches (default 0.75)
      min_inliers : int
          Minimum number of inliers required (default 20)
    
    Returns
    -------
      R_new : (3 x 3) numpy array
          Rotation matrix of the new camera
      t_new : (3 x 1) numpy array
          Translation vector of the new camera
      inlier_mask : (M x 1) numpy array
          Inlier mask from PnP RANSAC
      kp_new : list
          Keypoints detected in the new image
      desc_new : numpy array
          Descriptors for the new image
      correspondences : dict
          Dictionary containing:
            - "object_points_3d": 3D points used for PnP
            - "image_points_2d": 2D points in new image
            - "scene_indices": indices into scene_points_3d
            - "keypoint_indices": keypoint indices in new image
    """
    num_cams_so_far = len(descriptors_list)
    num_points = scene_points_3d.shape[0]

    # Detect SIFT in new image
    sift = cv2.SIFT_create()
    kp_new, desc_new = sift.detectAndCompute(new_img_gray, None)

    if desc_new is None or len(kp_new) < 4:
        raise RuntimeError("Not enough keypoints/descriptors in new image for PnP.")

    # Build descriptor list for 3D points: use last valid observation
    point_descriptors = []
    point_indices = []  # index into scene_points_3d

    for p_idx in range(num_points):
        feats = scene_point_features[p_idx]  # this may be shorter than num_cams_so_far

        # only iterate over the cameras that actually exist in this feature list
        for cam_idx in range(len(feats) - 1, -1, -1):
            kp_idx = feats[cam_idx]
            if kp_idx is not None:
                # safe: cam_idx < num_cams_so_far because feats was built earlier
                desc_cam = descriptors_list[cam_idx]
                point_descriptors.append(desc_cam[kp_idx])
                point_indices.append(p_idx)
                break  # stop at the last valid observation

    if len(point_descriptors) == 0:
        raise RuntimeError("No 3D points have valid descriptors to match against.")

    point_descriptors = np.asarray(point_descriptors, dtype=np.float32)

    # Match 3D-point descriptors to new image descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(point_descriptors, desc_new, k=2)

    good_obj_points = []
    good_img_points = []
    good_scene_indices = []
    good_kp_indices = []
    used_scene_indices = set()

    for m, n in matches_knn:
        if m.distance < good_match_ratio * n.distance:
            scene_idx = point_indices[m.queryIdx]
            if scene_idx in used_scene_indices:
                continue  # avoid duplicates
            used_scene_indices.add(scene_idx)

            X = scene_points_3d[scene_idx]
            u, v = kp_new[m.trainIdx].pt

            good_obj_points.append(X)
            good_img_points.append((u, v))
            good_scene_indices.append(scene_idx)
            good_kp_indices.append(m.trainIdx)

    if len(good_obj_points) < 4:
        raise RuntimeError(f"Not enough 2D–3D matches for PnP: {len(good_obj_points)} found.")

    obj_pts = np.asarray(good_obj_points, dtype=np.float32).reshape(-1, 1, 3)
    img_pts = np.asarray(good_img_points, dtype=np.float32).reshape(-1, 1, 2)

    # Solve PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < min_inliers:
        raise RuntimeError(f"PnP failed or too few inliers: {0 if inliers is None else len(inliers)}")

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec  # (3 x 1)

    inlier_mask = np.zeros((len(good_obj_points), 1), dtype=np.uint8)
    inlier_mask[inliers[:, 0]] = 1

    correspondences = {
        "object_points_3d": np.asarray(good_obj_points, dtype=np.float32),
        "image_points_2d": np.asarray(good_img_points, dtype=np.float32),
        "scene_indices": good_scene_indices,
        "keypoint_indices": good_kp_indices
    }

    # Note: camera_poses is updated inside this function
    # The caller should handle adding to camera_poses list

    return R_new, t_new, inlier_mask, kp_new, desc_new, correspondences
