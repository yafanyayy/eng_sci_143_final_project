""" 
estimate_essential_and_pose_first_pair(pair_matches, camera_matrix, pair_key=(0, 1))
Purpose: Estimate the essential matrix from a set of matched keypoints extracted from two well-matching images, with RANSAC for robustness. Also recover the relative pose (R, t) between the first two cameras.
Input: pair_matches dictionary, camera_matrix (3x3), pair_key tuple (default (0,1))
Output: Essential matrix E, rotation R, translation t, and inlier mask

Unit test(s) and/or other plans to verify and demonstrate correctness:
    - Determine a threshold to verify that at least a certain percentage of points are inliers 
"""

import cv2
import numpy as np


def estimate_essential_and_pose_first_pair(pair_matches, camera_matrix, pair_key=(0, 1)):
    """
    Estimate the essential matrix and relative pose (R, t) between
    the first two cameras (by default pair (0,1)) using RANSAC.

    Args
    ----
      pair_matches : dict
          From detect_and_match_keypoints()['pair_matches'].
          Must contain pair_matches[(0,1)] with:
              {
                "matches": list[cv2.DMatch],
                "points1": (N x 2) array of points in image 0,
                "points2": (N x 2) array of points in image 1
              }

      camera_matrix : (3 x 3) numpy array
          Intrinsic matrix K.

      pair_key : tuple (i, j)
          Which pair to use (default (0,1)).

    Returns
    -------
      E : (3 x 3) essential matrix
      R : (3 x 3) rotation from camera i to camera j
      t : (3 x 1) translation (up to scale) from camera i to camera j
      mask : (N x 1) uint8 inlier mask from RANSAC
    """
    i, j = pair_key
    if pair_key not in pair_matches:
        raise ValueError(f"pair_matches does not contain key {pair_key}")

    entry = pair_matches[pair_key]
    points1 = entry["points1"]  # Nx2 in image i
    points2 = entry["points2"]  # Nx2 in image j

    if points1.shape[0] < 5:
        raise ValueError(f"Not enough matches ({points1.shape[0]}) to estimate essential matrix.")

    # Estimate essential matrix with RANSAC
    E, mask = cv2.findEssentialMat(
        points1,
        points2,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise RuntimeError("findEssentialMat failed to compute a valid essential matrix.")

    # Recover relative pose between the two cameras
    #    (camera i is reference; result is pose of camera j)
    _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, camera_matrix, mask=mask)

    # mask_pose may refine the inliers; you can use it instead of mask if you want
    inlier_ratio = float(mask_pose.sum()) / float(mask_pose.size)
    print(f"Pair ({i},{j}): recovered pose, inlier ratio = {inlier_ratio:.2f}")

    return E, R, t, mask_pose
