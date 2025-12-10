""" 
estimate_essential_matrix (matched_keypoints)
Purpose: Estimate the essential matrix from a set of matched keypoints extracted from two well-matching images, with RANSAC for robustness
Input: Matched keypoints from detect_and_match_keypoints.py
Output: Essential Matrix

Unit test(s) and/or other plans to verify and demonstrate correctness:
    - Determine a threshold to verify that at least a certain percentage of points are inliers 
"""

import cv2
import numpy as np

def estimate_essential_matrix(matched_keypoints, keypoints_list, camera_matrix):
    """ Estimate the essential matrix using RANSAC to handle outliers """
    essential_matrices = []
    ransac_masks = []

    for i in range(len(matched_keypoints)):
        kp1 = keypoints_list[i]
        kp2 = keypoints_list[i + 1]
        matches = matched_keypoints[i]

        # convert matches to Nx2 arrays of points
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate the essential matrix using RANSAC
        essential_M, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        essential_matrices.append(essential_M)
        ransac_masks.append(mask)

    return essential_matrices, ransac_masks