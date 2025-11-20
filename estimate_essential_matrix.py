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

def estimate_essential_matrix(matched_keypoints, keypoints_list1, keypoints_list2, camera_matrix):
    """ Estimate the essential matrix using RANSAC to handle outliers """
    return None