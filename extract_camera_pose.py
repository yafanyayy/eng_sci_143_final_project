''' 
Purpose: Extract the rotation and translation of the first two cameras
Input: 
    - Essential matrix

Output:
    - The first two cameras' rotation and translation 
Unit test(s) and/or other plans to verify and demonstrate correctness:
    - Verify camera orientation relative to April Boards via provided visualization code
    - Reproject aprilboard points back onto board and verify accuracy 
    - Examine reprojection error, visualise it
'''

import cv2
import numpy as np

def extract_camera_pose(essential_matrix):
    ''' Extract camera pose (R, t) from the essential matrix '''

    # Camera 1: (K), R1 = I, t1 = 0 (world coordinates)
    R1 = np.eye(3)
    t = np.zeros((3, 1))

    # Camera 2: (K), R2, t2 from essential matrix -> Decompose using SVD
    U, _, Vt = np.linalg.svd(essential_matrix)

    # Special matrix w
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # two possible rotations for R2
    R2_1 = U @ w @ Vt
    R2_2 = U @ w.T @ Vt 

    # make sure that either possible R2 is a proper rotation (det(R) >= 1)
    if np.linalg.det(R2_1) < 0:
        R2_1 = -R2_1
    if np.linalg.det(R2_2) < 0:
        R2_2 = -R2_2

    # find the translation vector for each of the two R2 options
    t1 = U[:, 2].reshape(3, 1)
    t2 = -U[:, 2].reshape(3, 1)

    # Choose the first of the two options as our rotation and translation for camera 2 
    R2 = R2_1
    t = t1
    
    # check with visualizations whether this is the best option to use 

    return R2, t
