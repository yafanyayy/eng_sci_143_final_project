""" 
detect_and_match_keypoints(img_grayscales)
Purpose: Use the grayscaled images to detect the interest points using SIFT and return them as a list of keypoints for each image, which will be used for matching. 
Input: Grayscale set of images
Output: Matched keypoints

Unit test(s) and/or other plans to verify and demonstrate correctness:
    - Verify good matches via visualization code connecting matched keypoints
    - Adjust good_percentage for acceptable matches 
"""

import cv2
import numpy as np

# make sure to have grayscale images previously calculated taken in as input 

def detect_and_match_keypoints(img_grayscales):
    # Initialize SIFT object
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    # Detect keypoints and compute descriptors for each image
    for img in img_grayscales:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Match keypoints between consecutive images
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matched_keypoints = []

    good_percentage = 0.75  # Ratio test threshold (we can modify to get best results)

    for i in range(len(descriptors_list) - 1):
        matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)

        # Apply good percentage to select good matches
        good_matches = []
        for m, n in matches:
            if m.distance < good_percentage * n.distance:
                good_matches.append(m)

        matched_keypoints.append(good_matches)

        # do we want to return the keypoints_list as well?

    return matched_keypoints