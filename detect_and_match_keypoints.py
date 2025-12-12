""" 
detect_and_match_keypoints(img_grayscales, good_match_percentage=0.1)
Purpose: Use the grayscaled images to detect the interest points using SIFT and return them as a list of keypoints for each image, which will be used for matching. 
Input: Grayscale set of images, percentage of best matches to keep
Output: Dictionary with keypoints_list, descriptors_list, and pair_matches

Unit test(s) and/or other plans to verify and demonstrate correctness:
    - Verify good matches via visualization code connecting matched keypoints
    - Adjust good_percentage for acceptable matches 
"""

import cv2
import numpy as np


def find_matches(keypoints1, keypoints2, descriptors1, descriptors2, percentage):
    """
    Find interest-point matches between two lists of interest point descriptors.

    Args
    ----
      keypoints1     : list of cv2.KeyPoint from Image 1
      keypoints2     : list of cv2.KeyPoint from Image 2
      descriptors1   : descriptor array from Image 1 (N1 x D)
      descriptors2   : descriptor array from Image 2 (N2 x D)
      percentage     : float in [0,1], fraction of best matches to keep

    Returns
    -------
      good_matches : list[cv2.DMatch]
          Sorted by increasing distance (best first).
          For each m in good_matches:
              keypoints1[m.queryIdx]  → point in Image 1
              keypoints2[m.trainIdx]  → point in Image 2
      X1 : (M x 2) numpy array of (x,y) points from Image 1 (only good matches)
      X2 : (M x 2) numpy array of (x,y) points from Image 2 (only good matches)
    """

    # If no descriptors, return empty
    if descriptors1 is None or descriptors2 is None:
        return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Brute force matcher with L2 norm and cross-check
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors between images
    matches12 = bf.match(descriptors1, descriptors2)
    num_matches12 = len(matches12)

    if num_matches12 == 0:
        return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Sort matches by distance (smaller distance = better match)
    matches12 = sorted(matches12, key=lambda m: m.distance)

    # Clamp percentage to [0, 1]
    percentage = float(max(0.0, min(1.0, percentage)))
    num_good = max(1, int(percentage * num_matches12))

    # Keep the top 'percentage' of matches
    good_matches = matches12[:num_good]

    # Extract coordinates only for the good matches
    X1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    X2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    return good_matches, X1, X2


def detect_and_match_keypoints(img_grayscales, good_match_percentage=0.1):
    """
    Detect SIFT keypoints on all grayscale images and match keypoints between
    consecutive image pairs.

    Args
    ----
      img_grayscales       : list of grayscale images (numpy arrays)
      good_match_percentage: float in [0,1], fraction of best matches per pair

    Returns
    -------
      result : dict with keys
        - "keypoints_list"   : list where keypoints_list[i] is list of cv2.KeyPoint for image i
        - "descriptors_list" : list where descriptors_list[i] is descriptor array for image i
        - "pair_matches"     : dict keyed by (i, j) where j = i+1, e.g. (0,1), (1,2), ...
                               each entry is itself a dict:
                               {
                                 "matches" : list[cv2.DMatch],
                                 "points1" : (N x 2) array of points in image i,
                                 "points2" : (N x 2) array of points in image j
                               }
    """

    sift = cv2.SIFT_create()

    keypoints_list = []
    descriptors_list = []

    # Detect keypoints and descriptors for each image
    for img in img_grayscales:
        kps, desc = sift.detectAndCompute(img, None)
        keypoints_list.append(kps)
        descriptors_list.append(desc)

    # Match between consecutive image pairs
    pair_matches = {}
    num_images = len(img_grayscales)
    print(f"Processing {num_images} images")

    # Match all consecutive pairs
    for i in range(num_images - 1):
        kps1 = keypoints_list[i]
        kps2 = keypoints_list[i + 1]
        desc1 = descriptors_list[i]
        desc2 = descriptors_list[i + 1]

        matches, X1, X2 = find_matches(kps1, kps2, desc1, desc2, good_match_percentage)

        pair_matches[(i, i + 1)] = {
            "matches": matches,  # list of cv2.DMatch
            "points1": X1,       # Nx2 coords in image i
            "points2": X2        # Nx2 coords in image i+1
        }
        print(f"finished img {i}")

    result = {
        "keypoints_list": keypoints_list,
        "descriptors_list": descriptors_list,
        "pair_matches": pair_matches
    }
    return result
