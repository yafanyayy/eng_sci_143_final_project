from camera_calib import calibrate_camera
# from detect_and_match_keypoints import detect_and_match_keypoints
# from estimate_essential_matrix import estimate_essential_matrix
# from extract_camera_pose import extract_camera_pose
# from estimate_new_camera_pose import estimate_new_camera_pose
# from triangulation import triangulation
# from visualize_scene_points import visualize_scene_points
# from bundle_adjustment import bundle_adjustment

if __name__ == "__main__":
    # Run camera calibration
    K, distCoeffs = calibrate_camera()