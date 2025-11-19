"""
Camera calibration
Input:images
Output: calibration matrix and distortion coefficients
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
from glob import glob
import os
import pickle
from es143_utils import detect_aprilboard


def read_image_heic(fname):
    """
    Read HEIC image file. Falls back to OpenCV if not HEIC.
    
    Args:
        fname: Path to image file
        
    Returns:
        Image array (BGR format for OpenCV)
    """
    ext = os.path.splitext(fname)[1].lower()
    
    if ext == '.heic' or ext == '.heif':
        try:
            from PIL import Image
            import pillow_heif
            pillow_heif.register_heif_opener()
            
            img = Image.open(fname)
            # Convert PIL image to numpy array (RGB)
            img_array = np.array(img)
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        except ImportError:
            raise ImportError("pillow-heif is required to read HEIC files. Install with: pip install pillow-heif")
        except Exception as e:
            raise ValueError(f"Error reading HEIC file {fname}: {e}")
    else:
        # Use OpenCV for other formats
        return cv2.imread(fname)


def calibrate_camera(image_paths='./camera_calib_raw/*.HEIC', min_detections=30):
    """
    Calibrate camera using AprilTag board images (board 3 - at_board_c).
    
    Args:
        image_paths: List of image file paths or glob pattern string (default: './camera_calib_raw/*.HEIC')
        min_detections: Minimum number of detected tags per image to use for calibration (default: 30)
        
    Returns:
        calMatrix: 3x3 camera calibration matrix (K)
        distCoeffs: Distortion coefficients
    """

    
    # Load board 3 (at_board_c) from pickle file
    with open('AprilBoards2.pickle', 'rb') as f:
        data = pickle.load(f)
    board = data['at_board_c']
    
    # Handle glob pattern or list of paths
    if isinstance(image_paths, str):
        images = sorted(glob(image_paths))
    else:
        images = sorted(image_paths)
    
    assert images, f"No calibration images found: {image_paths}"
    
    print(f"{len(images)} images found")
    
    # Set up AprilTag detector
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    
    # Initialize 3D object points and 2D image points
    calObjPoints = []
    calImgPoints = []
    total_valid = 0
    image_shape = None
    
    # Loop through images
    for count, fname in enumerate(images):
        # Read image and convert to grayscale if necessary
        orig = read_image_heic(fname)
        if orig is None:
            print(f"Warning: Could not read {fname}, skipping...")
            continue
            
        if len(orig.shape) == 3:
            img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        else:
            img = orig
        
        if image_shape is None:
            image_shape = img.shape
        
        # Detect apriltags
        imgpoints, objpoints, tagIDs = detect_aprilboard(img, board, at_detector)
        
        print(f"{count} {fname}: {len(imgpoints)} imgpts, {len(objpoints)} objpts")
        
        # Append detections if enough are found
        if len(imgpoints) >= min_detections and len(objpoints) >= min_detections:
            total_valid += 1
            calObjPoints.append(objpoints.astype('float32'))
            calImgPoints.append(imgpoints.astype('float32'))
    
    assert total_valid > 0, f"No images with at least {min_detections} detections found"
    
    # Calibrate the camera
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = cv2.calibrateCamera(
        calObjPoints,
        calImgPoints,
        image_shape,    # uses image H,W to initialize the principal point to (H/2,W/2)
        None,           # no initial guess for the remaining entries of calMatrix
        None,           # initial guesses for distortion coefficients are all 0
        flags=None)     # default constraints (see documentation)
    
    # Print results
    np.set_printoptions(precision=5, suppress=True)
    print(f'\nRMSE of reprojected points: {reprojerr}')
    print(f'Total images used for calibration: {total_valid}')
    
    np.set_printoptions(precision=2, suppress=True)
    print('\nIntrinsic camera matrix (K):')
    print(calMatrix)
    
    np.set_printoptions(precision=5, suppress=True)
    print('\nDistortion coefficients:')
    print(distCoeffs)
    
    return calMatrix, distCoeffs

