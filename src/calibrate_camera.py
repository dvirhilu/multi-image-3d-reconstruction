'''
Multi-Image 3D Reconstruction
Automatic Camera Calibration

This script, given a set of ~20 images of a chessboard pattern at 
various angles, extracts the distortion parameters of the camera 
lens as well as the intrinsic parameters (focal length, principal 
point, skew coefficient).

The script takes in images from 
../camera_calibration/calib_images/<camera_name>/*
and outputs the camera parameters to 
../camera_calibration/calib_params/<camera_name>.yml

This script contains code from:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://medium.com/@aliyasineser/opencv-camera-calibration-e9a48bdd1844

Author:   Dvir Hilu
Date:     12/10/2020
'''

import numpy as np
import cv2
import glob
import argparse
from utils.file_io_utils import load_calib_images, save_calib_coefficients

parser = argparse.ArgumentParser(description='Camera Calibration Parameter Calculator')
parser.add_argument('-c', '--calib_name', default = "SamsungGalaxyA8", type = str,
        help = "The name of the camera used for calibration. Determines names of images" + 
                "the program will try to pull as well as the name of the file containing calibration data")
parser.add_argument('-s', '--square_size', default = 1.9, type = float,
        help = "The size of a square on the chess board in cm")
parser.add_argument('-w', '--width', default = 6, type = float,
        help = "Number of intersections (x corners) found on the thin side of the board")
parser.add_argument('-l', '--length', default = 9, type = float,
        help = "Number of intersections (x corners) found on the long side of the board")

args = parser.parse_args()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(calib_name, square_size, width, height):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = load_calib_images(calib_name)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate(args.calib_name, args.square_size, args.width, args.length)
    save_calib_coefficients(mtx, dist, args.calib_name)
    print("Calibration is finished. RMS: ", ret)